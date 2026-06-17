from __future__ import annotations

"""Agente estratégico calibrado para o jogo Nota Secreta.

A estratégia é deliberadamente híbrida:
- a LLM central é usada para decisões semânticas e geração de dica;
- heurísticas locais validam, desempatam e fazem fallback;
- todas as tools preservam a interface exigida pelo Game Master.

A implementação evita depender de ids, títulos específicos ou da base local:
usa apenas título e letra truncada recebidos durante a partida. Nada além de
heurística local determinística + a LLM central é usado (sem embeddings, sem
dependências novas).

Duas decisões da estratégia vêm direto da regra de pontuação do Game Master
(``_apply_scoring``):

1) Narrador — calibração de dificuldade em BANDA (zona Dixit).
   O narrador só pontua (+3) quando *alguns mas não todos* os adversários
   acertam a carta. Tanto a dica vaga demais (ninguém acerta) quanto a óbvia
   demais (todos acertam) zeram o narrador e ainda dão +2 aos demais.
   Classificamos a dica usando as cartas da própria mão como iscas
   (``_classify_clue_difficulty`` -> "vague" | "calibrated" | "obvious") e,
   fora da banda, fazemos UMA tentativa corretiva de geração (mais direta se
   vaga, mais oblíqua se óbvia). Persistindo fora da banda, caímos no fallback
   temático. Limite rígido de 1 chamada extra por rodada para não estourar o
   ``a2a_timeout`` do Game Master.

   ATENÇÃO (limitação deliberada): o sinal é overlap lexical
   (``_semantic_score``) medido sobre a *própria* mão. É um guard-rail contra
   os extremos (cópia literal de verso vs. dica genérica), não um medidor fino
   da dificuldade percebida pelos adversários — que seguram outras cartas,
   invisíveis ao narrador. ``margem_min``/``margem_max`` são empíricos.

2) Não-narrador — blefe explícito em ``select_card_by_clue``.
   O bônus por votos recebidos na própria carta é pago FORA do if/else de
   pontuação: é o único canal de pontos incondicional do jogo (até +3 por
   rodada). Mantemos o merge LLM+heurística, mas o desempate prefere a carta de
   match semântico mais forte (maior chance de ser confundida com a do
   narrador) em vez do menor índice.

Memória entre rodadas fica fora de escopo: o Game Master não devolve ao agente
quem narrou, qual carta venceu nem o placar, então modelagem de oponente é
impossível neste protocolo.
"""

import argparse
import logging
import re
import time
from typing import Any, Dict, List, Sequence

from base_agent import BaseAgent
from fasta2a import A2AApp, tool

app = A2AApp(name="LLMAgent")

LOGGER = logging.getLogger(__name__)

# Saída degenerada conhecida do serviço LLM em modo mock.
_MOCK_SENTINELS = {"memória tempo cidade", "memoria tempo cidade"}

# Defaults empíricos da banda de dificuldade do narrador.
DEFAULT_MARGEM_MIN = 0.5
DEFAULT_MARGEM_MAX = 4.0

# Orçamento de tempo (s) para a 1ª geração da dica. Se ela já demorou mais que
# isto, a 2ª chamada (correção) é pulada e caímos na heurística — assim send_clue
# nunca encosta no a2a_timeout=90s do Game Master. Medido no torneio: cada
# geração na CPU custa ~40-56s, então este orçamento desliga a correção na CPU
# (1 chamada ~50s, seguro) e a mantém só em hardware rápido (GPU, ~3s/chamada).
DEFAULT_CLUE_CALL_BUDGET = 18.0


class LLMAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        llm_url: str,
        margem_min: float = DEFAULT_MARGEM_MIN,
        margem_max: float = DEFAULT_MARGEM_MAX,
        clue_call_budget: float = DEFAULT_CLUE_CALL_BUDGET,
    ):
        super().__init__(name=name, llm_url=llm_url, request_timeout=60.0)
        self.last_narrator_card: Dict[str, Any] | None = None
        self.round_memory: List[Dict[str, Any]] = []
        self.margem_min = margem_min
        self.margem_max = margem_max
        self.clue_call_budget = clue_call_budget

    @tool()
    async def receive_hand(self, hand: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.hand = list(hand)
        return {"status": "ok", "hand_size": len(self.hand)}

    @tool()
    async def choose_card(self) -> Dict[str, Any]:
        """Escolhe uma carta boa para narrar.

        A carta ideal deve permitir uma dica associativa: nem literal demais,
        nem tão genérica que todos errem ou todos acertem.
        """
        if not self.hand:
            raise RuntimeError("Hand is empty")

        heuristic_order = self._rank_cards_for_narration(self.hand)
        llm_order = await self._llm_rank_narrator_cards(self.hand)
        chosen_idx = self._merge_rankings(heuristic_order, llm_order, len(self.hand))[0]

        chosen = self.hand[chosen_idx]
        self.last_narrator_card = chosen
        LOGGER.info("[%s] Carta narradora escolhida: %s", self.name, chosen.get("title", ""))
        return {"chosen_card": chosen}

    # ------------------------------------------------------------------
    # Narrador: dica calibrada na banda Dixit
    # ------------------------------------------------------------------

    @tool()
    async def send_clue(self, lyrics: str, max_words: int = 6) -> Dict[str, Any]:
        """Gera uma dica calibrada: nem vaga (ninguém acha) nem óbvia (todos acham)."""
        title = ""
        if self.last_narrator_card:
            title = str(self.last_narrator_card.get("title", ""))

        prompt = self._build_clue_prompt(lyrics=lyrics, title=title, max_words=max_words)
        start = time.monotonic()
        raw = await self.llm_generate(
            prompt,
            max_tokens=28,
            temperature=0.55,
            stop=["\n\n", "\nResposta:", "\nAnswer:", "###"],
        )
        first_call_elapsed = time.monotonic() - start
        clue = self._clean_clue(raw, lyrics=lyrics, title=title, max_words=max_words)
        clue = await self._calibrate_clue(
            clue, lyrics=lyrics, title=title, max_words=max_words,
            first_call_elapsed=first_call_elapsed,
        )

        self.clue_history.append(clue)
        self.round_memory.append({"role": "narrator", "title": title, "clue": clue})
        LOGGER.info("[%s] Dica calibrada: %s", self.name, clue)
        return {"clue": clue}

    async def _calibrate_clue(
        self,
        clue: str,
        lyrics: str,
        title: str,
        max_words: int,
        first_call_elapsed: float = 0.0,
    ) -> str:
        """Aplica a banda de dificuldade com no máximo UMA correção via LLM."""
        if self._is_degenerate_clue(clue):
            return self._thematic_fallback(lyrics, title, max_words)

        difficulty = self._difficulty_of(clue)
        if difficulty == "calibrated":
            return clue

        # A correção custa OUTRA chamada à LLM. Se a 1ª já foi lenta, pular a
        # correção e usar a heurística — evita estourar o a2a_timeout do GM e
        # honra o princípio "se demorou, melhor a heurística".
        if first_call_elapsed > self.clue_call_budget:
            LOGGER.info(
                "[%s] 1ª dica levou %.1fs (> %.1fs); pulando correção e usando heurística",
                self.name, first_call_elapsed, self.clue_call_budget,
            )
            return self._thematic_fallback(lyrics, title, max_words)

        # Fora da banda e ainda com tempo: uma única tentativa corretiva.
        direction = "direct" if difficulty == "vague" else "oblique"
        corrected = await self._regenerate_clue(lyrics, title, max_words, direction)
        if (
            corrected
            and not self._is_degenerate_clue(corrected)
            and self._difficulty_of(corrected) == "calibrated"
        ):
            return corrected

        # LLM não cooperou após a correção -> fallback temático (nunca joga a rodada fora).
        return self._thematic_fallback(lyrics, title, max_words)

    def _difficulty_of(self, clue: str) -> str:
        """Classifica a dica usando a mão atual e a carta-alvo do narrador."""
        if not self.hand or self.last_narrator_card is None:
            return "calibrated"
        return self._classify_clue_difficulty(clue, self.hand, self.last_narrator_card.get("id"))

    def _classify_clue_difficulty(
        self,
        clue: str,
        cards: Sequence[Dict[str, Any]],
        target_id: Any,
    ) -> str:
        """Banda de dificuldade usando as cartas da mão como iscas.

        Guard-rail contra extremos (não um medidor fino): pontua cada carta da
        mão com ``_semantic_score`` (overlap lexical) e devolve:

        - "vague"      -> alvo não fica em 1º, ou margem para a 2ª < margem_min;
        - "obvious"    -> margem para a 2ª > margem_max;
        - "calibrated" -> alvo em 1º e margem dentro de [margem_min, margem_max].
        """
        scored = sorted(
            ((self._semantic_score(card, clue), card.get("id")) for card in cards),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not scored:
            return "vague"

        best_score, best_id = scored[0]
        if best_score <= 0.0 or best_id != target_id:
            return "vague"

        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score
        if margin < self.margem_min:
            return "vague"
        if margin > self.margem_max:
            return "obvious"
        return "calibrated"

    async def _regenerate_clue(self, lyrics: str, title: str, max_words: int, direction: str) -> str:
        prompt = self._build_clue_correction_prompt(lyrics, title, max_words, direction)
        raw = await self.llm_generate(
            prompt,
            max_tokens=28,
            temperature=0.6,
            stop=["\n\n", "\nResposta:", "\nAnswer:", "###"],
        )
        return self._clean_clue(raw, lyrics=lyrics, title=title, max_words=max_words)

    def _build_clue_correction_prompt(
        self,
        lyrics: str,
        title: str,
        max_words: int,
        direction: str,
    ) -> str:
        short_lyrics = " ".join(lyrics.split()[:80])
        title_rule = f"\nNão use palavras do título: {title}." if title else ""
        if direction == "direct":
            adjust = (
                "A dica anterior ficou vaga demais: ninguém acharia a música.\n"
                "Crie uma dica mais concreta, com uma imagem ou tema mais nítido da letra,\n"
                "mas ainda sem copiar verso literal nem citar o título."
            )
        else:
            adjust = (
                "A dica anterior ficou óbvia demais: todos achariam a música.\n"
                "Crie uma dica mais oblíqua e poética, sugerindo o clima de longe,\n"
                "sem palavras diretas da letra nem do título."
            )
        return (
            "Refaça uma dica para Nota Secreta (jogo tipo Dixit com músicas brasileiras).\n"
            f"A dica deve ter de 2 a {max_words} palavras.\n"
            f"{adjust}"
            f"{title_rule}\n\n"
            f"Letra truncada:\n{short_lyrics}\n\n"
            "Dica:"
        )

    def _thematic_fallback(self, lyrics: str, title: str, max_words: int) -> str:
        fallback = self._fallback_thematic_clue(lyrics, title, max_words=max_words)
        fallback = self._remove_title_words(fallback, title, max_words=max_words)
        fallback = " ".join(fallback.split()[:max_words]).strip()
        if not fallback or self._is_degenerate_clue(fallback):
            return "memória em trânsito"
        return fallback

    # ------------------------------------------------------------------
    # Não-narrador: blefe (maximizar votos recebidos) e voto
    # ------------------------------------------------------------------

    @tool()
    async def select_card_by_clue(self, clue: str) -> Dict[str, Any]:
        """Escolhe a carta que maximiza o blefe (votos recebidos na própria carta)."""
        if not self.hand:
            raise RuntimeError("Hand is empty")

        heuristic_order = self._rank_cards_for_clue(clue, self.hand)
        llm_order = await self._llm_rank_cards_by_clue(clue, self.hand, purpose="selecionar")
        # Desempate explícito de blefe: preferir o match semântico mais forte,
        # que tem maior chance de ser confundido com a carta do narrador.
        tie_break = {idx: self._semantic_score(card, clue) for idx, card in enumerate(self.hand)}
        chosen_idx = self._merge_rankings(
            heuristic_order, llm_order, len(self.hand), tie_break=tie_break
        )[0]

        chosen = self.hand[chosen_idx]
        self.round_memory.append({"role": "melomano", "clue": clue, "played": chosen.get("title", "")})
        LOGGER.info("[%s] Blefe pela dica '%s': %s", self.name, clue, chosen.get("title", ""))
        return {"chosen_card": chosen}

    @tool()
    async def vote(self, clue: str, options: List[Dict[str, Any]], my_chosen_card: Dict[str, Any]) -> Dict[str, Any]:
        """Vota nas duas opções mais prováveis de serem a carta do narrador."""
        if len(options) < 3:
            return {"votes": []}

        my_idx = self._find_own_option(options, my_chosen_card)
        allowed = [idx for idx in range(len(options)) if idx != my_idx]

        heuristic_order = self._rank_cards_for_clue(clue, options, forbidden_idx=my_idx)
        llm_order = await self._llm_rank_cards_by_clue(
            clue,
            options,
            purpose="votar",
            forbidden_idx=my_idx,
        )
        ranked = self._merge_rankings(heuristic_order, llm_order, len(options), forbidden_idx=my_idx)

        votes: List[int] = []
        for idx in ranked:
            if idx in allowed and idx not in votes:
                votes.append(idx)
            if len(votes) == 2:
                break

        if len(votes) < 2:
            for idx in allowed:
                if idx not in votes:
                    votes.append(idx)
                if len(votes) == 2:
                    break

        self.vote_history.extend(votes)
        LOGGER.info("[%s] Votos para dica '%s': %s", self.name, clue, votes[:2])
        return {"votes": votes[:2]}

    # ------------------------------------------------------------------
    # Prompts e chamadas semânticas à LLM
    # ------------------------------------------------------------------

    async def _llm_rank_narrator_cards(self, cards: Sequence[Dict[str, Any]]) -> List[int]:
        options = "\n".join(self._card_prompt_line(card, idx) for idx, card in enumerate(cards))
        prompt = (
            "Você joga Nota Secreta, parecido com Dixit usando músicas brasileiras.\n"
            "Escolha a melhor carta para ser narrador: deve permitir uma dica curta, poética, "
            "sem citar título/letra literalmente, e com dificuldade média.\n"
            "Responda somente JSON no formato {\"ranking\":[0,1,2,3]}.\n\n"
            f"Cartas:\n{options}\n"
        )
        return await self._rank_from_llm(prompt, len(cards))

    async def _llm_rank_cards_by_clue(
        self,
        clue: str,
        cards: Sequence[Dict[str, Any]],
        purpose: str,
        forbidden_idx: int | None = None,
    ) -> List[int]:
        options = "\n".join(self._card_prompt_line(card, idx) for idx, card in enumerate(cards))
        restriction = ""
        if forbidden_idx is not None:
            restriction = f"\nNunca escolha a opção {forbidden_idx}, pois é a sua própria carta."

        prompt = (
            "Você joga Nota Secreta com letras de música.\n"
            f"Tarefa: {purpose} a carta mais relacionada à dica, usando título, temas e palavras-chave.\n"
            "Priorize relação semântica, não apenas palavra idêntica. Responda somente JSON "
            "no formato {\"ranking\":[indices em ordem]}."
            f"{restriction}\n\n"
            f"Dica: {clue}\n\n"
            f"Opções:\n{options}\n"
        )
        return await self._rank_from_llm(prompt, len(cards), forbidden_idx=forbidden_idx)

    async def _rank_from_llm(
        self,
        prompt: str,
        n_options: int,
        forbidden_idx: int | None = None,
    ) -> List[int]:
        raw = await self.llm_generate(
            prompt,
            max_tokens=45,
            temperature=0.15,
            stop=["\n\n", "###"],
        )

        obj = self._extract_json_object(raw)
        if obj:
            ranking = self._parse_ranking(obj, n_options)
            ranking = [idx for idx in ranking if idx != forbidden_idx]
            if ranking:
                return ranking

            scored = self._parse_score_map(obj, n_options, forbidden_idx=forbidden_idx)
            if scored:
                return scored

        parsed = self._parse_score_map_from_text(raw, n_options, forbidden_idx=forbidden_idx)
        if parsed:
            return parsed

        parsed = self._parse_loose_ranking(raw, n_options, forbidden_idx=forbidden_idx)
        return parsed

    def _build_clue_prompt(self, lyrics: str, title: str, max_words: int) -> str:
        short_lyrics = " ".join(lyrics.split()[:80])
        title_rule = f"\nNão use palavras do título: {title}." if title else ""
        return (
            "Crie uma dica para Nota Secreta, jogo parecido com Dixit usando músicas brasileiras.\n"
            f"A dica deve ter de 2 a {max_words} palavras.\n"
            "Ela deve sugerir tema, clima ou imagem da música, mas não pode copiar verso literal.\n"
            "Evite nomes próprios, título da música, artista e palavras óbvias repetidas na letra.\n"
            "Boa dica: relacionada, criativa e de dificuldade média.\n"
            "Responda apenas com a dica, sem explicação."
            f"{title_rule}\n\n"
            f"Letra truncada:\n{short_lyrics}\n\n"
            "Dica:"
        )

    # ------------------------------------------------------------------
    # Heurísticas de jogo
    # ------------------------------------------------------------------

    def _rank_cards_for_narration(self, cards: Sequence[Dict[str, Any]]) -> List[int]:
        scored = []
        for idx, card in enumerate(cards):
            lyrics = str(card.get("lyrics", ""))
            keywords = self._extract_keywords(lyrics)
            unique_ratio = len(set(keywords)) / max(1, len(keywords))
            word_count = len(lyrics.split())
            length_balance = 1.0 - min(1.0, abs(word_count - 55) / 55)
            title_words = set(self._extract_keywords(str(card.get("title", ""))))
            lyric_words = set(keywords)

            overly_literal = len(title_words & lyric_words) / max(1, len(title_words))
            theme_strength = min(1.0, len(set(keywords[:12])) / 8)
            score = (1.25 * theme_strength) + (0.9 * unique_ratio) + (0.7 * length_balance) - (0.45 * overly_literal)
            scored.append((score, idx))

        scored.sort(reverse=True)
        return [idx for _, idx in scored]

    def _rank_cards_for_clue(
        self,
        clue: str,
        cards: Sequence[Dict[str, Any]],
        forbidden_idx: int | None = None,
    ) -> List[int]:
        scored = []
        for idx, card in enumerate(cards):
            if idx == forbidden_idx:
                continue
            score = self._semantic_score(card, clue)
            scored.append((score, idx))

        scored.sort(reverse=True)
        return [idx for _, idx in scored]

    def _semantic_score(self, card: Dict[str, Any], clue: str) -> float:
        clue_terms = set(self._extract_keywords(clue))
        title_terms = set(self._extract_keywords(str(card.get("title", ""))))
        lyric_terms = set(self._extract_keywords(str(card.get("lyrics", "")))[:30])

        if not clue_terms:
            return 0.0

        title_overlap = len(clue_terms & title_terms)
        lyric_overlap = len(clue_terms & lyric_terms)
        phrase_bonus = 0.0
        normalized_clue = self._normalize_text_for_match(clue)
        normalized_title = self._normalize_text_for_match(str(card.get("title", "")))
        normalized_lyrics = self._normalize_text_for_match(str(card.get("lyrics", "")))

        if normalized_clue and normalized_clue in normalized_title:
            phrase_bonus += 1.4
        if normalized_clue and normalized_clue in normalized_lyrics:
            phrase_bonus += 0.8

        coverage = (title_overlap + lyric_overlap) / max(1, len(clue_terms))
        return (2.2 * title_overlap) + (1.0 * lyric_overlap) + (1.5 * coverage) + phrase_bonus

    def _merge_rankings(
        self,
        heuristic_order: Sequence[int],
        llm_order: Sequence[int],
        n_options: int,
        forbidden_idx: int | None = None,
        tie_break: Dict[int, float] | None = None,
    ) -> List[int]:
        """Borda count LLM+heurística com desempate opcional por ``tie_break``.

        Sem ``tie_break`` o empate é resolvido pelo menor índice (usado por
        choose_card e vote); com ``tie_break``, o maior valor vence (blefe).
        """
        candidates = [idx for idx in range(n_options) if idx != forbidden_idx]
        scores = {idx: 0.0 for idx in candidates}

        for weight, ranking in ((1.0, heuristic_order), (1.35, llm_order)):
            for pos, idx in enumerate(ranking):
                if idx in scores:
                    scores[idx] += weight * (n_options - pos)

        # Garante presença de todos os candidatos em caso de ranking incompleto.
        for pos, idx in enumerate(heuristic_order):
            if idx in scores:
                scores[idx] += 0.05 * (n_options - pos)

        tb = tie_break or {}
        return sorted(
            candidates,
            key=lambda idx: (scores[idx], tb.get(idx, 0.0), -idx),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Sanitização, parsing e utilidades
    # ------------------------------------------------------------------

    def _clean_clue(self, raw: str, lyrics: str, title: str, max_words: int) -> str:
        clue = self._sanitize_clue(raw.strip(), max_words=max_words, lyrics=lyrics)
        clue = self._remove_title_words(clue, title, max_words=max_words)

        if self._is_bad_clue(clue, lyrics=lyrics, title=title):
            clue = self._fallback_thematic_clue(lyrics, title, max_words=max_words)

        clue = self._remove_title_words(clue, title, max_words=max_words)
        if self._is_bad_clue(clue, lyrics=lyrics, title=title):
            clue = "memória em trânsito"

        return " ".join(clue.split()[:max_words]).strip()

    def _is_degenerate_clue(self, clue: str) -> bool:
        """Detecta dica vazia/pobre ou a saída fixa do mock do serviço LLM."""
        norm = self._normalize_text_for_match(clue)
        if not norm:
            return True
        if norm in _MOCK_SENTINELS:
            return True
        return len(self._extract_keywords(clue)) < 2

    def _remove_title_words(self, clue: str, title: str, max_words: int) -> str:
        if not title:
            return clue

        title_terms = set(self._extract_keywords(title))
        kept = []
        for word in clue.split():
            normalized = self._normalize_text_for_match(word)
            if normalized and normalized not in title_terms:
                kept.append(word)

        return " ".join(kept[:max_words]).strip()

    def _is_bad_clue(self, clue: str, lyrics: str, title: str) -> bool:
        useful = self._extract_keywords(clue)
        if len(useful) < 2:
            return True
        if len(clue.split()) > 6:
            return True
        if self._is_literal_substring_of_lyrics(clue, lyrics):
            return True
        title_terms = set(self._extract_keywords(title))
        return bool(title_terms and title_terms <= set(useful))

    def _fallback_thematic_clue(self, lyrics: str, title: str, max_words: int) -> str:
        text = f"{title} {lyrics}".lower()
        themes = [
            (("amor", "paixão", "beijo", "coração"), "afeto fora de lugar"),
            (("saudade", "lembrança", "memória", "passado"), "memória que insiste"),
            (("cidade", "rua", "avenida", "prédio", "asfalto"), "cidade em movimento"),
            (("mar", "onda", "praia", "barco", "rio"), "horizonte de água"),
            (("noite", "lua", "estrela", "escuro", "madrugada"), "noite em suspenso"),
            (("tempo", "dia", "ano", "hora", "amanhã"), "tempo fora do eixo"),
            (("dor", "triste", "choro", "solidão"), "silêncio depois da queda"),
            (("festa", "dança", "samba", "carnaval"), "corpo em festa"),
            (("liberdade", "voar", "vento", "estrada"), "vontade sem destino"),
        ]

        for markers, clue in themes:
            if any(marker in text for marker in markers):
                return " ".join(clue.split()[:max_words])

        keywords = [word for word in self._extract_keywords(lyrics) if word not in set(self._extract_keywords(title))]
        if len(keywords) >= 2:
            return " ".join([keywords[0], "em", keywords[1]][:max_words])

        return "memória em trânsito"

    def _card_prompt_line(self, card: Dict[str, Any], idx: int) -> str:
        title = str(card.get("title", "")).strip()
        keywords = ", ".join(self._song_keywords(card, limit=8))
        return f"{idx}: título={title}; palavras-chave={keywords}"

    def _find_own_option(self, options: Sequence[Dict[str, Any]], my_chosen_card: Dict[str, Any]) -> int:
        my_id = my_chosen_card.get("id")
        for idx, option in enumerate(options):
            if option.get("id") == my_id:
                return idx
        return -1

    def _parse_loose_ranking(
        self,
        response: str,
        n_options: int,
        forbidden_idx: int | None = None,
    ) -> List[int]:
        out: List[int] = []
        for raw in re.findall(r"\d+", response):
            idx = int(raw)
            # Aceita numeração 1-based: um índice == n_options mapeia para o
            # último (n_options-1). Os demais índices válidos seguem 0-based.
            if idx == n_options:
                idx -= 1
            if 0 <= idx < n_options and idx != forbidden_idx and idx not in out:
                out.append(idx)
        return out

    def _mock_llm_response(self, prompt: str, max_tokens: int = 40) -> str:
        """Fallback local quando a requisição ao serviço LLM falha.

        Para prompts de ranking devolvemos vazio (sem dígitos) de propósito:
        assim a heurística decide sozinha, em vez de a identidade [0,1,2,3]
        enviesar o Borda count para o índice 0.
        """
        if "\"ranking\"" in prompt:
            return "{}"
        return self._fallback_thematic_clue(prompt, "", max_words=min(6, max_tokens))


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("game_master_url")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--llm-url", default="http://127.0.0.1:9000")
    parser.add_argument("--name", default=None)
    parser.add_argument("--margem-min", type=float, default=DEFAULT_MARGEM_MIN)
    parser.add_argument("--margem-max", type=float, default=DEFAULT_MARGEM_MAX)
    parser.add_argument(
        "--clue-call-budget", type=float, default=DEFAULT_CLUE_CALL_BUDGET,
        help="Tempo (s) da 1ª geração acima do qual a correção é pulada. 0 desliga a correção.",
    )
    args = parser.parse_args()

    agent = LLMAgent(
        name=args.name or f"LLMAgent_{args.port}",
        llm_url=args.llm_url,
        margem_min=args.margem_min,
        margem_max=args.margem_max,
        clue_call_budget=args.clue_call_budget,
    )
    app.register(agent)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
