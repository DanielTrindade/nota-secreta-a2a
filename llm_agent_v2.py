from __future__ import annotations

"""Agente estratégico calibrado (v2) para o jogo Nota Secreta.

Variante do ``LLMAgent`` (llm_agent.py) com duas melhorias derivadas direto da
regra de pontuação do Game Master (``_apply_scoring``). O agente antigo
permanece intacto; aqui apenas sobrescrevemos o comportamento que muda, então
as 5 tools A2A mantêm exatamente a mesma assinatura.

1) Narrador — calibração de dificuldade em BANDA (zona Dixit).
   O narrador só pontua (+3) quando *alguns mas não todos* os adversários
   acertam a carta. Tanto a dica vaga demais (ninguém acerta) quanto a óbvia
   demais (todos acertam) zeram o narrador e ainda dão +2 aos demais. Em vez do
   teste binário ``_clue_points_to_card``, classificamos a dica usando as cartas
   da própria mão como iscas:

       _classify_clue_difficulty(clue, cards, target_id)
           -> "vague" | "calibrated" | "obvious"

   A dica é "calibrated" se, e só se, a carta-alvo fica em 1º (pela pontuação
   semântica local) e a margem para a 2ª colocada cai dentro de
   [margem_min, margem_max]. Fora da banda, fazemos UMA tentativa corretiva de
   geração (mais direta se vaga, mais oblíqua se óbvia); persistindo fora da
   banda, caímos no fallback temático já existente. Limite rígido de 1 chamada
   extra por rodada para não estourar o timeout do Game Master.

   ATENÇÃO (limitação deliberada): o sinal é overlap lexical
   (``_semantic_score``) medido sobre a *própria* mão. É um guard-rail contra os
   extremos (cópia literal de verso vs. dica genérica), não um medidor fino da
   dificuldade percebida pelos adversários — que seguram outras cartas,
   invisíveis ao narrador. ``margem_min``/``margem_max`` são empíricos e devem
   ser calibrados com ``ab_test.py``.

2) Não-narrador — blefe explícito em ``select_card_by_clue``.
   O bônus por votos recebidos na própria carta é pago FORA do if/else de
   pontuação: é o único canal de pontos incondicional do jogo (até +3 por
   rodada). Mantemos o merge LLM+heurística, mas o desempate passa a preferir a
   carta de match semântico mais forte (maior chance de ser confundida com a do
   narrador) em vez do menor índice. A seleção continua sendo só sobre o blefe;
   não tentamos "adivinhar" aqui.

Memória entre rodadas continua fora de escopo: o Game Master não devolve ao
agente quem narrou, qual carta venceu nem o placar, então modelagem de oponente
é impossível neste protocolo. Nada além de heurística local determinística + a
LLM central é usado (sem embeddings, sem dependências novas).
"""

import argparse
import logging
from typing import Any, Dict, List, Sequence

from fasta2a import A2AApp, tool
from llm_agent import LLMAgent

app = A2AApp(name="CalibratedLLMAgent")
LOGGER = logging.getLogger(__name__)

# Defaults empíricos da banda de dificuldade. Calibre com ab_test.py.
DEFAULT_MARGEM_MIN = 0.5
DEFAULT_MARGEM_MAX = 4.0


class CalibratedLLMAgent(LLMAgent):
    def __init__(
        self,
        name: str,
        llm_url: str,
        margem_min: float = DEFAULT_MARGEM_MIN,
        margem_max: float = DEFAULT_MARGEM_MAX,
    ):
        super().__init__(name=name, llm_url=llm_url)
        self.margem_min = margem_min
        self.margem_max = margem_max

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
        raw = await self.llm_generate(
            prompt,
            max_tokens=28,
            temperature=0.55,
            stop=["\n\n", "\nResposta:", "\nAnswer:", "###"],
        )
        clue = self._clean_clue(raw, lyrics=lyrics, title=title, max_words=max_words)
        clue = await self._calibrate_clue(clue, lyrics=lyrics, title=title, max_words=max_words)

        self.clue_history.append(clue)
        self.round_memory.append({"role": "narrator", "title": title, "clue": clue})
        LOGGER.info("[%s] Dica calibrada: %s", self.name, clue)
        return {"clue": clue}

    async def _calibrate_clue(self, clue: str, lyrics: str, title: str, max_words: int) -> str:
        """Aplica a banda de dificuldade com no máximo UMA correção via LLM."""
        if self._is_degenerate_clue(clue):
            return self._thematic_fallback(lyrics, title, max_words)

        difficulty = self._difficulty_of(clue)
        if difficulty == "calibrated":
            return clue

        # Fora da banda: uma única tentativa corretiva (limite rígido).
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
    # Não-narrador: blefe (maximizar votos recebidos)
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

    def _merge_rankings(
        self,
        heuristic_order: Sequence[int],
        llm_order: Sequence[int],
        n_options: int,
        forbidden_idx: int | None = None,
        tie_break: Dict[int, float] | None = None,
    ) -> List[int]:
        """Borda count LLM+heurística com desempate opcional por ``tie_break``.

        Sem ``tie_break`` o comportamento é idêntico ao do agente antigo (empate
        resolvido pelo menor índice), preservando choose_card e vote herdados.
        """
        candidates = [idx for idx in range(n_options) if idx != forbidden_idx]
        scores = {idx: 0.0 for idx in candidates}

        for weight, ranking in ((1.0, heuristic_order), (1.35, llm_order)):
            for pos, idx in enumerate(ranking):
                if idx in scores:
                    scores[idx] += weight * (n_options - pos)

        for pos, idx in enumerate(heuristic_order):
            if idx in scores:
                scores[idx] += 0.05 * (n_options - pos)

        tb = tie_break or {}
        return sorted(
            candidates,
            key=lambda idx: (scores[idx], tb.get(idx, 0.0), -idx),
            reverse=True,
        )


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
    args = parser.parse_args()

    agent = CalibratedLLMAgent(
        name=args.name or f"CalibratedLLMAgent_{args.port}",
        llm_url=args.llm_url,
        margem_min=args.margem_min,
        margem_max=args.margem_max,
    )
    app.register(agent)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
