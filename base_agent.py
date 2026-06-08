from __future__ import annotations

"""Classe base dos agentes.
Marco Cristo, 2026 (com ajuda de ChatGPT)

Neste codigo vcs vao achar varias heuristicas que eu usei para 
corrigir problemas das respostas que eu recebia da LLM. Resolvi deixar
aqui apeans para vcs terem ideia de como eu lidava com os varios
bugs que encontrava nas respostas da LLM...

Concentra o cliente REST do LLM, parsing de respostas e heurísticas simples
reaproveitáveis entre o agente estratégico e o baseline aleatório.
"""

import hashlib
import json
import logging
import re
import traceback
from collections import Counter
from typing import Any, Dict, List, Sequence

import aiohttp

LOGGER = logging.getLogger(__name__)

STOPWORDS = {
    "a", "o", "e", "de", "do", "da", "em", "um", "uma", "que", "para",
    "por", "com", "na", "no", "as", "os", "se", "eu", "me", "te", "tu",
    "você", "voce", "nós", "nos", "eles", "elas", "ao", "à", "às", "dos",
    "das", "mais", "menos", "muito", "muita", "ser", "estar", "ter", "sou",
}


class BaseAgent:
    def __init__(self, name: str, llm_url: str = "http://127.0.0.1:9000", request_timeout: float = 60.0):
        self.name = name
        self.llm_url = llm_url.rstrip("/")
        self.request_timeout = request_timeout

        self.hand: List[Dict[str, Any]] = []
        self.vote_history: List[int] = []
        self.clue_history: List[str] = []

        self._llm_cache: Dict[str, str] = {}

    async def llm_generate(
        self,
        prompt: str,
        max_tokens: int = 40,
        temperature: float = 0.2,
        stop: Sequence[str] | None = None,
    ) -> str:
        'Interface com LLM'
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": list(stop or []),
        }
        cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        try:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self.llm_url}/generate", json=payload) as resp:
                    raw = await resp.text()
                    print(f"[{self.name}] LLM status: {resp.status}")
                    print(f"[{self.name}] LLM raw response: {raw[:500]}")
                    resp.raise_for_status()
                    data = await resp.json()
                    text = str(data.get("text", "")).strip()
        except Exception as e:
            print(f"[{self.name}] LLM request failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            text = self._mock_llm_response(prompt, max_tokens=max_tokens)

        self._llm_cache[cache_key] = text
        return text

    async def llm_generate_json(
        self,
        prompt: str,
        max_tokens: int = 40,
        temperature: float = 0.2,
        stop: Sequence[str] | None = None,
    ) -> Dict[str, Any] | None:
        text = await self.llm_generate(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)
        return self._extract_json_object(text)

    def _extract_json_object(self, text: str) -> Dict[str, Any] | None:
        text = text.strip()

        # tentativa direta
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # tenta achar primeiro bloco {...}
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        return None

    def _extract_keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-záàâãéêíóôõúç]+", text.lower())
        filtered = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
        counts = Counter(filtered)
        return [w for w, _ in counts.most_common()]

    def _song_keywords(self, song: Dict[str, Any], limit: int = 5) -> List[str]:
        text = f"{song.get('title', '')} {song.get('lyrics', '')}"
        return self._extract_keywords(text)[:limit]
    
    def _song_brief(self, song: Dict[str, Any], idx: int | None = None) -> str:
        prefix = f"{idx}: " if idx is not None else ""
        title = song.get("title", "")
        keywords = ", ".join(self._song_keywords(song, limit=5))
        return f"{prefix}Título={title} | Palavras-chave={keywords}"
    
    def _fallback_clue_from_lyrics(self, lyrics: str, max_words: int = 6) -> str:
        # Se td deu errado, caiu aqui... entao vai tentar criar uma pista pequena,
        # formada com palavras da propria letra:
        # 2 ou mais palavras (cada uma com mais de 2 letras)
        kws = [w for w in self._extract_keywords(lyrics) if len(w) > 2]

        if len(kws) >= 3:
            return " ".join(kws[: min(3, max_words)])

        if len(kws) == 2:
            return " ".join(kws)

        if len(kws) == 1:
            # so conseguiu 1 palavra? adiciona "distante"
            return f"{kws[0]} distante"

        return "saudade em trânsito" # deu tudo errado? manda uma dica qqualquer, meio vaga
    
    def _normalize_text_for_match(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-záàâãéêíóôõúç0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_literal_substring_of_lyrics(self, clue: str, lyrics: str) -> bool:
        clue_norm = self._normalize_text_for_match(clue)
        lyrics_norm = self._normalize_text_for_match(lyrics)

        if not clue_norm:
            return False

        return clue_norm in lyrics_norm
    
    def _sanitize_clue(self, clue: str, max_words: int, lyrics: str) -> str:
        ''' Estas sanitizacoes foram baseadas na minha experiencia com a Phi3.5
            Servem como ilustracoes para vcs de coisas que podem fazer para lidar
            com as respostas meio malucas que a LLM pdoe fornecer.
            Varias delas tem a ver com os prompts que eu usei e, portanto, nap
            necessariamente fazem sentido pra vcs.
        '''
        clue = clue.replace("\n", " ").strip()

        # remove prefixos no início
        clue = re.sub(
            r"^(dica|clue|resposta|response|answer)\s*:\s*",
            "",
            clue,
            flags=re.IGNORECASE,
        )

        # corta qualquer continuação tipo "Response:", "Resposta:", "Answer:"
        clue = re.split(
            r"\b(?:resposta|response|answer)\s*:",
            clue,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]

        # remove marcadores markdown soltos -- isso deu muito problema na minha experiencia
        clue = re.sub(r"#+", " ", clue)

        # remove aspas e pontuação nas bordas
        clue = clue.strip(" .,:;!-\"'")

        words = clue.split()
        words = words[:max_words]
        clue = " ".join(words).strip(" .,:;!-\"'")

        # remove restos explícitos
        banned = {"answer", "response", "resposta", "clue", "dica"}
        tokens = [w for w in clue.split() if w.lower() not in banned]
        clue = " ".join(tokens[:max_words]).strip()

        # rejeita dica vazia ou pobre demais
        useful_tokens = [w for w in clue.split() if len(w) > 2]
        if len(useful_tokens) < 2:
            clue = ""

        # rejeita se a dica for uma cópia literal de trecho da letra
        # esta eh uma heuristica q vcs podem querer tirar por exemplo
        if clue and self._is_literal_substring_of_lyrics(clue, lyrics):
            clue = ""

        # deu tudo errado? entao fallback!
        if not clue:
            clue = self._fallback_clue_from_lyrics(lyrics, max_words=max_words)

        clue = clue.strip(" .,:;!-\"'“”‘’")

        return clue
    
    def _parse_song_choice(self, response: str, n_options: int) -> int | None:
        nums = [int(x) for x in re.findall(r"\d+", response)]
        for n in nums:
            if 0 <= n < n_options:
                return n
        return None
    
    def _parse_option_from_text(self, response: str, n_options: int) -> int | None:
        # Casso foram baseados em varios problemas que eu tive em testes...
        text = response.strip()

        # caso "4: Título=..."
        m = re.match(r"^\s*(\d+)\s*:", text)
        if m:
            idx = int(m.group(1))
            # a LLM às vezes usa numeração 1-based
            if 0 <= idx < n_options:
                return idx
            if 1 <= idx <= n_options:
                return idx - 1

        # fallback: primeiro número encontrado
        nums = [int(x) for x in re.findall(r"\d+", text)]
        for idx in nums:
            if 0 <= idx < n_options:
                return idx
        for idx in nums:
            if 1 <= idx <= n_options:
                return idx - 1

        return None
    
    def _parse_score_map_from_text(
        self,
        response: str,
        n_options: int,
        forbidden_idx: int | None = None,
    ) -> List[int]:
        parsed: List[tuple[float, int]] = []

        # aceita linhas como:
        # 0: 3
        # 1: 8
        # ou "0 -> 3"
        for m in re.finditer(r"(\d+)\s*[:=\-]\s*(\d+(?:\.\d+)?)", response):
            try:
                idx = int(m.group(1))
                score = float(m.group(2))
            except Exception:
                continue

            # aceita 1-based se necessário
            if 1 <= idx <= n_options and not (0 <= idx < n_options):
                idx = idx - 1

            if 0 <= idx < n_options and idx != forbidden_idx:
                parsed.append((score, idx))

        parsed.sort(reverse=True)
        out: List[int] = []
        for _, idx in parsed:
            if idx not in out:
                out.append(idx)
        return out
    
    def _parse_ranking(self, obj: Dict[str, Any], n_options: int) -> List[int]:
        ranking = obj.get("ranking")
        out: List[int] = []

        if isinstance(ranking, list):
            for item in ranking:
                try:
                    idx = int(item)
                except Exception:
                    continue
                if 0 <= idx < n_options and idx not in out:
                    out.append(idx)

        # aceita também {"index": N}
        if not out and "index" in obj:
            try:
                idx = int(obj["index"])
                if 0 <= idx < n_options:
                    out.append(idx)
            except Exception:
                pass

        return out

    def _parse_score_map(self, obj: Dict[str, Any], n_options: int, forbidden_idx: int | None = None) -> List[int]:
        scores = obj.get("scores")
        if not isinstance(scores, dict):
            return []

        parsed: List[tuple[float, int]] = []
        for k, v in scores.items():
            try:
                idx = int(k)
                score = float(v)
            except Exception:
                continue
            if 0 <= idx < n_options and idx != forbidden_idx:
                parsed.append((score, idx))

        parsed.sort(reverse=True)
        return [idx for _, idx in parsed]

    def _score_song_for_clue(self, song: Dict[str, Any], clue: str) -> float:
        clue_words = set(self._extract_keywords(clue))
        song_words = set(self._song_keywords(song, limit=20))
        overlap = len(clue_words & song_words)
        title_bonus = 0.5 if any(word in song.get("title", "").lower() for word in clue_words) else 0.0
        return overlap + title_bonus

    def _optimality_score(self, song: Dict[str, Any]) -> float:
        lyrics = song.get("lyrics", "")
        lyrics_len = len(lyrics)
        score = 1.0 - abs(lyrics_len - 200) / 200
        words = re.findall(r"[a-záàâãéêíóôõúç]+", lyrics.lower())
        total_words = len(words)
        unique_words = len(set(words))
        if total_words:
            score += unique_words / total_words
        if "amor" in lyrics.lower():
            score -= 0.3
        return score

    def _mock_llm_response(self, prompt: str, max_tokens: int = 40) -> str:
        words = self._extract_keywords(prompt)
        if not words:
            return "memória tempo cidade"
        return " ".join(words[: min(6, max_tokens)])

    async def receive_hand(self, hand: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.hand = list(hand)
        return {"status": "ok", "hand_size": len(self.hand)}