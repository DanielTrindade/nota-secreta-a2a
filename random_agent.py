from __future__ import annotations

""" Agente baseline aleatório para comparação e testes de infra.
    Marco Cristo, 2026
    Este é exatamente o agente aleatorio que vai competir com vcs
    no torneio. Ganhem dele sistematicamente e voces ja tem 1 pto
"""

import argparse
import random
from typing import Any, Dict, List

from base_agent import BaseAgent
from fasta2a import A2AApp, tool

app = A2AApp(name="RandomAgent")


class RandomAgent(BaseAgent):
    def __init__(self, name: str, llm_url: str):
        super().__init__(name=name, llm_url=llm_url)

    @tool()
    async def receive_hand(self, hand: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.hand = list(hand)
        return {"status": "ok", "hand_size": len(self.hand)}

    @tool()
    async def choose_card(self) -> Dict[str, Any]:
        """Escolhe carta aleatoriamente entre as que estao na mao"""
        chosen = random.choice(self.hand)
        return {"chosen_card": chosen}

    @tool()
    async def send_clue(self, lyrics: str, max_words: int = 6) -> Dict[str, Any]:
        """Usa como dica as primeiras palvras da musica ou um fallback"""
        words = self._extract_keywords(lyrics)[:max_words]
        clue = " ".join(words[:max_words]) if words else "canção brasileira misteriosa"
        return {"clue": clue}

    @tool()
    async def select_card_by_clue(self, clue: str) -> Dict[str, Any]:
        """Seleciona uma carta aleatoria"""
        chosen = random.choice(self.hand)
        return {"chosen_card": chosen}

    @tool()
    async def vote(self, clue: str, options: List[Dict[str, Any]], my_chosen_card: Dict[str, Any]) -> Dict[str, Any]:
        """Escolhe dois votos aleatórios válidos."""
        my_idx = next(i for i, option in enumerate(options) if option["id"] == my_chosen_card["id"])
        choices = [i for i in range(len(options)) if i != my_idx]
        votes = random.sample(choices, 2)
        return {"votes": votes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("game_master_url")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--llm-url", default="http://127.0.0.1:9000")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    agent = RandomAgent(name=args.name or f"RandomAgent_{args.port}", llm_url=args.llm_url)
    app.register(agent)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
