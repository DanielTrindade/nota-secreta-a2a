#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exibe logs JSON em um relatório resumido mais fácil de ler.
Marco Cristo, 2026. Com (muita) ajuda da Chat GPT

Note que este render não preserva cada campo JSON original, mas apresenta a partida 
em uma forma mais fácil de ler por uma pessoa. Isso facilita depuração do jogo.
Eu usei muito este script para ver o que o meu agnete fazia.

Usage:
    python render_log_readable.py partida.json
    python render_log_readable.py partida.json -o partida_legivel.txt
    python render_log_readable.py partida.json --lyrics-chars 90 --wrap 100
"""
from __future__ import annotations

import argparse
import json
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def clip(text: str, limit: int) -> str:
    """Retorna um fragmento do texto de entreda em uma só linha. 
       Ex: os primeiros <limit> caracteres da letra de uma música"""
    cleaned = " ".join(str(text).split())
    if limit <= 0:
        return cleaned
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def wrap_block(text: str, width: int, indent: str = "") -> str:
    """Ajusta texto preservando as quebras de parágrafos na medida do possível."""
    paragraphs = str(text).splitlines() or [""]
    chunks: List[str] = []
    for para in paragraphs:
        if not para.strip():
            chunks.append("")
            continue
        chunks.append(
            textwrap.fill(
                para,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(chunks)


def terminal_width(default: int = 120) -> int:
    try:
        return max(80, shutil.get_terminal_size((default, 24)).columns)
    except Exception:
        return default


def build_table(rows: Sequence[Sequence[str]], headers: Sequence[str]) -> str:
    """Cria uma tabela de texto alinhada sem usar nenhum pacote externo 
       (só para garantir que esse código vai rodar em vários ambientes)."""
    all_rows = [list(map(str, headers))] + [list(map(str, row)) for row in rows]
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]

    def fmt(row: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    lines = [fmt(headers), fmt(["-" * w for w in widths])]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


class ReadableLogRenderer:
    def __init__(self, data: Mapping[str, Any], lyrics_chars: int = 80, wrap: int = 110):
        self.data = data
        self.lyrics_chars = lyrics_chars
        self.wrap = wrap
        self.agents: Dict[int, Dict[str, Any]] = {
            int(agent["id"]): dict(agent) for agent in data.get("agents", [])
        }

    def render(self) -> str:
        parts = [self._render_header(), self._render_initial_hands(), self._render_rounds(), self._render_footer()]
        return "\n\n".join(part for part in parts if part.strip()) + "\n"

    def _render_header(self) -> str:
        return f"Log: {self.data.get('timestamp', 'N/A')}"

    def _render_initial_hands(self) -> str:
        hands = self.data.get("initial_hands", [])
        if not hands:
            return "Lyrics\n(no start hand in log)"

        rows: List[List[str]] = []
        for hand in hands:
            agent_id = int(hand.get("agent", -1))
            cards = hand.get("cards", []) or []
            if not cards:
                rows.append([
                    str(agent_id),
                    self.agents.get(agent_id, {}).get("type", hand.get("type", "?")),
                    hand.get("agent_name", self.agents.get(agent_id, {}).get("name", "?")),
                    "-",
                    "-",
                    "-",
                ])
                continue

            first = True
            for card in cards:
                rows.append([
                    str(agent_id) if first else "",
                    self.agents.get(agent_id, {}).get("type", hand.get("type", "?")) if first else "",
                    hand.get("agent_name", self.agents.get(agent_id, {}).get("name", "?")) if first else "",
                    str(card.get("id", "")),
                    str(card.get("title", "")),
                    clip(card.get("lyrics", ""), self.lyrics_chars),
                ])
                first = False

        return "Letras\n" + build_table(rows, ["id", "type", "name", "card_id", "title", "lyrics"])

    def _render_rounds(self) -> str:
        rounds = self.data.get("rounds", [])
        rendered = [self._render_round(round_data) for round_data in rounds]
        return "\n\n".join(rendered)

    def _render_round(self, round_data: Mapping[str, Any]) -> str:
        round_no = round_data.get("round", "?")
        narrador = int(round_data.get("narrador", -1))
        clue = str(round_data.get("clue", ""))
        played_cards = list(round_data.get("played_cards", []))
        votes_by_agent = {int(item["agent"]): item for item in round_data.get("votes_by_agent", [])}
        scores = list(round_data.get("scores", []))
        cumulative = list(round_data.get("cumulative_scores", []))

        drawn_cards = {
            int(item["agent"]): item
            for item in round_data.get("drawn_cards", [])
        }

        option_to_voters: Dict[int, List[int]] = {int(card["option"]): [] for card in played_cards}
        for agent_vote in round_data.get("votes_by_agent", []):
            voter = int(agent_vote["agent"])
            for opt in agent_vote.get("voted_options", []):
                option_to_voters.setdefault(int(opt), []).append(voter)

        played_by_agent = {int(card["agent"]): card for card in played_cards}

        voter_columns = [str(agent_id) for agent_id in sorted(self.agents)]
        headers = ["id", "card_id", "title", *voter_columns, "Pts", "Acm", "New id", "New lyrics"]
        rows: List[List[str]] = []
        for agent_id in sorted(self.agents):
            card = played_by_agent.get(agent_id)
            if card is None:
                continue
            option = int(card["option"])
            voters = set(option_to_voters.get(option, []))
            voter_marks = []
            for voter_id in sorted(self.agents):
                if voter_id == narrador:
                    # Mantem a coluna do narrador visível mesmo que ele nunca vote
                    voter_marks.append("x" if voter_id in voters else "")
                else:
                    voter_marks.append("x" if voter_id in voters else "")
            new_card = drawn_cards.get(agent_id, {})

            rows.append([
                str(agent_id),
                str(card.get("card_id", "")),
                str(card.get("title", "")),
                *voter_marks,
                str(scores[agent_id]) if agent_id < len(scores) else "",
                str(cumulative[agent_id]) if agent_id < len(cumulative) else "",
                str(new_card.get("card_id", "") or ""),
                str(new_card.get("title", "") or ""),
            ])

        block_parts = [
            f"round {round_no}",
            f'Clue (agent {narrador} - {self.agents.get(narrador, {}).get("name", "?")}): "{clue}"',
            build_table(rows, headers),
        ]
        return "\n".join(part for part in block_parts if part)

    def _render_footer(self) -> str:
        winner = self.data.get("winner")
        winner_name = self.agents.get(int(winner), {}).get("name", "?") if winner is not None else "?"
        final_scores = self.data.get("final_scores", [])
        score_text = ", ".join(
            f"{agent_id}:{final_scores[agent_id]}"
            for agent_id in range(min(len(final_scores), len(self.agents)))
        )
        lines = [
            f"Final_scores: [{score_text}]" if score_text else "Final_scores: []",
            f"Winner: {winner} ({winner_name})",
            f"Total_rounds: {self.data.get('total_rounds', '?')}",
        ]
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Nota Secreta logs in readable text format.")
    parser.add_argument("input", type=Path, help="Path to the JSON log file.")
    parser.add_argument("-o", "--output", type=Path, help="Write the readable output to this file.")
    parser.add_argument(
        "--lyrics-chars",
        type=int,
        default=80,
        help="Maximum number of characters shown in lyric previews (default: 80).",
    )
    parser.add_argument(
        "--wrap",
        type=int,
        default=min(120, terminal_width()),
        help="Wrap width for long detail lines (default: terminal width, capped at 120).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    renderer = ReadableLogRenderer(data, lyrics_chars=args.lyrics_chars, wrap=args.wrap)
    output = renderer.render()

    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
