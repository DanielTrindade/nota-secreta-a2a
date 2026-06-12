"""Testes do agente calibrado (llm_agent_v2.CalibratedLLMAgent).

Estilo igual ao tests/test_scoring.py: pytest puro, asserções diretas sobre o
comportamento externo observável. Dois seams:

- Seam B (unitário puro): o classificador de banda de dificuldade e o desempate
  do merge de blefe. Entradas sintéticas, sem LLM.
- Seam A (boundary): monkeypatch de ``llm_generate`` para devolver saídas
  canônicas e exercitar as tools send_clue / select_card_by_clue / vote.
"""

import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_agent_v2 import CalibratedLLMAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(margem_min=0.5, margem_max=4.0):
    agent = CalibratedLLMAgent(name="t", llm_url="http://127.0.0.1:1",
                               margem_min=margem_min, margem_max=margem_max)
    return agent


def card(card_id, title, lyrics):
    return {"id": card_id, "title": title, "lyrics": lyrics}


def queue_llm(agent, responses):
    """Substitui llm_generate por uma fila determinística e conta chamadas."""
    state = {"n": 0}

    async def fake(prompt, *args, **kwargs):
        i = state["n"]
        state["n"] += 1
        return responses[min(i, len(responses) - 1)]

    agent.llm_generate = fake
    return state


# Iscas sem nenhuma sobreposição lexical com a dica "cidade luz noite".
DECOYS = [
    card(2, "Mar", "ondas batendo forte sereia"),
    card(3, "Cafe", "xicara quente manha fria"),
    card(4, "Estrada", "asfalto longo poeira seca"),
]


# ---------------------------------------------------------------------------
# Seam B - classificador de banda (puro)
# ---------------------------------------------------------------------------

def test_classify_calibrated():
    """Alvo em 1º com margem dentro da banda -> 'calibrated'."""
    agent = make_agent()
    target = card(1, "Sertao", "cidade grande tem muita gente andando")  # overlap=1
    cards = [target] + DECOYS
    assert agent._classify_clue_difficulty("cidade luz noite", cards, 1) == "calibrated"


def test_classify_obvious():
    """Alvo domina por margem enorme (dica quase literal) -> 'obvious'."""
    agent = make_agent()
    target = card(1, "Sertao", "cidade clara com luz de noite estrelada")  # overlap=3
    cards = [target] + DECOYS
    assert agent._classify_clue_difficulty("cidade luz noite", cards, 1) == "obvious"


def test_classify_vague_when_target_not_first():
    """Uma isca casa mais com a dica que o alvo -> 'vague'."""
    agent = make_agent()
    target = card(1, "X", "cidade apenas")                # overlap=1
    decoy = card(2, "Y", "luz noite forte serena")        # overlap=2 > alvo
    cards = [target, decoy, DECOYS[1], DECOYS[2]]
    assert agent._classify_clue_difficulty("cidade luz noite", cards, 1) == "vague"


# ---------------------------------------------------------------------------
# Seam B - desempate do blefe no merge (puro)
# ---------------------------------------------------------------------------

def test_merge_tiebreak_prefers_stronger_match():
    """Com Borda empatado, o maior tie_break (match mais forte) vence."""
    agent = make_agent()
    ranked = agent._merge_rankings([], [], 3, tie_break={0: 5.0, 1: 9.0, 2: 1.0})
    assert ranked[0] == 1


def test_merge_without_tiebreak_is_backward_compatible():
    """Sem tie_break, empate de Borda cai no menor índice (comportamento antigo)."""
    agent = make_agent()
    ranked = agent._merge_rankings([], [], 3)
    assert ranked == [0, 1, 2]


# ---------------------------------------------------------------------------
# Seam A - send_clue (monkeypatch llm_generate)
# ---------------------------------------------------------------------------

def test_send_clue_returns_calibrated_as_is_single_call():
    agent = make_agent()
    target = card(1, "Sertao", "cidade grande tem muita gente andando")
    agent.hand = [target] + DECOYS
    agent.last_narrator_card = target
    calls = queue_llm(agent, ["cidade luz noite"])

    res = asyncio.run(agent.send_clue(lyrics="uma letra qualquer sobre estradas", max_words=6))

    assert res["clue"] == "cidade luz noite"
    assert calls["n"] == 1  # nenhuma correção quando já está calibrada


def test_send_clue_off_band_then_correction_fails_uses_thematic_fallback():
    agent = make_agent()
    target = card(1, "Sertao", "cidade clara com luz de noite estrelada")  # vira 'obvious'
    agent.hand = [target] + DECOYS
    agent.last_narrator_card = target
    # As duas gerações continuam óbvias -> cai no fallback temático.
    calls = queue_llm(agent, ["cidade luz noite", "cidade luz noite"])

    res = asyncio.run(agent.send_clue(lyrics="saudade lembranca do passado distante", max_words=6))

    assert calls["n"] == 2  # 1 inicial + no máximo 1 correção
    assert res["clue"] == "memória que insiste"


def test_send_clue_off_band_then_correction_succeeds():
    agent = make_agent()
    target = card(1, "Sertao", "cidade clara com luz de noite estrelada")  # 'obvious'
    agent.hand = [target] + DECOYS
    agent.last_narrator_card = target
    # A correção devolve uma dica calibrada (overlap=1 com o alvo).
    calls = queue_llm(agent, ["cidade luz noite", "cidade apenas sozinho"])

    res = asyncio.run(agent.send_clue(lyrics="saudade lembranca do passado distante", max_words=6))

    assert calls["n"] == 2
    assert res["clue"] == "cidade apenas sozinho"


# ---------------------------------------------------------------------------
# Seam A - select_card_by_clue (blefe) e vote
# ---------------------------------------------------------------------------

def test_select_card_by_clue_picks_strongest_bluff():
    agent = make_agent()
    agent.hand = [
        card(10, "A", "cidade clara com luz de noite estrelada"),  # match forte
        card(11, "B", "ondas batendo forte sereia"),
        card(12, "C", "xicara quente manha fria"),
        card(13, "D", "asfalto longo poeira seca"),
    ]
    queue_llm(agent, ["{}"])  # LLM não desempata (ranking vazio)

    res = asyncio.run(agent.select_card_by_clue(clue="cidade luz noite"))

    assert res["chosen_card"]["id"] == 10


def test_vote_returns_two_valid_distinct_votes():
    agent = make_agent()
    options = [
        card(20, "A", "cidade luz"),
        card(21, "B", "mar onda"),
        card(22, "C", "festa samba"),
        card(23, "D", "amor coracao"),
        card(24, "E", "noite lua"),
        card(25, "F", "estrada poeira"),
    ]
    own = options[2]
    queue_llm(agent, ["{}"])

    res = asyncio.run(agent.vote(clue="cidade luz noite", options=options, my_chosen_card=own))
    votes = res["votes"]

    assert len(votes) == 2
    assert len(set(votes)) == 2
    assert 2 not in votes                       # nunca vota na própria carta
    assert all(0 <= v < len(options) for v in votes)
