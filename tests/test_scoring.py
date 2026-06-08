"""Testes unitários da regra de pontuação."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from game_master import NotaSecretaGame


def build_game(tmp_path, simplified=False):
    csv_path = tmp_path / "songs.csv"
    csv_path.write_text(
        "id,title,artist,lyrics\n1,A,B,letra um\n2,C,D,letra dois\n3,E,F,letra tres\n4,G,H,letra quatro\n5,I,J,letra cinco\n6,K,L,letra seis\n",
        encoding="utf-8",
    )
    return NotaSecretaGame(db_path=str(csv_path), simplified=simplified, log_dir=str(tmp_path / "logs"))


def test_complete_scoring(tmp_path):
    """Conferir o exemplo completo do enunciado com 2 votos."""
    game = build_game(tmp_path, simplified=False)
    game.agents = [type("Agent", (), {"id": i})() for i in range(6)]
    played_cards = [
        {"option": 0, "agent": 0, "card_id": 14},
        {"option": 1, "agent": 1, "card_id": 87},
        {"option": 2, "agent": 2, "card_id": 33},
        {"option": 3, "agent": 3, "card_id": 52},
        {"option": 4, "agent": 4, "card_id": 19},
        {"option": 5, "agent": 5, "card_id": 61},
    ]
    votes = [[], [0, 3], [0, 4], [2, 4], [0, 2], [1, 2]]
    scores, received = game._apply_scoring(0, played_cards, votes)
    assert received == [3, 1, 3, 1, 2, 0]
    assert scores == [3, 4, 6, 1, 5, 0]


def test_simplified_scoring(tmp_path):
    """Conferir a versão simplificada fornecida aos alunos."""
    game = build_game(tmp_path, simplified=True)
    game.agents = [type("Agent", (), {"id": i})() for i in range(6)]
    played_cards = [{"option": i, "agent": i, "card_id": i} for i in range(6)]
    votes = [[], [0, 2], [0, 3], [0, 4], [1, 2], [1, 2]]
    scores, _ = game._apply_scoring(0, played_cards, votes)
    assert scores[0] == 3
    assert scores[1] == 3
    assert scores[2] == 3
    assert scores[3] == 3
    assert scores[4] == 0
    assert scores[5] == 0
