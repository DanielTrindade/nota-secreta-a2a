from __future__ import annotations

"""Game Master do projeto Nota Secreta.
Marco Cristo, 2026 (com ajuda do Chat GPT)

VERSÃO PARA OS ALUNOS.

Este arquivo coordena toda a partida:
- recebe o registro dos agentes;
- distribui as cartas iniciais;
- conduz as rodadas;
- pede dicas, cartas e votos aos agentes;
- aplica a pontuação;
- repõe as cartas jogadas;
- salva um log completo da partida em JSON.
"""

import argparse
import csv
import json
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Sequence

import aiohttp
from fastapi import FastAPI, HTTPException
import uvicorn


# -----------------------------------------------------------------------------
# Estruturas auxiliares
# -----------------------------------------------------------------------------

@dataclass
class AgentRef:
    """Representa um agente registrado na partida.

    Campos:
    - id: índice interno do agente no Game Master
    - name: nome exibido nos logs
    - url: endpoint HTTP do agente
    - kind: tipo do agente (ex.: strategic, random)
    """
    id: int
    name: str
    url: str
    kind: str


class A2AClient:
    """Cliente mínimo para chamar as tools A2A/JSON-RPC dos agentes.

    O Game Master fala com os agentes enviando requisições HTTP POST para o
    endpoint /rpc. Cada chamada usa o método e os parâmetros esperados pela tool.
    """

    def __init__(self, timeout: float = 90.0):
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def call(self, agent: AgentRef, method: str, **params: Any) -> Any:
        """Chama uma tool remota de um agente.

        Exemplo de uso:
            await self.client.call(agent, "choose_card")

        Se a chamada falhar no nível HTTP, resp.raise_for_status() lança erro.
        Se o agente responder com campo "error" no JSON-RPC, também lançamos erro.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": f"{agent.id}-{method}",
            "method": method,
            "params": params,
        }

        started = time.perf_counter()
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.post(f"{agent.url.rstrip('/')}/rpc", json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

                elapsed = round((time.perf_counter() - started) * 1000, 2)
                print(f"[game_master] {agent.name}.{method} took {elapsed} ms")

                if data.get("error"):
                    raise RuntimeError(data["error"]["message"])

                return data.get("result")


# -----------------------------------------------------------------------------
# Lógica principal do jogo
# -----------------------------------------------------------------------------

class NotaSecretaGame:
    """Estado completo de uma partida do jogo Nota Secreta."""

    def __init__(
        self,
        db_path: str,
        target_score: int = 30,
        simplified: bool = False,
        log_dir: str = "logs",
        a2a_timeout: float = 90.0,
    ):
        # Base de músicas
        self.db_path = Path(db_path)

        # Na versão simplificada, o alvo cai para 15 para encurtar a partida.
        self.target_score = 15 if simplified else target_score
        self.simplified = simplified

        # Diretório onde os logs serão gravados.
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Carrega as músicas uma única vez.
        self.songs = self._load_songs(self.db_path)

        # Cliente usado para chamar as tools dos agentes.
        self.client = A2AClient(timeout=a2a_timeout)

        # Estado da partida.
        self.agents: List[AgentRef] = []
        self.scores: List[int] = []
        self.hands: List[List[Dict[str, Any]]] = []

        # O baralho é mantido como deque para permitir pop da esquerda com eficiência.
        self.deck: Deque[Dict[str, Any]] = deque()

        # Estruturas de log.
        self.round_logs: List[Dict[str, Any]] = []
        self.initial_hands_log: List[Dict[str, Any]] = []
        self.last_log_path: str | None = None

    # -------------------------------------------------------------------------
    # Funções auxiliares de representação
    # -------------------------------------------------------------------------

    def _truncate_lyrics_words(self, lyrics: str, max_words: int = 80) -> str:
        """Trunca a letra para no máximo max_words palavras.

        Esta decisão reduz custo e chance de timeout ao enviar contexto para a LLM.
        """
        words = lyrics.split()
        return " ".join(words[:max_words])

    def _agent_view_card(self, song: Dict[str, Any]) -> Dict[str, Any]:
        """Monta a versão enxuta de uma carta enviada aos agentes.

        Escolhas de projeto:
        - mantemos o id, porque ele é necessário para identificar a carta;
        - mantemos o título;
        - omitimos o artista;
        - truncamos a letra.

        Em muitos testes, enviar letra inteira e metadados demais só aumentou o
        tamanho do prompt e piorou robustez/latência. No fim, optei por estas
        reduções para garantir que o jogo funcionaria para todos, mesmo aqueles
        que tivessem uma maquina com poucos recursos para testar seu agente.
        """
        return {
            "id": song["id"],
            "title": song["title"],
            "lyrics": self._truncate_lyrics_words(song["lyrics"], max_words=80),
        }

    # -------------------------------------------------------------------------
    # Carregamento da base
    # -------------------------------------------------------------------------

    def _load_songs(self, path: Path) -> List[Dict[str, Any]]:
        """Carrega a base CSV de músicas.

        O método tenta ser tolerante a:
        - UTF-8 
        - delimitador ',' ou ';'

        A base esperada precisa ter as colunas:
        - id
        - title
        - artist
        - lyrics
        """
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            sample = fh.read(4096)
            fh.seek(0)

            # Heurística inicial para delimitador.
            delimiter = ";" if ";" in sample.splitlines()[0] else ","

            # Tentativa mais refinada com csv.Sniffer.
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                delimiter = dialect.delimiter
            except csv.Error:
                pass

            reader = csv.DictReader(fh, delimiter=delimiter, quotechar='"')
            songs: List[Dict[str, Any]] = []

            for row in reader:
                if not row:
                    continue
                songs.append(
                    {
                        "id": int(row["id"]),
                        "title": row["title"],
                        "artist": row["artist"],
                        "lyrics": row["lyrics"],
                    }
                )

            return songs

    # -------------------------------------------------------------------------
    # Baralho
    # -------------------------------------------------------------------------

    def prepare_deck(self) -> None:
        """Prepara um novo baralho embaralhado."""
        songs = list(self.songs)
        random.shuffle(songs)
        self.deck = deque(songs)

    def draw_card(self) -> Dict[str, Any]:
        """Compra uma carta do topo do baralho.

        Se o baralho acabar, ele é recriado e embaralhado novamente.
        """
        if not self.deck:
            self.prepare_deck()
        return self.deck.popleft()

    # -------------------------------------------------------------------------
    # Registro e distribuição de cartas
    # -------------------------------------------------------------------------

    async def register_agent(self, name: str, url: str, kind: str = "external") -> Dict[str, Any]:
        """Registra um novo agente na partida.

        Cada agente recebe um id incremental.
        Não permitimos registrar duas vezes o mesmo URL.
        """
        if any(agent.url == url for agent in self.agents):
            raise HTTPException(status_code=400, detail="Agent already registered")

        agent = AgentRef(id=len(self.agents), name=name, url=url, kind=kind)
        self.agents.append(agent)

        return {"status": "ok", "agent_id": agent.id, "registered": len(self.agents)}

    async def distribute_initial_hands(self) -> None:
        """Distribui 4 cartas para cada agente e envia a mão inicial via tool.

        Além de atualizar o estado interno, também registramos isso no log.
        """
        self.prepare_deck()
        self.scores = [0 for _ in self.agents]
        self.hands = []
        self.initial_hands_log = []

        for agent in self.agents:
            hand = [self.draw_card() for _ in range(4)]
            self.hands.append(hand)

            self.initial_hands_log.append(
                {
                    "agent": agent.id,
                    "agent_name": agent.name,
                    "cards": hand,
                }
            )

            # Enviamos ao agente apenas a visão enxuta das cartas.
            agent_hand = [self._agent_view_card(song) for song in hand]
            await self.client.call(agent, "receive_hand", hand=agent_hand)

    async def replenish_hand(self, agent_idx: int, played_card_id: int) -> Dict[str, Any]:
        """Remove a carta jogada da mão do agente e recompõe a mão para 4 cartas.

        Retorna a nova carta comprada, para facilitar o log.
        """
        current = [song for song in self.hands[agent_idx] if song["id"] != played_card_id]
        drawn_card = None

        while len(current) < 4:
            drawn_card = self.draw_card()
            current.append(drawn_card)

        self.hands[agent_idx] = current

        agent_hand = [self._agent_view_card(song) for song in current]
        await self.client.call(self.agents[agent_idx], "receive_hand", hand=agent_hand)

        return drawn_card

    # -------------------------------------------------------------------------
    # Funções auxiliares de pontuação
    # -------------------------------------------------------------------------

    def _option_index_of_agent(self, played_cards: Sequence[Dict[str, Any]], agent_idx: int) -> int:
        """Descobre qual opção embaralhada pertence a um certo agente."""
        return next(item["option"] for item in played_cards if item["agent"] == agent_idx)

    def _apply_scoring(
        self,
        narrator_idx: int,
        played_cards: Sequence[Dict[str, Any]],
        votes: Sequence[List[int]],
    ) -> tuple[List[int], List[int]]:
        """Aplica a regra de pontuação da rodada.

        Retorna:
        - round_scores: pontos ganhos por cada agente nesta rodada
        - received_votes: quantidade de votos recebidos por cada opção embaralhada

        Regras:
        - se ninguém acerta a carta do narrador, ou se todos acertam:
            todos os não narradores ganham 2
        - caso contrário:
            narrador ganha 3
            quem acertou a carta do narrador ganha 3
        - na versão completa:
            cada não narrador também ganha até 3 pontos pelos votos recebidos
            em sua própria carta
        """
        narrator_option = self._option_index_of_agent(played_cards, narrator_idx)

        # Conta quantos votos cada opção recebeu.
        received_votes = [0 for _ in played_cards]
        for agent_votes in votes:
            for vote in agent_votes:
                received_votes[vote] += 1

        scorers = [0 for _ in self.agents]
        non_narrator_agents = [idx for idx in range(len(self.agents)) if idx != narrator_idx]

        # Quem acertou a carta do narrador?
        correct_guessers = [idx for idx in non_narrator_agents if narrator_option in votes[idx]]

        if self.simplified:
            if len(correct_guessers) == 0 or len(correct_guessers) == len(non_narrator_agents):
                for idx in non_narrator_agents:
                    scorers[idx] += 2
            else:
                scorers[narrator_idx] += 3
                for idx in correct_guessers:
                    scorers[idx] += 3
            return scorers, received_votes

        # Versão completa de pontuação.
        if len(correct_guessers) == 0 or len(correct_guessers) == len(non_narrator_agents):
            for idx in non_narrator_agents:
                scorers[idx] += 2
        else:
            scorers[narrator_idx] += 3
            for idx in correct_guessers:
                scorers[idx] += 3

        # Pontos extras por votos recebidos na própria carta (máximo 3).
        for agent_idx in non_narrator_agents:
            option_idx = self._option_index_of_agent(played_cards, agent_idx)
            scorers[agent_idx] += min(3, received_votes[option_idx])

        return scorers, received_votes

    # -------------------------------------------------------------------------
    # Loop principal da partida
    # -------------------------------------------------------------------------

    async def play_game(self) -> Dict[str, Any]:
        """Executa a partida completa, rodada por rodada.

        A partida sempre exige exatamente 6 agentes registrados.
        Embora em teoria o jogo suportasse até 12, preferi deixar fixo assim 
        pq foi como fiz todos os testes.
        """
        if len(self.agents) != 6:
            raise RuntimeError("Game requires exactly 6 registered agents")

        await self.distribute_initial_hands()

        round_number = 1
        narrator_idx = 0

        # Continua até alguém alcançar a pontuação-alvo.
        while max(self.scores) < self.target_score:
            narrator = self.agents[narrator_idx]

            # 1) O narrador escolhe sua carta.
            chosen_card = (await self.client.call(narrator, "choose_card"))["chosen_card"]

            narrator_card_full = next(
                song for song in self.hands[narrator_idx] if song["id"] == chosen_card["id"]
            )
            narrator_card_view = self._agent_view_card(narrator_card_full)

            # 2) O narrador gera a dica.
            clue = (
                await self.client.call(
                    narrator,
                    "send_clue",
                    lyrics=narrator_card_view["lyrics"],
                    max_words=6,
                )
            )["clue"]

            # 3) Cada agente escolhe qual carta vai jogar nesta rodada.
            # O narrador joga a sua própria carta escolhida.
            played_by_agent: Dict[int, Dict[str, Any]] = {
                narrator_idx: next(
                    song for song in self.hands[narrator_idx] if song["id"] == chosen_card["id"]
                )
            }

            for idx, agent in enumerate(self.agents):
                if idx == narrator_idx:
                    continue

                chosen = (await self.client.call(agent, "select_card_by_clue", clue=clue))["chosen_card"]
                chosen_full = next(song for song in self.hands[idx] if song["id"] == chosen["id"])
                played_by_agent[idx] = chosen_full

            # 4) Embaralhar as cartas submetidas e gerar as opções visíveis.
            submissions: List[Dict[str, Any]] = []
            for idx, song in played_by_agent.items():
                submissions.append({"agent": idx, "card": song})

            random.shuffle(submissions)

            played_cards = [
                {
                    "option": i,
                    "agent": item["agent"],
                    "agent_name": self.agents[item["agent"]].name,
                    "card_id": item["card"]["id"],
                    "title": item["card"]["title"],
                    "artist": item["card"]["artist"],
                    "lyrics": item["card"]["lyrics"],
                }
                for i, item in enumerate(submissions)
            ]

            # Versão enxuta das opções enviada aos agentes.
            options = [self._agent_view_card(item["card"]) for item in submissions]

            # Mapa útil para enriquecer o log:
            # opção embaralhada -> agente dono da carta.
            option_to_agent = {item["option"]: item["agent"] for item in played_cards}

            # 5) Votação.
            votes: List[List[int]] = []
            votes_by_agent: List[Dict[str, Any]] = []

            for idx, agent in enumerate(self.agents):
                own_idx = self._option_index_of_agent(played_cards, idx)

                if idx == narrator_idx:
                    # O narrador não vota.
                    agent_votes = []
                else:
                    my_chosen_card_view = self._agent_view_card(played_by_agent[idx])

                    agent_votes = (
                        await self.client.call(
                            agent,
                            "vote",
                            clue=clue,
                            options=options,
                            my_chosen_card=my_chosen_card_view,
                        )
                    )["votes"]

                    # Garantias de segurança:
                    # - remove voto na própria carta
                    # - mantém no máximo 2 votos
                    agent_votes = [v for v in agent_votes if v != own_idx][:2]

                    # Se o agente devolver resposta inválida, aplicamos fallback:
                    # usar as duas primeiras opções válidas que não sejam a sua.
                    if len(set(agent_votes)) != 2 or len(agent_votes) != 2:
                        agent_votes = [i for i in range(len(options)) if i != own_idx][:2]

                votes.append(agent_votes)
                votes_by_agent.append(
                    {
                        "agent": idx,
                        "agent_name": self.agents[idx].name,
                        "own_option": own_idx,
                        "own_card_id": played_by_agent[idx]["id"],
                        "voted_options": list(agent_votes),
                        "voted_agents": [option_to_agent[v] for v in agent_votes],
                    }
                )

            # 6) Aplicar pontuação.
            round_scores, received_votes = self._apply_scoring(narrator_idx, played_cards, votes)
            self.scores = [cur + delta for cur, delta in zip(self.scores, round_scores)]

            # Mapa por agente, útil para ler o log sem precisar converter opção -> agente.
            received_votes_by_agent = {item["agent"]: received_votes[item["option"]] for item in played_cards}

            # 7) Repor as cartas jogadas.
            drawn_cards: List[Dict[str, Any]] = []
            for idx, song in played_by_agent.items():
                new_card = await self.replenish_hand(idx, song["id"])
                drawn_cards.append(
                    {
                        "agent": idx,
                        "agent_name": self.agents[idx].name,
                        "card_id": new_card["id"] if new_card else None,
                        "title": new_card["title"] if new_card else None,
                    }
                )

            # 8) Salvar tudo no log da rodada.
            self.round_logs.append(
                {
                    "round": round_number,
                    "narrador": narrator_idx,
                    "narrador_option": self._option_index_of_agent(played_cards, narrator_idx),
                    "clue": clue,
                    "played_cards": played_cards,
                    "votes": votes,
                    "votes_by_agent": votes_by_agent,
                    "received_votes": received_votes,
                    "received_votes_by_agent": received_votes_by_agent,
                    "scores": round_scores,
                    "cumulative_scores": list(self.scores),
                    "drawn_cards": drawn_cards,
                }
            )

            # Próxima rodada: próximo agente vira narrador.
            round_number += 1
            narrator_idx = (narrator_idx + 1) % len(self.agents)

        # 9) Final da partida.
        winner = max(range(len(self.scores)), key=lambda i: self.scores[i])

        log_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "agents": [{"id": a.id, "type": a.kind, "name": a.name} for a in self.agents],
            "initial_hands": self.initial_hands_log,
            "rounds": self.round_logs,
            "final_scores": self.scores,
            "winner": winner,
            "total_rounds": len(self.round_logs),
        }

        # Salva o log em arquivo.
        path = self.log_dir / f"partida_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        self.last_log_path = str(path)
        log_payload["log_file"] = str(path)

        return log_payload


# -----------------------------------------------------------------------------
# Aplicação FastAPI
# -----------------------------------------------------------------------------

def build_app(game: NotaSecretaGame) -> FastAPI:
    """Monta a aplicação FastAPI que expõe o Game Master."""
    app = FastAPI(title="Nota Secreta Game Master")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Endpoint simples de saúde do serviço."""
        return {"status": "ok", "registered_agents": len(game.agents)}

    @app.post("/register")
    async def register(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Registra um agente enviado pelo cliente."""
        return await game.register_agent(
            name=payload["name"],
            url=payload["url"],
            kind=payload.get("kind", "external"),
        )

    @app.post("/play")
    async def play() -> Dict[str, Any]:
        """Executa a partida completa."""
        return await game.play_game()

    return app


# -----------------------------------------------------------------------------
# Execução do script
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="brazilian_songs.csv")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--target-score", type=int, default=30)
    parser.add_argument("--simplified", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--a2a-timeout", type=float, default=90.0)
    args = parser.parse_args()

    game = NotaSecretaGame(
        args.db,
        target_score=args.target_score,
        simplified=args.simplified,
        log_dir=args.log_dir,
        a2a_timeout=args.a2a_timeout,
    )

    uvicorn.run(build_app(game), host=args.host, port=args.port, log_level="info")
