from __future__ import annotations

"""A/B test: agente antigo (llm_agent.py) vs. novo (llm_agent_v2.py).

Os dois agentes jogam a MESMA partida, lado a lado, junto de 4 RandomAgents.
A cada partida ALTERNAMOS as cadeiras (seats) ocupadas pelo antigo e pelo novo,
para anular a vantagem posicional — o narrador começa na cadeira 0 e o papel
gira de forma circular. Ao fim, agregamos pontos totais, vitórias e o confronto
direto (quem pontuou mais na própria partida).

Infra reaproveitada do run_game.py. O serviço LLM e os 6 agentes sobem uma única
vez; um Game Master novo é criado por partida (ele zera o placar a cada /play e
registra os agentes na ordem das cadeiras que escolhemos).

Dois cenários:
- padrão: 1 antigo + 1 novo + 4 RandomAgents (confronto isolado contra baseline);
- --all-strategic: 3 antigos + 3 novos (campo todo estratégico, como no torneio),
  intercalados e com as cadeiras trocadas a cada partida; os pontos são somados
  por lado (soma dos 3 antigos vs. soma dos 3 novos).

Uso:
    # rápido, sem modelo (testa as camadas determinísticas: calibração + blefe)
    python ab_test.py --games 10 --force-mock --target-score 15

    # campo todo estratégico (3 antigos vs 3 novos)
    python ab_test.py --games 10 --all-strategic --model Phi-3.5-mini-instruct-Q4_K_M.gguf

    # com o modelo real (testa também a semântica via LLM)
    python ab_test.py --games 10 --model Phi-3.5-mini-instruct-Q4_K_M.gguf

Observação: em --force-mock a LLM devolve uma saída fixa e ambos os agentes caem
em fallback; o A/B então mede sobretudo as heurísticas (banda de calibração e
desempate de blefe). Para avaliar o ganho semântico, rode com --model.
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import aiohttp

from run_game import find_free_port, register_agent, wait_http

ROOT = Path(__file__).resolve().parent

OLD_SCRIPT = "llm_agent.py"
NEW_SCRIPT = "llm_agent_v2.py"


async def play_once(gm_url: str, timeout: float) -> Dict[str, Any]:
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(f"{gm_url}/play") as resp:
            resp.raise_for_status()
            return await resp.json()


def spawn_agent(script: str, port: int, name: str, llm_url: str, extra: List[str]) -> subprocess.Popen:
    # O game_master_url posicional é exigido pelo argparse dos agentes, mas eles
    # nunca o usam (é o Game Master quem chama o agente). Passamos um placeholder.
    cmd = [
        sys.executable, str(ROOT / script), "http://127.0.0.1:1",
        "--port", str(port), "--llm-url", llm_url, "--name", name,
    ] + extra
    return subprocess.Popen(cmd, cwd=str(ROOT))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--model", default=None)
    parser.add_argument("--force-mock", action="store_true")
    parser.add_argument("--db", default=str(ROOT / "brazilian_songs.csv"))
    parser.add_argument("--target-score", type=int, default=30)
    parser.add_argument("--base-port", type=int, default=8101)
    parser.add_argument("--margem-min", type=float, default=0.5)
    parser.add_argument("--margem-max", type=float, default=4.0)
    parser.add_argument("--all-strategic", action="store_true",
                        help="3 antigos vs 3 novos (campo todo estratégico, como no torneio).")
    args = parser.parse_args()

    processes: List[subprocess.Popen] = []
    used_ports: set[int] = set()

    def reserve(start: int) -> int:
        port = find_free_port(start)
        while port in used_ports:
            port = find_free_port(port + 1)
        used_ports.add(port)
        return port

    try:
        # 1) Serviço LLM (uma vez).
        llm_port = reserve(9100)
        llm_url = f"http://127.0.0.1:{llm_port}"
        llm_cmd = [sys.executable, str(ROOT / "llm_service.py"),
                   "--port", str(llm_port), "--max-concurrency", "1"]
        if args.model:
            llm_cmd += ["--model", args.model]
        if args.force_mock or not args.model:
            llm_cmd.append("--force-mock")
        processes.append(subprocess.Popen(llm_cmd, cwd=str(ROOT)))
        await wait_http(f"{llm_url}/health", timeout=300)
        print(f"[ab_test] LLM service em {llm_url}")

        # 2) Os 6 agentes (uma vez), conforme o cenário escolhido.
        new_extra = ["--margem-min", str(args.margem_min), "--margem-max", str(args.margem_max)]
        if args.all_strategic:
            agent_specs = (
                [(OLD_SCRIPT, "old", [])] * 3
                + [(NEW_SCRIPT, "new", new_extra)] * 3
            )
        else:
            agent_specs = [
                (OLD_SCRIPT, "old", []),
                (NEW_SCRIPT, "new", new_extra),
                ("random_agent.py", "r1", []),
                ("random_agent.py", "r2", []),
                ("random_agent.py", "r3", []),
                ("random_agent.py", "r4", []),
            ]
        print(f"[ab_test] Cenário: {'all-strategic (3 old vs 3 new)' if args.all_strategic else '1 old + 1 new + 4 random'}")

        agents: List[Dict[str, str]] = []
        for tag, (script, role, extra) in enumerate(agent_specs):
            port = reserve(args.base_port)
            url = f"http://127.0.0.1:{port}"
            name = f"{role}{tag}"
            processes.append(spawn_agent(script, port, name, llm_url, extra))
            await wait_http(f"{url}/health")
            agents.append({"role": role, "url": url, "name": name})
            print(f"[ab_test] Agente '{name}' ({script}) em {url}")

        olds = [a for a in agents if a["role"] == "old"]
        news = [a for a in agents if a["role"] == "new"]
        randoms = [a for a in agents if a["role"].startswith("r")]

        totals = {"old": 0, "new": 0}
        wins = {"old": 0, "new": 0, "random": 0}
        head = {"new": 0, "old": 0, "tie": 0}

        for game in range(args.games):
            # Alterna a cada partida quem ocupa a cadeira 0 (1º narrador).
            if args.all_strategic:
                # Intercala 3 antigos e 3 novos; o lado que lidera troca por paridade.
                first, second = (olds, news) if game % 2 == 0 else (news, olds)
                order = [agent for pair in zip(first, second) for agent in pair]
            else:
                if game % 2 == 0:
                    order = [olds[0], news[0]] + randoms
                else:
                    order = [news[0], olds[0]] + randoms

            gm_port = reserve(8200)
            gm_url = f"http://127.0.0.1:{gm_port}"
            gm_cmd = [sys.executable, str(ROOT / "game_master.py"),
                      "--port", str(gm_port), "--db", args.db,
                      "--target-score", str(args.target_score), "--log-dir", str(ROOT / "logs")]
            gm_proc = subprocess.Popen(gm_cmd, cwd=str(ROOT))
            processes.append(gm_proc)
            await wait_http(f"{gm_url}/health")

            for seat, agent in enumerate(order):
                kind = "strategic" if agent["role"] in ("old", "new") else "random"
                await register_agent(gm_url, name=f"seat{seat}_{agent['role']}", url=agent["url"], kind=kind)

            result = await play_once(gm_url, timeout=max(120.0, args.target_score * 120.0))
            scores = result["final_scores"]
            # Soma por lado (no modo padrão é só 1 de cada).
            score_old = sum(s for seat, s in enumerate(scores) if order[seat]["role"] == "old")
            score_new = sum(s for seat, s in enumerate(scores) if order[seat]["role"] == "new")

            totals["old"] += score_old
            totals["new"] += score_new

            win_role = order[result["winner"]]["role"]
            if win_role == "old":
                wins["old"] += 1
            elif win_role == "new":
                wins["new"] += 1
            else:
                wins["random"] += 1

            if score_new > score_old:
                head["new"] += 1
            elif score_old > score_new:
                head["old"] += 1
            else:
                head["tie"] += 1

            print(
                f"[ab_test] partida {game + 1}/{args.games}: "
                f"old={score_old}  new={score_new}  "
                f"vencedor=seat{result['winner']}({win_role})  rodadas={result['total_rounds']}"
            )

            gm_proc.terminate()
            try:
                gm_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gm_proc.kill()

        n = args.games
        print("\n==== RESULTADO A/B (old=llm_agent.py | new=llm_agent_v2.py) ====")
        print(f"Partidas............: {n}")
        print(f"Pontos totais.......: old={totals['old']}  new={totals['new']}")
        print(f"Média por partida...: old={totals['old'] / n:.2f}  new={totals['new'] / n:.2f}")
        print(f"Vitórias............: old={wins['old']}  new={wins['new']}  random={wins['random']}")
        print(f"Confronto direto....: new={head['new']}  old={head['old']}  empates={head['tie']}")
        if totals["new"] > totals["old"]:
            verdict = "NOVO (llm_agent_v2.py)"
        elif totals["old"] > totals["new"]:
            verdict = "ANTIGO (llm_agent.py)"
        else:
            verdict = "EMPATE"
        print(f"Melhor por pontos...: {verdict}")
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
        for proc in reversed(processes):
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
