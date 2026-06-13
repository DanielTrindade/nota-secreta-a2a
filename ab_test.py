from __future__ import annotations

"""A/B test: agente antigo (llm_agent.py) vs. novo (llm_agent_v2.py).

POR PADRÃO usa o MODELO REAL sugerido pelo professor (Phi-3.5-mini-instruct
GGUF). É esse o cenário que vale: a calibração de dica do narrador só faz efeito
quando a LLM gera dicas de verdade — em mock ela fica inerte e os dois agentes
empatam. O modo `--force-mock` existe só como smoke test rápido da infra.

Os dois agentes jogam a MESMA partida, lado a lado. A cada partida ALTERNAMOS as
cadeiras (seats) ocupadas pelo antigo e pelo novo, para anular a vantagem
posicional — o narrador começa na cadeira 0 e o papel gira de forma circular.

Relatório final:
- pontos totais, vitórias e confronto direto (quem pontuou mais na partida);
- desempenho COMO NARRADOR: pontos e taxa de acerto da "zona Dixit" (rodadas em
  que o narrador pontuou +3, ou seja, alguns mas não todos acertaram). É a
  métrica que a calibração do narrador tenta melhorar.

Cenários:
- padrão:          1 antigo + 1 novo + 4 RandomAgents (confronto contra baseline);
- --all-strategic: 3 antigos vs. 3 novos (campo todo estratégico, como no torneio),
                   intercalados e com as cadeiras trocadas a cada partida.

Uso:
    # comparação real (modelo do professor). Requer o GGUF baixado (ver README/notebook).
    python ab_test.py --all-strategic --games 6

    # apontando para outro caminho de modelo
    python ab_test.py --all-strategic --games 6 --model caminho/para/modelo.gguf

    # smoke test rápido da infra (NÃO mede qualidade semântica)
    python ab_test.py --games 4 --force-mock --target-score 15
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from run_game import find_free_port, register_agent, wait_http

ROOT = Path(__file__).resolve().parent

OLD_SCRIPT = "llm_agent.py"
NEW_SCRIPT = "llm_agent_v2.py"

# Caminho padrão do modelo, igual ao que o notebook baixa em models/.
DEFAULT_MODEL = "models/phi-3.5-mini-instruct-q4_k_m.gguf"


async def play_once(gm_url: str, timeout: float) -> Dict[str, Any]:
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(f"{gm_url}/play") as resp:
            resp.raise_for_status()
            return await resp.json()


async def fetch_health(url: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()


def spawn_agent(script: str, port: int, name: str, llm_url: str, extra: List[str]) -> subprocess.Popen:
    # O game_master_url posicional é exigido pelo argparse dos agentes, mas eles
    # nunca o usam (é o Game Master quem chama o agente). Passamos um placeholder.
    cmd = [
        sys.executable, str(ROOT / script), "http://127.0.0.1:1",
        "--port", str(port), "--llm-url", llm_url, "--name", name,
    ] + extra
    return subprocess.Popen(cmd, cwd=str(ROOT))


def resolve_model(args: argparse.Namespace) -> Optional[str]:
    """Resolve o caminho do modelo, ou None se for para rodar em mock.

    Em modo real, aborta cedo (retornando "" não, mas saindo via SystemExit) se o
    arquivo não existir, para o usuário não "testar" em mock sem perceber.
    """
    if args.force_mock:
        return None
    model = Path(args.model)
    if not model.is_absolute():
        model = ROOT / model
    if not model.exists():
        print(f"[ab_test] ERRO: modelo não encontrado em {model}")
        print("[ab_test] Baixe o GGUF Phi-3.5-mini-instruct-Q4_K_M (ver seção 5 do notebook)")
        print("[ab_test] ou rode com --force-mock para um smoke test rápido da infra.")
        raise SystemExit(2)
    return str(model)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Caminho do GGUF. Padrão: {DEFAULT_MODEL}")
    parser.add_argument("--force-mock", action="store_true",
                        help="Usa a LLM em mock (smoke test). NÃO mede qualidade semântica.")
    parser.add_argument("--db", default=str(ROOT / "brazilian_songs.csv"))
    parser.add_argument("--target-score", type=int, default=30)
    parser.add_argument("--base-port", type=int, default=8101)
    parser.add_argument("--margem-min", type=float, default=0.5)
    parser.add_argument("--margem-max", type=float, default=4.0)
    parser.add_argument("--all-strategic", action="store_true",
                        help="3 antigos vs 3 novos (campo todo estratégico, como no torneio).")
    parser.add_argument("--llm-max-concurrency", type=int, default=1)
    parser.add_argument("--game-timeout", type=float, default=None,
                        help="Timeout (s) por partida. Padrão: max(600, target_score*120).")
    args = parser.parse_args()

    model_path = resolve_model(args)
    use_mock = model_path is None
    game_timeout = args.game_timeout or max(600.0, args.target_score * 120.0)

    if use_mock:
        print("[ab_test] MODO: MOCK (smoke test) — a calibração do narrador fica inerte; "
              "use apenas para validar a infra.")
    else:
        print(f"[ab_test] MODO: MODELO REAL -> {model_path}")
        print("[ab_test] Atenção: em CPU cada partida pode levar vários minutos "
              "(muitas chamadas à LLM em série).")

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
                   "--port", str(llm_port), "--max-concurrency", str(args.llm_max_concurrency)]
        if use_mock:
            llm_cmd.append("--force-mock")
        else:
            llm_cmd += ["--model", model_path]
        processes.append(subprocess.Popen(llm_cmd, cwd=str(ROOT)))
        # Modelo real demora a carregar; damos uma folga maior no health.
        await wait_http(f"{llm_url}/health", timeout=900 if not use_mock else 120)

        health = await fetch_health(f"{llm_url}/health")
        mode = health.get("mode")
        print(f"[ab_test] LLM service em {llm_url} (mode={mode})")
        # Segurança: se pedimos modelo real mas o serviço caiu em mock (llama-cpp
        # não carregou), abortamos — senão o "teste real" seria um mock disfarçado.
        if not use_mock and mode != "llama-cpp":
            print("[ab_test] ERRO: o serviço LLM está em modo mock (llama-cpp não carregou).")
            print("[ab_test] Verifique a instalação de llama-cpp-python e o caminho do modelo.")
            return

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
        cenario = "all-strategic (3 old vs 3 new)" if args.all_strategic else "1 old + 1 new + 4 random"
        print(f"[ab_test] Cenário: {cenario} | partidas={args.games} | target={args.target_score}")

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
        # Métricas específicas do narrador (onde a calibração age).
        narr_pts = {"old": 0, "new": 0}
        narr_rounds = {"old": 0, "new": 0}
        narr_hits = {"old": 0, "new": 0}  # rodadas em que o narrador pontuou (zona Dixit)

        for game in range(args.games):
            # Alterna a cada partida quem ocupa a cadeira 0 (1º narrador).
            if args.all_strategic:
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

            result = await play_once(gm_url, timeout=game_timeout)
            scores = result["final_scores"]
            score_old = sum(s for seat, s in enumerate(scores) if order[seat]["role"] == "old")
            score_new = sum(s for seat, s in enumerate(scores) if order[seat]["role"] == "new")

            totals["old"] += score_old
            totals["new"] += score_new

            # Métrica de narrador, rodada a rodada.
            for rnd in result.get("rounds", []):
                nidx = rnd["narrador"]
                role = order[nidx]["role"]
                if role in narr_pts:
                    pts = rnd["scores"][nidx]
                    narr_pts[role] += pts
                    narr_rounds[role] += 1
                    if pts > 0:
                        narr_hits[role] += 1

            win_role = order[result["winner"]]["role"]
            if win_role in wins:
                wins[win_role] += 1
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

        def rate(role: str) -> float:
            return (100.0 * narr_hits[role] / narr_rounds[role]) if narr_rounds[role] else 0.0

        print("\n==== RESULTADO A/B (old=llm_agent.py | new=llm_agent_v2.py) ====")
        print(f"Modo................: {'MOCK (smoke)' if use_mock else 'MODELO REAL'}")
        print(f"Cenário.............: {cenario}")
        print(f"Partidas............: {n}")
        print(f"Pontos totais.......: old={totals['old']}  new={totals['new']}")
        print(f"Média por partida...: old={totals['old'] / n:.2f}  new={totals['new'] / n:.2f}")
        print(f"Vitórias............: old={wins['old']}  new={wins['new']}  random={wins['random']}")
        print(f"Confronto direto....: new={head['new']}  old={head['old']}  empates={head['tie']}")
        print("Como narrador (zona Dixit é onde a calibração age):")
        print(f"  old: {narr_pts['old']} pts em {narr_rounds['old']} rodadas  "
              f"| acertou a zona em {narr_hits['old']} ({rate('old'):.0f}%)")
        print(f"  new: {narr_pts['new']} pts em {narr_rounds['new']} rodadas  "
              f"| acertou a zona em {narr_hits['new']} ({rate('new'):.0f}%)")
        if totals["new"] > totals["old"]:
            verdict = "NOVO (llm_agent_v2.py)"
        elif totals["old"] > totals["new"]:
            verdict = "ANTIGO (llm_agent.py)"
        else:
            verdict = "EMPATE"
        print(f"Melhor por pontos...: {verdict}")
        if use_mock:
            print("[ab_test] Lembrete: em MOCK os agentes tendem a empatar; rode sem "
                  "--force-mock (modelo real) para um veredito que valha.")
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
