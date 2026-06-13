from __future__ import annotations

""" Script para subir toda a arquitetura e disparar uma partida completa.
    Marco Cristo, 2026
    Esta versão foi a primeira e não é ciente do torneio.
    Ela é muito útil, contudo, para o desenvolvimento do seu agente, pois
    não exige que vc faça o trabalho pesado de sugir e registrar todos os 
    agentes de uma partida.
"""

import argparse
import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import aiohttp

ROOT = Path(__file__).resolve().parent


def find_free_port(start_port: int, host: str = "127.0.0.1", max_tries: int = 200) -> int:
    """Encontrar uma porta TCP livre a partir de uma porta base.

    A função tenta a porta desejada e, se ela estiver ocupada, segue procurando
    nas próximas portas. Isso deixa a execução mais robusta quando houve uma
    execução anterior mal encerrada ou quando outras aplicações estão usando as
    portas padrão do projeto.
    """
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Nenhuma porta livre encontrada a partir de {start_port}")



async def wait_http(url: str, timeout: float = 60.0) -> None:
    """Esperar até um endpoint responder 200 OK."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return
        except Exception:
            pass
        await asyncio.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}")


async def register_agent(game_master_url: str, name: str, url: str, kind: str) -> Dict[str, Any]:
    """Registra o agente."""
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{game_master_url.rstrip('/')}/register", json={"name": name, "url": url, "kind": kind}) as resp:
            resp.raise_for_status()
            return await resp.json()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--force-mock", action="store_true")
    parser.add_argument("--game-master-url", default=None,
                        help="URL do Game Master. Se omitida, o script escolhe uma porta livre automaticamente.")
    parser.add_argument("--llm-url", default=None,
                        help="URL do serviço LLM. Se omitida, o script escolhe uma porta livre automaticamente.")
    parser.add_argument("--base-port", type=int, default=8001)
    parser.add_argument("--db", default=str(ROOT / "brazilian_songs.csv"))
    parser.add_argument("--all-strategic", action="store_true",
        help="Sobe 6 agentes estratégicos em vez de 1 estratégico + 5 aleatórios.",
    )
    parser.add_argument("--agent", default="llm_agent.py",
        help="Script do agente estratégico a usar (ex.: llm_agent.py ou llm_agent_v2.py).",
    )
    parser.add_argument("--llm-max-concurrency", type=int, default=1)
    args = parser.parse_args()

    processes: List[subprocess.Popen[str]] = []
    try:
        # Escolher portas livres de forma robusta para evitar falhas quando a
        # porta padrão já estiver em uso por outra aplicação ou por uma
        # execução anterior que não encerrou corretamente.
        llm_port = int(args.llm_url.rsplit(":", 1)[1]) if args.llm_url else find_free_port(9000)
        gm_port = int(args.game_master_url.rsplit(":", 1)[1]) if args.game_master_url else find_free_port(8000)
        llm_url = args.llm_url or f"http://127.0.0.1:{llm_port}"
        game_master_url = args.game_master_url or f"http://127.0.0.1:{gm_port}"

        print(f"[run_game] LLM Service em {llm_url}")
        print(f"[run_game] Game Master em {game_master_url}")

        # 1. Subir o serviço LLM centralizado.
        llm_cmd = [sys.executable, str(ROOT / "llm_service.py"), 
                   "--port", str(llm_port), 
                   "--max-concurrency", str(args.llm_max_concurrency)]
        if args.model:
            llm_cmd.extend(["--model", args.model])
        if args.force_mock or not args.model:
            llm_cmd.append("--force-mock")
        processes.append(subprocess.Popen(llm_cmd, cwd=str(ROOT)))
        await wait_http(f"{llm_url}/health", timeout=300)

        # 2. Subir o Game Master.
        gm_cmd = [sys.executable, str(ROOT / "game_master.py"), "--port", str(gm_port), "--db", args.db, "--target-score", "30", "--log-dir", str(ROOT / "logs")]
        processes.append(subprocess.Popen(gm_cmd, cwd=str(ROOT)))
        await wait_http(f"{game_master_url}/health")

        # 3. Subir os agentes.
        # O nome exibido nos logs deriva do script do agente estratégico, para
        # deixar claro qual versão está jogando (ex.: llm_agent vs llm_agent_v2).
        strategic_name = Path(args.agent).stem
        if args.all_strategic:
            agent_specs = [
                (args.agent, args.base_port + i, strategic_name, "strategic")
                for i in range(6)
            ]
        else:
            agent_specs = [
                (args.agent, args.base_port, strategic_name, "strategic"),
                ("random_agent.py", args.base_port + 1, "RandomAgent", "random"),
                ("random_agent.py", args.base_port + 2, "RandomAgent", "random"),
                ("random_agent.py", args.base_port + 3, "RandomAgent", "random"),
                ("random_agent.py", args.base_port + 4, "RandomAgent", "random"),
                ("random_agent.py", args.base_port + 5, "RandomAgent", "random"),
            ]

        used_ports = {llm_port, gm_port}
        for idx, (script, preferred_port, base_name, kind) in enumerate(agent_specs, start=1):
            port = find_free_port(preferred_port)
            while port in used_ports:
                port = find_free_port(port + 1)
            used_ports.add(port)
            name = f"{base_name}_{idx}"
            agent_url = f"http://127.0.0.1:{port}"
            print(f"[run_game] Subindo {name} ({kind}) em {agent_url}")
            cmd = [sys.executable, str(ROOT / script), game_master_url, "--port", str(port), "--llm-url", llm_url, "--name", name]
            processes.append(subprocess.Popen(cmd, cwd=str(ROOT)))
            await wait_http(f"{agent_url}/health")
            await register_agent(game_master_url, name=name, url=agent_url, kind=kind)

        # 4. Pedir ao Game Master para executar a partida completa.
        timeout = aiohttp.ClientTimeout(total=1200)  # 20 minutos é o tempo máximo de uma partida 
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{game_master_url}/play") as resp:
                resp.raise_for_status()
                result = await resp.json()
        print("Final scores:", result["final_scores"])
        print("Winner:", result["winner"])
        print("Total rounds:", result["total_rounds"])
        print("Log file:", result.get("log_file", "(not provided)"))
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
