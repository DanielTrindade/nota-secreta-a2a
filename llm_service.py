from __future__ import annotations

""" Serviço LLM centralizado exposto por REST/FastAPI.
    Simula uma API similar aa usada pela Chat GPT que 
    é quase um padrão da industira hoje.
    Marco Cristo, 2026. Com ajuda da Chat GPT.
"""

import argparse
import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import traceback
import uvicorn


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    stop: List[str] = Field(default_factory=list)


class LocalLLM:
    def __init__(self, model_path: Optional[str] = None, force_mock: bool = False):
        self.model_path = model_path
        self.force_mock = force_mock
        self.mode = "mock"
        self.llm = None

        if model_path and not force_mock:
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=max(2, os.cpu_count() or 2),
                    n_gpu_layers=-1,
                    verbose=False,
                )
                self.mode = "llama-cpp"
            except Exception:
                self.mode = "mock"

    def generate(self, prompt: str, max_tokens: int, temperature: float, stop: List[str]) -> Dict[str, Any]:
        if self.mode == "llama-cpp" and self.llm is not None:
            output = self.llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=stop or None)
            text = output["choices"][0]["text"].strip()
            usage = output.get("usage", {})
            return {"text": text, "usage": usage, "mode": self.mode}

        # se mock ou deu erro na LLM...
        text = "memória tempo cidade"
        usage = {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(text.split()),
            "total_tokens": len(prompt.split()) + len(text.split()),
        }
        return {"text": text, "usage": usage, "mode": self.mode}


class QueueProcessor:
    'Implementa uma fila de requisicoes para lidar com varias requisicoes simultaneas'
    def __init__(self, backend: LocalLLM, max_concurrency: int = 1):
        self.backend = backend
        self.sem = asyncio.Semaphore(max_concurrency)

    async def generate(self, request: GenerateRequest) -> Dict[str, Any]:
        async with self.sem:
            started = time.perf_counter()
            print(
                f"[llm_service] prompt_chars={len(request.prompt)} "
                f"max_tokens={request.max_tokens} temp={request.temperature}"
            )
            result = self.backend.generate(
                request.prompt,
                request.max_tokens,
                request.temperature,
                request.stop,
            )
            result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return result


STATE: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default=None)
    parser.add_argument("--force-mock", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=1)
    args, _ = parser.parse_known_args()

    backend = LocalLLM(model_path=args.model, force_mock=args.force_mock)
    STATE["backend"] = QueueProcessor(backend, max_concurrency=args.max_concurrency)
    yield


app = FastAPI(title="Nota Secreta LLM Service", lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    processor: QueueProcessor = STATE["backend"]
    return {"status": "ok", "mode": processor.backend.mode}


@app.post("/generate")
async def generate(req: GenerateRequest) -> Dict[str, Any]:
    try:
        processor: QueueProcessor = STATE["backend"]
        return await processor.generate(req)
    except Exception as e:
        print("[llm_service] generate failed:", type(e).__name__, e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--force-mock", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run("llm_service:app", host=args.host, port=args.port, reload=False)