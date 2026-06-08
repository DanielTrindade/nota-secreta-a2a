from __future__ import annotations

"""Mini-camada A2A compatível com o estilo dado no enunciado do projeto.
   Marco Cristo, 2026
   Implementada para garantir que o código vai rodar em qualqeur ambiente 
   sem necessidade real de usar a biblioteca fastA2A que ainda está 
   em desenvolvimento e poderia quebrar ao longo do tempo entre 
   a publicação do enunciado e desenvolvimento. Tambem tive problemas
   com requisitos distintos da A2A real entre Mac e Linux. Assim, 
   preferi pedir para Chat GPT gerar esse codigo agnostico, que
   deve funcionar em diferentes ambientes Python, inclusive Windows,
   que nao tenho como testar.

Ela implementa apenas o necessário para o projeto:
- ``A2AApp`` para subir o servidor do agente
- ``@tool`` para marcar tools expostas remotamente
- endpoint ``/rpc`` com JSON-RPC sobre HTTP
"""

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ToolCallable = Callable[..., Awaitable[Any]]


def tool(name: Optional[str] = None):
    """Marcar uma função assíncrona como tool A2A."""

    def decorator(func: ToolCallable) -> ToolCallable:
        setattr(func, "_is_a2a_tool", True)
        setattr(func, "_a2a_tool_name", name or func.__name__)
        return func

    return decorator


@dataclass
class RegisteredTool:
    """Representação interna de uma tool registrada."""

    name: str
    handler: Callable[..., Awaitable[Any]]
    signature: inspect.Signature


class A2ARequest(BaseModel):
    jsonrpc: str = Field(default="2.0")
    id: str | int | None = None
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)


class A2AResponse(BaseModel):
    jsonrpc: str = Field(default="2.0")
    id: str | int | None = None
    result: Any | None = None
    error: Dict[str, Any] | None = None


class A2AApp:
    """Aplicação HTTP que expõe tools via JSON-RPC 2.0."""

    def __init__(self, name: str):
        self.name = name
        self.app = FastAPI(title=name)
        self._tools: Dict[str, RegisteredTool] = {}
        self._bind_routes()

    def _bind_routes(self) -> None:
        @self.app.get("/health")
        async def health() -> Dict[str, Any]:
            return {"status": "ok", "name": self.name, "tools": sorted(self._tools)}

        @self.app.get("/tools")
        async def list_tools() -> Dict[str, Any]:
            return {
                "name": self.name,
                "tools": [
                    {"name": t.name, "parameters": list(t.signature.parameters.keys())}
                    for t in self._tools.values()
                ],
            }

        @self.app.post("/rpc", response_model=A2AResponse)
        async def rpc(request: A2ARequest) -> A2AResponse:
            if request.jsonrpc != "2.0":
                raise HTTPException(status_code=400, detail="Only JSON-RPC 2.0 is supported")

            registered = self._tools.get(request.method)
            if registered is None:
                return A2AResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Method not found: {request.method}"},
                )

            try:
                # bound = registered.signature.bind_partial(**request.params)
                # result = await registered.handler(*bound.args, **bound.kwargs)
                # return A2AResponse(id=request.id, result=result)
                bound = registered.signature.bind_partial(**request.params)
                result = registered.handler(*bound.args, **bound.kwargs)

                print(f"[A2A] method={request.method} return_type={type(result)}")

                if inspect.isawaitable(result):
                    result = await result

                return A2AResponse(id=request.id, result=result)
            except TypeError as exc:
                return A2AResponse(
                    id=request.id,
                    error={"code": -32602, "message": f"Invalid params: {exc}"},
                )
            except Exception as exc:  # pragma: no cover
                return A2AResponse(id=request.id, error={"code": -32000, "message": str(exc)})

    def register(self, obj: Any) -> None:
        """Registrar automaticamente todos os métodos decorados com ``@tool``."""
        for attr_name in dir(obj):
            attr = getattr(obj, attr_name)
            if callable(attr) and getattr(attr, "_is_a2a_tool", False):
                name = getattr(attr, "_a2a_tool_name", attr.__name__)
                self._tools[name] = RegisteredTool(name, attr, inspect.signature(attr))

    def run(self, host: str = "127.0.0.1", port: int = 8001) -> None:
        import uvicorn

        uvicorn.run(self.app, host=host, port=port, log_level="info")
