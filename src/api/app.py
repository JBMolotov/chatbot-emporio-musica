"""API HTTP mínima, com validação e autenticação opcional por chave."""
from __future__ import annotations

from functools import lru_cache
from hmac import compare_digest

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from agent.core import EmporioMusicaAgent
from bootstrap import build_agent
from config import Settings, get_settings


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    session_id: str
    response: str


@lru_cache
def get_agent() -> EmporioMusicaAgent:
    return build_agent(get_settings())


def require_api_key(x_api_key: str | None = Header(default=None), settings: Settings = Depends(get_settings)) -> None:
    # Sem chave configurada, útil apenas em desenvolvimento local. Produção deve configurá-la.
    if settings.app_env.lower() == "production" and not settings.api_key:
        raise HTTPException(status_code=503, detail="API sem autenticação configurada.")
    if settings.api_key and not (x_api_key and compare_digest(x_api_key, settings.api_key)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Não autorizado.")


app = FastAPI(title="Empório da Música API", version="0.2.0")


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(request: ChatRequest, agent: EmporioMusicaAgent = Depends(get_agent)) -> ChatResponse:
    try:
        response = agent.handle_message(session_id=request.session_id, user_message=request.message.strip())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail="O serviço de IA está indisponível no momento.") from exc
    return ChatResponse(session_id=request.session_id, response=response)
