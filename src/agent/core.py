"""Orquestrador do agente de atendimento."""

from __future__ import annotations

from dataclasses import dataclass

from agent.llm_client import GeminiLLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentDependencies:
    """Agrupa as dependências injetadas no agente.

    Usar um único objeto de dependências (em vez de vários parâmetros soltos)
    facilita adicionar novas fontes de contexto no futuro sem alterar a
    assinatura de `EmporioMusicaAgent`.
    """

    # TODO



class EmporioMusicaAgent:
    """Agente de atendimento da loja Empório da Música."""

    def __init__(self, dependencies: AgentDependencies) -> None:
        self._deps = dependencies

    def handle_message(self, *, session_id: str, user_message: str) -> str:
        """Processa uma mensagem do usuário e retorna a resposta do agente.
        """
        raise NotImplementedError(
            "Implementar o pipeline de atendimento do agente "
            f"(session_id={session_id!r})."
        )
