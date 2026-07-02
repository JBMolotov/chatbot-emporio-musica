"""Orquestrador do agente de atendimento.

Este módulo é o ponto central que vai:
    1. Recuperar contexto relevante via `rag.retriever.BaseRetriever`
       (ex.: políticas da loja).
    2. Consultar dados tabulares via
       `tabular_data.service.BaseTabularDataService` (produtos, pedidos,
       promoções etc.), possivelmente através de `tools`.
    3. Recuperar/atualizar o histórico da sessão via
       `memory.conversation_history.BaseConversationHistoryStore`.
    4. Montar o prompt final (system prompt do Empório da Música) e chamar
       `agent.llm_client.GeminiLLMClient`.
    5. Persistir a nova troca de mensagens no histórico.

"""

from __future__ import annotations

from dataclasses import dataclass

from agent.llm_client import GeminiLLMClient
from memory.conversation_history import BaseConversationHistoryStore
from rag.retriever import Retriever
from tabular_data.service import BaseTabularDataService
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentDependencies:
    """Agrupa as dependências injetadas no agente.

    Usar um único objeto de dependências (em vez de vários parâmetros soltos)
    facilita adicionar novas fontes de contexto no futuro sem alterar a
    assinatura de `EmporioMusicaAgent`.
    """

    llm_client: GeminiLLMClient
    retriever: Retriever | None = None
    tabular_data_service: BaseTabularDataService | None = None
    history_store: BaseConversationHistoryStore | None = None


class EmporioMusicaAgent:
    """Agente de atendimento da loja Empório da Música."""

    def __init__(self, dependencies: AgentDependencies) -> None:
        self._deps = dependencies

    def handle_message(self, *, session_id: str, user_message: str) -> str:
        """Processa uma mensagem do usuário e retorna a resposta do agente.

        Ponto de extensão principal: aqui entra o pipeline completo descrito
        no docstring do módulo (RAG + dados tabulares + histórico + prompt
        final + chamada ao LLM). Nenhuma lógica de negócio, prompt ou
        ferramenta está implementada nesta versão inicial.
        """

        logger.debug(
            "Recebida mensagem do usuário (session_id=%s): %r",
            session_id,
            user_message,
        )

        
