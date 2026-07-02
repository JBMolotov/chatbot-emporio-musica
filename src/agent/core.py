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
from datetime import datetime

from agent.llm_client import GeminiLLMClient
from memory.conversation_history import BaseConversationHistoryStore, ConversationMessage, MessageRole
from rag.retriever import Retriever
from rag.vector_store import VectorDocument
from tabular_data.service import BaseTabularDataService
from tools.base import BaseTool
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
    tools: list[BaseTool] | None = None


class EmporioMusicaAgent:
    """Agente de atendimento da loja Empório da Música."""

    def __init__(self, dependencies: AgentDependencies) -> None:
        self._deps = dependencies

    def handle_message(self, *, session_id: str, user_message: str) -> str:
        """Processa uma mensagem do usuário e retorna a resposta do agente.
        """

        logger.debug(
            "Recebida mensagem do usuário (session_id=%s): %r",
            session_id,
            user_message,
        )

        # Recuperar histórico da sessão
        history = self._deps.history_store.get_history(session_id=session_id) if self._deps.history_store else []

        # Recuperar contexto relevante
        retrieved_context = self._deps.retriever.retrieve(user_message) if self._deps.retriever else []

        # Montar o prompt de sistema (persona do Empório da Música + instruções
        # de uso das tools) e a lista de `messages` (histórico + nova mensagem).
        system_prompt = self._build_system_prompt(retrieved_context)
        messages = [self._to_gemini_message(m) for m in history]
        messages.append({"role": "user", "content": user_message})

        # Chamar o LLM para gerar a resposta, passando o prompt de sistema, o
        # histórico + nova mensagem do usuário e as tools disponíveis.

        tools = [t.to_gemini_function_declaration() for t in self._deps.tools] if self._deps.tools else []

        llm_response = self._deps.llm_client.generate_response(
            system=system_prompt,
            messages=messages,
            tools=tools,
        )

        # Persistir a nova troca de mensagens no histórico (usuário + assistente).

        if self._deps.history_store:
            now = datetime.now()
            self._deps.history_store.add_message(
                session_id, ConversationMessage(role=MessageRole.USER, content=user_message, timestamp=now)
            )
            self._deps.history_store.add_message(
                session_id, ConversationMessage(role=MessageRole.ASSISTANT, content=llm_response, timestamp=now)
            )

        return llm_response

    @staticmethod
    def _build_system_prompt(retrieved_context: list[VectorDocument]) -> str:
        base_prompt = (
            "Você é um assistente virtual da loja Empório da Música. "
            "Seu objetivo é ajudar os clientes a encontrar produtos, "
            "verificar o status de pedidos e fornecer informações sobre "
            "as políticas da loja. Use as ferramentas disponíveis quando "
            "necessário para fornecer respostas precisas e úteis."
        )
        if not retrieved_context:
            return base_prompt

        context_block = "\n".join(f"- {doc.text}" for doc in retrieved_context)
        return f"{base_prompt}\n\nContexto relevante das políticas da loja:\n{context_block}"

    @staticmethod
    def _to_gemini_message(message: ConversationMessage) -> dict[str, str]:
        role = "user" if message.role == MessageRole.USER else "model"
        return {"role": role, "content": message.content}
    