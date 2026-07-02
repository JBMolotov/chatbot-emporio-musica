"""Testes do orquestrador do agente (`EmporioMusicaAgent`)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from agent.core import AgentDependencies, EmporioMusicaAgent
from memory.conversation_history import ConversationMessage, MessageRole
from memory.json_history_store import JsonConversationHistoryStore
from rag.vector_store import VectorDocument


class _FakeLLMClient:
    """Double de `GeminiLLMClient`: devolve uma resposta fixa em vez de chamar a API."""

    def __init__(self) -> None:
        self.last_call: dict[str, Any] | None = None

    def generate_response(self, *, system: str, messages: list[dict[str, Any]], tools=None) -> str:
        self.last_call = {"system": system, "messages": messages, "tools": tools}
        return "resposta do assistente"


class _FakeRetriever:
    def __init__(self, documents: list[VectorDocument]) -> None:
        self._documents = documents

    def retrieve(self, query: str, top_k: int = 5) -> list[VectorDocument]:
        return self._documents


def test_handle_message_without_optional_deps_does_not_crash() -> None:
    # retriever, tabular_data_service, history_store e tools são todos opcionais
    # (None por padrão) — handle_message não pode quebrar sem eles.
    llm_client = _FakeLLMClient()
    agent = EmporioMusicaAgent(AgentDependencies(llm_client=llm_client))

    response = agent.handle_message(session_id="s1", user_message="olá")

    assert response == "resposta do assistente"


def test_handle_message_builds_system_prompt_with_retrieved_context() -> None:
    llm_client = _FakeLLMClient()
    doc = VectorDocument(id="a", text="troca em até 30 dias", metadata={"page": 4})
    retriever = _FakeRetriever([doc])
    agent = EmporioMusicaAgent(AgentDependencies(llm_client=llm_client, retriever=retriever))

    response = agent.handle_message(session_id="s1", user_message="qual a política de troca?")

    assert response == "resposta do assistente"
    assert "troca em até 30 dias" in llm_client.last_call["system"]
    assert llm_client.last_call["messages"] == [{"role": "user", "content": "qual a política de troca?"}]


def test_handle_message_without_retriever_has_base_system_prompt_only() -> None:
    llm_client = _FakeLLMClient()
    agent = EmporioMusicaAgent(AgentDependencies(llm_client=llm_client))

    agent.handle_message(session_id="s1", user_message="oi")

    assert "Contexto relevante" not in llm_client.last_call["system"]


def test_handle_message_converts_history_roles_for_gemini() -> None:
    llm_client = _FakeLLMClient()
    history_store = _FakeHistoryStoreWithFixedHistory(
        [
            ConversationMessage(role=MessageRole.USER, content="pergunta anterior", timestamp=datetime.now()),
            ConversationMessage(role=MessageRole.ASSISTANT, content="resposta anterior", timestamp=datetime.now()),
        ]
    )
    agent = EmporioMusicaAgent(AgentDependencies(llm_client=llm_client, history_store=history_store))

    agent.handle_message(session_id="s1", user_message="nova pergunta")

    assert llm_client.last_call["messages"] == [
        {"role": "user", "content": "pergunta anterior"},
        {"role": "model", "content": "resposta anterior"},
        {"role": "user", "content": "nova pergunta"},
    ]


def test_handle_message_persists_user_and_assistant_turns(tmp_path: Path) -> None:
    llm_client = _FakeLLMClient()
    history_store = JsonConversationHistoryStore(str(tmp_path))
    agent = EmporioMusicaAgent(AgentDependencies(llm_client=llm_client, history_store=history_store))

    agent.handle_message(session_id="s1", user_message="oi")

    saved = history_store.get_history("s1")
    assert [(m.role, m.content) for m in saved] == [
        (MessageRole.USER, "oi"),
        (MessageRole.ASSISTANT, "resposta do assistente"),
    ]


class _FakeHistoryStoreWithFixedHistory:
    def __init__(self, history: list[ConversationMessage]) -> None:
        self._history = history

    def get_history(self, session_id: str, limit: int | None = None) -> list[ConversationMessage]:
        return self._history

    def add_message(self, session_id: str, message: ConversationMessage) -> None:
        pass

    def clear_session(self, session_id: str) -> None:
        pass
