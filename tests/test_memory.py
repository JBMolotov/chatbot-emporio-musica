"""Testes do histórico de conversas (`JsonConversationHistoryStore`)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from memory.conversation_history import ConversationMessage, MessageRole
from memory.json_history_store import JsonConversationHistoryStore


@pytest.fixture
def store(tmp_path: Path) -> JsonConversationHistoryStore:
    return JsonConversationHistoryStore(str(tmp_path))


def _message(role: MessageRole, content: str, minute: int = 0) -> ConversationMessage:
    return ConversationMessage(role=role, content=content, timestamp=datetime(2026, 1, 1, 10, minute))


class TestAddMessage:
    def test_add_message_is_json_serializable(self, store: JsonConversationHistoryStore) -> None:
        # Antes da correção, isso levantava TypeError: Object of type datetime is not JSON serializable.
        store.add_message("s1", _message(MessageRole.USER, "oi"))

        assert (store.storage_dir / "s1.json").exists()

    def test_appends_multiple_messages_in_order(self, store: JsonConversationHistoryStore) -> None:
        store.add_message("s1", _message(MessageRole.USER, "oi", minute=0))
        store.add_message("s1", _message(MessageRole.ASSISTANT, "olá!", minute=1))

        history = store.get_history("s1")

        assert [m.content for m in history] == ["oi", "olá!"]


class TestGetHistory:
    def test_unknown_session_returns_empty_list(self, store: JsonConversationHistoryStore) -> None:
        assert store.get_history("nunca_existiu") == []

    def test_limit_returns_most_recent_messages(self, store: JsonConversationHistoryStore) -> None:
        for i in range(3):
            store.add_message("s1", _message(MessageRole.USER, f"msg{i}", minute=i))

        history = store.get_history("s1", limit=2)

        assert [m.content for m in history] == ["msg1", "msg2"]

    def test_limit_zero_returns_empty_list(self, store: JsonConversationHistoryStore) -> None:
        store.add_message("s1", _message(MessageRole.USER, "oi"))

        assert store.get_history("s1", limit=0) == []

    def test_reload_from_disk_after_restart(self, tmp_path: Path) -> None:
        first_process = JsonConversationHistoryStore(str(tmp_path))
        first_process.add_message("s1", _message(MessageRole.USER, "oi"))
        first_process.add_message("s1", _message(MessageRole.ASSISTANT, "olá!", minute=1))

        second_process = JsonConversationHistoryStore(str(tmp_path))
        history = second_process.get_history("s1")

        assert [m.content for m in history] == ["oi", "olá!"]
        assert all(isinstance(m.timestamp, datetime) for m in history)
        assert all(isinstance(m.role, MessageRole) for m in history)


class TestClearSession:
    def test_removes_in_memory_history_and_json_file(self, store: JsonConversationHistoryStore) -> None:
        store.add_message("s1", _message(MessageRole.USER, "oi"))

        store.clear_session("s1")

        assert store.get_history("s1") == []
        assert not (store.storage_dir / "s1.json").exists()

    def test_clears_session_never_loaded_in_this_process(self, tmp_path: Path) -> None:
        first_process = JsonConversationHistoryStore(str(tmp_path))
        first_process.add_message("s1", _message(MessageRole.USER, "oi"))

        second_process = JsonConversationHistoryStore(str(tmp_path))
        second_process.clear_session("s1")

        assert not (tmp_path / "s1.json").exists()
