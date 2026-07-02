"""
Histórico de conversas em memória usando JSON para armazenamento.
Este módulo implementa a interface `BaseConversationHistoryStore` para persistência do histórico de conversas por sessão em memória, com a capacidade de salvar e carregar o histórico em arquivos JSON.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from memory.conversation_history import BaseConversationHistoryStore, ConversationMessage, MessageRole

class JsonConversationHistoryStore(BaseConversationHistoryStore):
    """
    Implementação do armazenamento de histórico de conversas em memória usando JSON.
    """

    def __init__(self, storage_dir: str):
        """
        Inicializa o armazenamento em memória e define o diretório para salvar os arquivos JSON.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history: dict[str, list[ConversationMessage]] = {}

    def add_message(self, session_id: str, message: ConversationMessage) -> None:
        """
        Adiciona uma mensagem ao histórico da sessão.
        """
        self._ensure_loaded(session_id)
        self.history[session_id].append(message)
        self._save_to_json(session_id)

    def get_history(self, session_id: str, limit: int | None = None) -> list[ConversationMessage]:
        """
        Retorna o histórico de mensagens da sessão (mais recentes por último).
        """
        self._ensure_loaded(session_id)
        history = self.history[session_id]
        if limit is None:
            return history
        return history[-limit:] if limit > 0 else []

    def clear_session(self, session_id: str) -> None:
        """
        Remove todo o histórico de uma sessão.
        """
        self.history.pop(session_id, None)
        json_file_path = self._json_path(session_id)
        if json_file_path.exists():
            json_file_path.unlink()

    def _json_path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def _ensure_loaded(self, session_id: str) -> None:
        """Carrega o histórico da sessão do disco na primeira vez que é acessada."""
        if session_id in self.history:
            return

        json_file_path = self._json_path(session_id)
        if not json_file_path.exists():
            self.history[session_id] = []
            return

        with json_file_path.open("r", encoding="utf-8") as f:
            raw_messages = json.load(f)

        self.history[session_id] = [
            ConversationMessage(
                role=MessageRole(raw["role"]),
                content=raw["content"],
                timestamp=datetime.fromisoformat(raw["timestamp"]),
            )
            for raw in raw_messages
        ]

    def _save_to_json(self, session_id: str) -> None:
        """
        Salva o histórico da sessão em um arquivo JSON.
        """
        raw_messages = [
            {
                "role": message.role.value,
                "content": message.content,
                "timestamp": message.timestamp.isoformat(),
            }
            for message in self.history[session_id]
        ]
        with self._json_path(session_id).open("w", encoding="utf-8") as f:
            json.dump(raw_messages, f, ensure_ascii=False, indent=4)