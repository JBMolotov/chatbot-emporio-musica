"""Abstração de armazenamento do histórico de conversas.

Este módulo define a interface `BaseConversationHistoryStore` para persistência do histórico de conversas por sessão.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationMessage:
    """Uma mensagem trocada dentro de uma sessão de atendimento."""

    role: MessageRole
    content: str
    timestamp: datetime


class BaseConversationHistoryStore(ABC):
    """Interface para persistência do histórico de conversas por sessão."""

    @abstractmethod
    def add_message(self, session_id: str, message: ConversationMessage) -> None:
        """Adiciona uma mensagem ao histórico da sessão."""
        raise NotImplementedError

    @abstractmethod
    def get_history(self, session_id: str, limit: int | None = None) -> list[ConversationMessage]:
        """Retorna o histórico de mensagens da sessão (mais recentes por último)."""
        raise NotImplementedError

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """Remove todo o histórico de uma sessão."""
        raise NotImplementedError
