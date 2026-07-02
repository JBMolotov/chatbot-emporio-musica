"""Abstração de indexação de documentos para o futuro sistema de RAG.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from rag.vector_store import BaseVectorStore


class BaseIndexer(ABC):
    """Interface para indexação de documentos-fonte no vector store."""

    def __init__(self, vector_store: BaseVectorStore) -> None:
        self._vector_store = vector_store

    @abstractmethod
    def index_document(self, source_path: Path) -> None:
        """Lê, faz o chunking e indexa um documento-fonte (ex o PDF de políticas)."""
        raise NotImplementedError

    @abstractmethod
    def refresh_index(self) -> None:
        """Reprocessa todas as fontes configuradas e atualiza o índice."""
        raise NotImplementedError
