"""Abstração de um vector store para o sistema de RAG.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorDocument:
    """Representa um chunk de documento a ser indexado/recuperado."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


class BaseVectorStore(ABC):
    """Interface para um armazenamento de vetores (embeddings)."""

    @abstractmethod
    def add_documents(self, documents: list[VectorDocument]) -> None:
        """Adiciona (ou atualiza) documentos/chunks no índice vetorial."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5) -> list[VectorDocument]:
        """Retorna os `top_k` documentos mais similares à consulta."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, document_ids: list[str]) -> None:
        """Remove documentos do índice pelos seus IDs."""
        raise NotImplementedError

    @abstractmethod
    def persist(self) -> None:
        """Persiste o índice em disco (ver `Settings.vector_store_path`)."""
        raise NotImplementedError
