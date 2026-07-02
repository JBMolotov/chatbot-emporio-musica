"""Abstração de um vector store para o sistema de RAG.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent.llm_client import GeminiLLMClient

@dataclass
class VectorDocument:
    """Representa um chunk de documento a ser indexado/recuperado."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    
class BaseVectorStore(ABC):
    """Interface para um armazenamento de vetores (embeddings)."""

    def create_embedding(self, text: str) -> list[float]:
        """Cria um embedding para o texto fornecido."""

        return GeminiLLMClient().create_embedding(text)
    
    def add_documents(self, documents: list[VectorDocument]) -> None:
        """Adiciona (ou atualiza) documentos/chunks no índice vetorial."""
        
        for document in documents:
            embedding = self.create_embedding(document.text)
            self.store_embedding(embedding, document)

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
