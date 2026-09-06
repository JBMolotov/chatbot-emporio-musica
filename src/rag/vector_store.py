"""Contratos e tipos do índice vetorial.

O store recebe um provedor de embeddings por injeção de dependência. Isso evita
que a camada de persistência crie clientes de LLM ou leia configuração global.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    def create_embedding(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class VectorDocument:
    """Chunk imutável indexado no RAG; ``score`` é similaridade, de 0 a 1."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    def with_score(self, score: float | None) -> "VectorDocument":
        return replace(self, score=score)


class BaseVectorStore(ABC):
    """Interface de um índice vetorial persistível."""

    def create_embedding(self, text: str) -> list[float]:
        """Cria um embedding para ``text``."""
        # Fallback de compatibilidade; implementações de produção devem injetar
        # um provedor. O import tardio evita acoplamento/ciclo no carregamento.
        from agent.llm_client import GeminiLLMClient
        return GeminiLLMClient().create_embedding(text)

    @abstractmethod
    def store_embedding(self, embedding: list[float], document: VectorDocument) -> None:
        raise NotImplementedError

    def add_documents(self, documents: list[VectorDocument]) -> None:
        for document in documents:
            self.store_embedding(self.create_embedding(document.text), document)

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5) -> list[VectorDocument]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, document_ids: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def persist(self) -> None:
        raise NotImplementedError
