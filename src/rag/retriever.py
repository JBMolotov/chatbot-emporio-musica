"""Abstração de recuperação de contexto para o sistema de RAG.
"""

from __future__ import annotations

from rag.vector_store import BaseVectorStore, VectorDocument


class Retriever:
    """Interface para recuperação de trechos relevantes de documentos."""

    def __init__(self, vector_store: BaseVectorStore) -> None:
        self._vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[VectorDocument]:
        """Retorna os trechos mais relevantes para `query`."""
        
        return self._vector_store.similarity_search(query, top_k)
