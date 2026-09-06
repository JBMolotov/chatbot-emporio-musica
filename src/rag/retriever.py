"""Abstração de recuperação de contexto para o sistema de RAG.
"""

from __future__ import annotations

import re
import unicodedata

from rag.vector_store import BaseVectorStore, VectorDocument


class Retriever:
    """Interface para recuperação de trechos relevantes de documentos."""

    def __init__(self, vector_store: BaseVectorStore, min_similarity: float = 0.55, top_k: int = 4) -> None:
        if not 0 <= min_similarity <= 1:
            raise ValueError("min_similarity deve estar entre 0 e 1")
        self._vector_store = vector_store
        self._min_similarity = min_similarity
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[VectorDocument]:
        """Retorna chunks com fusão semântica + lexical e limiar de confiança."""
        candidates = self._vector_store.similarity_search(query, top_k or self._top_k)
        query_terms = self._terms(query)
        reranked: list[VectorDocument] = []
        for document in candidates:
            semantic = document.score or 0.0
            lexical = len(query_terms & self._terms(document.text)) / len(query_terms) if query_terms else 0.0
            # O sinal lexical protege consultas com nomes, códigos e números;
            # o embedding continua sendo o sinal predominante para linguagem natural.
            score = 0.8 * semantic + 0.2 * lexical
            if score >= self._min_similarity:
                reranked.append(document.with_score(score))
        return sorted(reranked, key=lambda document: document.score or 0.0, reverse=True)

    @staticmethod
    def _terms(text: str) -> set[str]:
        normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
        return {term for term in re.findall(r"[a-z0-9]{3,}", normalized)}
