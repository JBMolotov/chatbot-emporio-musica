"""Implementação de um índice vetorial em memória, usando FAISS para busca de similaridade.
"""

from __future__ import annotations

import pickle

import faiss
import numpy as np

from config import Settings, get_settings
from rag.vector_store import BaseVectorStore, VectorDocument

class InMemoryVectorStore(BaseVectorStore):
    """Implementação do vector store usando FAISS em memória."""

    def __init__(self, embedding_dim: int, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents_by_index: list[VectorDocument] = []
        self._load_if_exists()

    def _index_path(self):
        return self._settings.vector_store_path / "index.faiss"

    def _documents_path(self):
        return self._settings.vector_store_path / "documents.pkl"

    def _load_if_exists(self) -> None:
        if self._index_path().exists() and self._documents_path().exists():
            self.index = faiss.read_index(str(self._index_path()))
            with self._documents_path().open("rb") as f:
                self.documents_by_index = pickle.load(f)

    def store_embedding(self, embedding: list[float], document: VectorDocument) -> None:
        """Armazena o embedding no vector store associado ao documento."""
        self.index.add(np.array([embedding], dtype='float32'))
        self.documents_by_index.append(document)

    def similarity_search(self, query: str, top_k: int = 5) -> list[VectorDocument]:
        """Retorna os `top_k` documentos mais similares à consulta."""
        if self.index.ntotal == 0:
            return []
        query_embedding = np.array([self.create_embedding(query)], dtype='float32')
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for distance, i in zip(distances[0], indices[0]):
            if i < 0:
                continue
            document = self.documents_by_index[i]
            document.score = float(distance)
            results.append(document)
        return results

    def delete(self, document_ids: list[str]) -> None:
        """Remove documentos do índice pelos seus IDs."""
        # FAISS não suporta remoção direta, então precisamos reconstruir o índice
        remaining_documents = [doc for doc in self.documents_by_index if doc.id not in document_ids]
        self.index.reset()
        self.documents_by_index = []
        for doc in remaining_documents:
            embedding = self.create_embedding(doc.text)
            self.store_embedding(embedding, doc)

    def persist(self) -> None:
        """Persiste o estado do vector store em `Settings.vector_store_path`."""
        self._settings.vector_store_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_path()))
        with self._documents_path().open('wb') as f:
            pickle.dump(self.documents_by_index, f)