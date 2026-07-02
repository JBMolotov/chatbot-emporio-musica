"""Implementação de um índice vetorial em memória, usando FAISS para busca de similaridade.
"""

from __future__ import annotations

import faiss
import numpy as np

from rag.vector_store import BaseVectorStore, VectorDocument

class InMemoryVectorStore(BaseVectorStore):
    """Implementação do vector store usando FAISS em memória."""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        #Não é obrigatório, mas ajuda em delete e persist
        self.embeddings: list[np.ndarray] = []
        self.documents_by_index: list[VectorDocument] = []

    def store_embedding(self, embedding: list[float], document: VectorDocument) -> None:
        """Armazena o embedding no vector store associado ao documento."""
        self.index.add(np.array([embedding], dtype='float32'))
        self.documents_by_index.append(document)

    def similarity_search(self, query: str, top_k: int = 5) -> list[VectorDocument]:
        """Retorna os `top_k` documentos mais similares à consulta."""
        query_embedding = np.array([self.create_embedding(query)], dtype='float32')
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents_by_index[i] for i in indices[0] if i < len(self.documents_by_index)]

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
        """Persiste o estado do vector store."""
        # FAISS não suporta persistência direta, então precisamos salvar os embeddings e documentos
        np.save('embeddings.npy', np.array(self.embeddings, dtype='float32'))
        with open('documents.txt', 'w') as f:
            for doc in self.documents_by_index:
                f.write(f"{doc.id}\t{doc.text}\t{doc.metadata}\n")