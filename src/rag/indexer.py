"""Abstração de indexação de documentos para o futuro sistema de RAG.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from pathlib import Path

from rag.vector_store import BaseVectorStore, VectorDocument


class BaseIndexer(ABC):
    """Interface para indexação de documentos-fonte no vector store."""

    def __init__(self, vector_store: BaseVectorStore) -> None:
        self._vector_store = vector_store

    @abstractmethod
    def _chunk_document(self, document: VectorDocument) -> list[VectorDocument]:
        """Divide o documento em chunks menores, se necessário, para indexação.
        """
        
        document_chunks = []
        # Lógica de chunking (dividir por parágrafos)
        paragraphs = document.text.split("\n\n")
        for i, paragraph in enumerate(paragraphs):
            chunk_id = f"{document.id}_chunk_{i}"
            chunk = VectorDocument(id=chunk_id, text=paragraph, metadata=document.metadata)
            document_chunks.append(chunk)
        return document_chunks


    @abstractmethod
    def _read_documents(self, source_path: Path) -> Generator[VectorDocument, None, None]:
        """Lê o conteúdo do documento-fonte e retorna uma lista de documentos/chunks.
        """

        for document in self._read_documents(source_path):
            chunks = self._chunk_document(document)
            for chunk in chunks:
                yield VectorDocument(id=chunk.id, text=chunk.text, metadata=chunk.metadata)
        

    @abstractmethod
    def index_documents(self, source_path: Path) -> None:
        """Indexa o documento-fonte no vector store, criando embeddings para cada chunk."""
     
        for document in self._read_documents(source_path):
            chunks = self._chunk_document(document)
            for chunk in chunks:
                embedding = self._vector_store.create_embedding(chunk)
                self._vector_store.store_embedding(embedding)

    @abstractmethod
    def refresh_index(self) -> None:
        """Reprocessa todas as fontes configuradas e atualiza o índice."""
        
        indexed_sources = self._vector_store.get_indexed_sources()
        for source_path in indexed_sources:
            self.index_documents(source_path)
