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
        self._indexed_sources: list[Path] = []

    @abstractmethod
    def _read_documents(self, source_path: Path) -> Generator[VectorDocument, None, None]:
        """Lê o conteúdo bruto do documento-fonte e retorna um documento por unidade
        lógica (ex.: uma página de PDF), antes do chunking.
        """
        raise NotImplementedError

    def _chunk_document(self, document: VectorDocument) -> list[VectorDocument]:
        """Divide o documento em chunks menores, se necessário, para indexação."""

        document_chunks = []
        # Lógica de chunking (dividir por parágrafos)
        paragraphs = document.text.split("\n\n")
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            chunk_id = f"{document.id}_chunk_{i}"
            chunk = VectorDocument(id=chunk_id, text=paragraph, metadata=document.metadata)
            document_chunks.append(chunk)
        return document_chunks

    def index_documents(self, source_path: Path) -> None:
        """Indexa o documento-fonte no vector store, criando embeddings para cada chunk."""

        for document in self._read_documents(source_path):
            for chunk in self._chunk_document(document):
                embedding = self._vector_store.create_embedding(chunk.text)
                self._vector_store.store_embedding(embedding, chunk)

        if source_path not in self._indexed_sources:
            self._indexed_sources.append(source_path)

    def refresh_index(self) -> None:
        """Reprocessa todas as fontes já indexadas nesta sessão e atualiza o índice."""

        sources = list(self._indexed_sources)
        for source_path in sources:
            self.index_documents(source_path)
