"""Abstração de indexação de documentos para o futuro sistema de RAG.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
import hashlib
from pathlib import Path
import re

from rag.vector_store import BaseVectorStore, VectorDocument


class BaseIndexer(ABC):
    """Interface para indexação de documentos-fonte no vector store."""

    def __init__(self, vector_store: BaseVectorStore, chunk_size: int = 900, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap deve ser menor que chunk_size")
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._indexed_sources: list[Path] = []

    @abstractmethod
    def _read_documents(self, source_path: Path) -> Generator[VectorDocument, None, None]:
        """Lê o conteúdo bruto do documento-fonte e retorna um documento por unidade
        lógica (ex.: uma página de PDF), antes do chunking.
        """
        raise NotImplementedError

    def _chunk_document(self, document: VectorDocument) -> list[VectorDocument]:
        """Divide o documento em chunks menores, se necessário, para indexação."""

        text = re.sub(r"[ \t]+", " ", document.text)
        text = re.sub(r"\n{2,}", "\n", text).strip()
        if not text:
            return []
        # Prefere fronteiras de sentença; fallback é corte duro para textos PDF ruins.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[VectorDocument] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and len(current) + len(sentence) + 1 > self._chunk_size:
                chunks.append(self._make_chunk(document, len(chunks), current))
                current = (current[-self._chunk_overlap:] + " " + sentence).strip()
            else:
                current = f"{current} {sentence}".strip()
            while len(current) > self._chunk_size:
                cut = current.rfind(" ", 0, self._chunk_size)
                cut = cut if cut > self._chunk_size // 2 else self._chunk_size
                chunks.append(self._make_chunk(document, len(chunks), current[:cut]))
                current = current[max(0, cut - self._chunk_overlap):].strip()
        if current:
            chunks.append(self._make_chunk(document, len(chunks), current))
        return chunks

    @staticmethod
    def _make_chunk(document: VectorDocument, number: int, text: str) -> VectorDocument:
        metadata = dict(document.metadata)
        metadata["chunk"] = number
        return VectorDocument(id=f"{document.id}_chunk_{number}", text=text, metadata=metadata)

    def index_documents(self, source_path: Path) -> None:
        """Indexa o documento-fonte no vector store, criando embeddings para cada chunk."""

        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Documento de políticas não encontrado: {source_path}")
        fingerprint = hashlib.sha256(source_path.read_bytes()).hexdigest()
        # Atualizações são idempotentes: evita chunks duplicados e resultados antigos.
        delete_by_source = getattr(self._vector_store, "delete_by_source", None)
        if delete_by_source:
            delete_by_source(str(source_path))
        for document in self._read_documents(source_path):
            document = VectorDocument(document.id, document.text, {
                **document.metadata, "source": str(source_path), "source_fingerprint": fingerprint,
            })
            for chunk in self._chunk_document(document):
                self._vector_store.store_embedding(self._vector_store.create_embedding(chunk.text), chunk)

        if source_path not in self._indexed_sources:
            self._indexed_sources.append(source_path)

    def refresh_index(self) -> None:
        """Reprocessa todas as fontes já indexadas nesta sessão e atualiza o índice."""

        sources = list(self._indexed_sources)
        for source_path in sources:
            self.index_documents(source_path)
