"""Implementação de indexação para o PDF de políticas da loja, usando pypdf.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from pypdf import PdfReader

from rag.indexer import BaseIndexer
from rag.vector_store import VectorDocument


class PdfPolicyIndexer(BaseIndexer):
    """Lê um PDF página a página e produz um `VectorDocument` por página."""

    def _read_documents(self, source_path: Path) -> Generator[VectorDocument, None, None]:
        reader = PdfReader(str(source_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            yield VectorDocument(
                id=f"{source_path.stem}_page_{page_number}",
                text=text,
                metadata={"source": str(source_path), "page": page_number},
            )
