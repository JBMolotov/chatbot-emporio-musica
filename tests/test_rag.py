"""Testes do sistema de RAG (vector store, indexer e retriever).

Nenhum teste aqui chama a API real do Gemini: `BaseVectorStore.create_embedding`
é substituído por um embedding determinístico (hash do texto) via fixture
`autouse`, o que também evita a necessidade de uma `GOOGLE_API_KEY` válida.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from config import Settings
from rag.indexer import BaseIndexer
from rag.in_memory import InMemoryVectorStore
from rag.pdf_indexer import PdfPolicyIndexer
from rag.retriever import Retriever
from rag.vector_store import BaseVectorStore, VectorDocument

POLICIES_PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "políticas.pdf"


def _fake_embedding(self: BaseVectorStore, text: str) -> list[float]:
    """Embedding determinístico (hash do texto) — só para isolar os testes da API real."""
    digest = hashlib.sha256(text.encode()).digest()[:8]
    return [b / 255 for b in digest]


@pytest.fixture(autouse=True)
def _no_real_gemini_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BaseVectorStore, "create_embedding", _fake_embedding)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(google_api_key="test-key", vector_store_path=tmp_path / "vector_store")


class TestVectorDocument:
    def test_defaults(self) -> None:
        doc = VectorDocument(id="1", text="olá")
        assert doc.metadata == {}
        assert doc.score is None


class TestBaseVectorStore:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseVectorStore()  # type: ignore[abstract]


class TestInMemoryVectorStore:
    def test_similarity_search_orders_by_distance(self, settings: Settings) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        near = VectorDocument(id="near", text="troca e devolução de produtos")
        far = VectorDocument(id="far", text="totalmente outro assunto qualquer")
        store.store_embedding(store.create_embedding(near.text), near)
        store.store_embedding(store.create_embedding(far.text), far)

        results = store.similarity_search("troca e devolução de produtos", top_k=2)

        assert [doc.id for doc in results] == ["near", "far"]
        assert results[0].score == pytest.approx(1.0, abs=1e-6)

    def test_similarity_search_respects_top_k(self, settings: Settings) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        for i in range(5):
            doc = VectorDocument(id=str(i), text=f"documento {i}")
            store.store_embedding(store.create_embedding(doc.text), doc)

        results = store.similarity_search("documento", top_k=2)

        assert len(results) == 2

    def test_similarity_search_on_empty_store_returns_empty_list(self, settings: Settings) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        assert store.similarity_search("qualquer coisa") == []

    def test_persist_and_reload_restores_state(self, settings: Settings) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        doc = VectorDocument(id="a", text="documento persistido")
        store.store_embedding(store.create_embedding(doc.text), doc)
        store.persist()

        reloaded = InMemoryVectorStore(embedding_dim=8, settings=settings)

        assert [doc.id for doc in reloaded.documents_by_index] == ["a"]
        assert reloaded.index.ntotal == 1

    def test_delete_removes_document_and_rebuilds_index(self, settings: Settings) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        keep = VectorDocument(id="keep", text="documento a manter")
        drop = VectorDocument(id="drop", text="documento a remover")
        store.store_embedding(store.create_embedding(keep.text), keep)
        store.store_embedding(store.create_embedding(drop.text), drop)

        store.delete(["drop"])

        assert [doc.id for doc in store.documents_by_index] == ["keep"]
        assert store.index.ntotal == 1

    def test_store_replaces_same_document_id(self, settings: Settings) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        store.store_embedding(store.create_embedding("primeira"), VectorDocument(id="same", text="primeira"))
        store.store_embedding(store.create_embedding("segunda"), VectorDocument(id="same", text="segunda"))
        assert [document.text for document in store.documents_by_index] == ["segunda"]

    def test_search_does_not_mutate_stored_document_score(self, settings: Settings) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        document = VectorDocument(id="a", text="texto")
        store.store_embedding(store.create_embedding(document.text), document)
        store.similarity_search("texto")
        assert store.documents_by_index[0].score is None


class _FakeIndexer(BaseIndexer):
    """Indexer de teste: devolve documentos fixos em vez de ler um arquivo real."""

    def __init__(self, vector_store: InMemoryVectorStore, raw_documents: list[VectorDocument]) -> None:
        super().__init__(vector_store)
        self._raw_documents = raw_documents

    def _read_documents(self, source_path: Path):
        yield from self._raw_documents


class TestBaseIndexer:
    def test_chunk_document_splits_by_paragraph_and_skips_blank(self, settings: Settings) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        indexer = _FakeIndexer(store, [])
        document = VectorDocument(id="doc", text="primeiro parágrafo\n\n\n\nsegundo parágrafo")

        chunks = indexer._chunk_document(document)

        assert [c.text for c in chunks] == ["primeiro parágrafo\nsegundo parágrafo"]
        assert [c.id for c in chunks] == ["doc_chunk_0"]

    def test_index_documents_stores_one_embedding_per_chunk(self, settings: Settings, tmp_path: Path) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        raw = [VectorDocument(id="doc", text="um parágrafo\n\noutro parágrafo")]
        indexer = _FakeIndexer(store, raw)
        source_path = tmp_path / "fonte.txt"
        source_path.write_text("fonte", encoding="utf-8")

        indexer.index_documents(source_path)

        assert len(store.documents_by_index) == 1
        assert source_path in indexer._indexed_sources

    def test_refresh_index_reprocesses_already_indexed_sources(self, settings: Settings, tmp_path: Path) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        raw = [VectorDocument(id="doc", text="um parágrafo")]
        indexer = _FakeIndexer(store, raw)
        source_path = tmp_path / "fonte.txt"
        source_path.write_text("fonte", encoding="utf-8")
        indexer.index_documents(source_path)

        indexer.refresh_index()

        assert len(store.documents_by_index) == 1


class TestPdfPolicyIndexer:
    def test_reads_real_policies_pdf(self, settings: Settings) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        indexer = PdfPolicyIndexer(store)

        indexer.index_documents(POLICIES_PDF_PATH)

        assert len(store.documents_by_index) > 0
        first_chunk = store.documents_by_index[0]
        assert first_chunk.metadata["source"] == str(POLICIES_PDF_PATH)
        assert first_chunk.metadata["page"] == 1


class TestRetriever:
    def test_retrieve_delegates_to_vector_store_similarity_search(self, settings: Settings) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        doc = VectorDocument(id="a", text="política de troca em até 30 dias")
        store.store_embedding(store.create_embedding(doc.text), doc)
        retriever = Retriever(store)

        results = retriever.retrieve("política de troca em até 30 dias", top_k=1)

        assert [doc.id for doc in results] == ["a"]

    def test_filters_low_similarity(self, settings: Settings) -> None:
        store = InMemoryVectorStore(embedding_dim=8, settings=settings)
        document = VectorDocument(id="a", text="política de troca")
        store.store_embedding(store.create_embedding(document.text), document)
        assert Retriever(store, min_similarity=1.0).retrieve("assunto distante") == []
