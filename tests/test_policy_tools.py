"""Testes de `SearchStorePoliciesTool`."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from config import Settings
from rag.in_memory import InMemoryVectorStore
from rag.retriever import Retriever
from rag.vector_store import BaseVectorStore, VectorDocument
from tools.policy_tools import SearchStorePoliciesTool


def _fake_embedding(self: BaseVectorStore, text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()[:8]
    return [b / 255 for b in digest]


@pytest.fixture(autouse=True)
def _no_real_gemini_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BaseVectorStore, "create_embedding", _fake_embedding)


@pytest.fixture
def tool(tmp_path: Path) -> SearchStorePoliciesTool:
    settings = Settings(google_api_key="test-key", vector_store_path=tmp_path)
    store = InMemoryVectorStore(embedding_dim=8, settings=settings)
    doc = VectorDocument(id="a", text="política de troca em até 30 dias", metadata={"page": 4})
    store.store_embedding(store.create_embedding(doc.text), doc)
    return SearchStorePoliciesTool(Retriever(store))


class TestSearchStorePoliciesTool:
    def test_run_returns_json_serializable_list_of_dicts(self, tool: SearchStorePoliciesTool) -> None:
        results = tool.run(query="política de troca em até 30 dias")

        assert isinstance(results, list)
        assert results[0]["text"] == "política de troca em até 30 dias"
        assert results[0]["metadata"] == {"page": 4}
        json.dumps(results, allow_nan=False)

    def test_run_without_query_raises(self, tool: SearchStorePoliciesTool) -> None:
        with pytest.raises(ValueError):
            tool.run()
