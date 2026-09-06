"""Backend Chroma opcional, compatível com :class:`BaseVectorStore`.

O projeto usa FAISS por padrão para o modo local. Este backend existe para
quando a persistência/filtragem nativa do Chroma for preferível, sem alterar o
indexador, retriever ou tools. Os embeddings continuam sendo produzidos pelo
provedor configurado pela aplicação; Chroma não baixa um modelo implícito.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from config import Settings, get_settings
from rag.vector_store import BaseVectorStore, EmbeddingProvider, VectorDocument


class ChromaVectorStore(BaseVectorStore):
    """Store Chroma persistente com distância cosseno e upsert por ID.

    Requer o extra ``chroma``. ``PersistentClient`` persiste automaticamente;
    portanto :meth:`persist` é propositalmente uma operação sem efeito.
    """

    def __init__(
        self,
        embedding_dim: int,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        collection_name: str = "emporio_policies",
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # erro acionável para quem escolheu o backend
            raise ImportError("Chroma não instalado. Execute: pip install -e '.[chroma]'") from exc
        if not collection_name.replace("_", "").isalnum() or len(collection_name) < 3:
            raise ValueError("collection_name deve ter ao menos 3 caracteres alfanuméricos, '-' ou '_'.")
        self._settings = settings or get_settings()
        self.embedding_dim = embedding_dim
        self._embedding_provider = embedding_provider
        self._client = chromadb.PersistentClient(path=str(self._settings.vector_store_path / "chroma"))
        # embedding_function=None força o uso explícito e consistente do Gemini.
        self._collection = self._client.get_or_create_collection(
            name=collection_name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def create_embedding(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Não é possível criar embedding para texto vazio.")
        if self._embedding_provider is None:
            from agent.llm_client import GeminiLLMClient
            vector = GeminiLLMClient(settings=self._settings).create_embedding(text)
        else:
            vector = self._embedding_provider.create_embedding(text)
        if len(vector) != self.embedding_dim:
            raise ValueError(f"Embedding com dimensão {len(vector)}; esperado {self.embedding_dim}.")
        norm = math.sqrt(sum(float(value) ** 2 for value in vector))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Embedding inválido: vetor nulo ou não finito.")
        return [float(value) / norm for value in vector]

    @staticmethod
    def _metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """Chroma aceita apenas metadados escalares; rejeita entrada ambígua cedo."""
        clean: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            is_scalar = isinstance(value, (str, int, float, bool))
            is_valid_float = not isinstance(value, float) or math.isfinite(value)
            if is_scalar and is_valid_float:
                clean[str(key)] = value
            elif value is not None:
                raise ValueError(f"Metadado '{key}' não é escalar e não pode ser salvo no Chroma.")
        return clean

    def store_embedding(self, embedding: list[float], document: VectorDocument) -> None:
        if not document.id or not document.text.strip():
            raise ValueError("Todo chunk precisa de id e texto não vazio.")
        normalized = self.create_embedding_from_vector(embedding)
        self._collection.upsert(
            ids=[document.id], embeddings=[normalized], documents=[document.text], metadatas=[self._metadata(document.metadata)]
        )

    def create_embedding_from_vector(self, embedding: list[float]) -> list[float]:
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding com dimensão {len(embedding)}; esperado {self.embedding_dim}.")
        norm = math.sqrt(sum(float(value) ** 2 for value in embedding))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Embedding inválido: vetor nulo ou não finito.")
        return [float(value) / norm for value in embedding]

    def similarity_search(self, query: str, top_k: int = 5) -> list[VectorDocument]:
        if top_k <= 0 or self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self.create_embedding(query)], n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            VectorDocument(id=identifier, text=text or "", metadata=metadata or {}, score=max(0.0, 1.0 - float(distance)))
            for identifier, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def delete(self, document_ids: list[str]) -> None:
        if document_ids:
            self._collection.delete(ids=document_ids)

    def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def is_source_current(self, source: Path, fingerprint: str) -> bool:
        found = self._collection.get(where={"source": str(source)}, include=["metadatas"])
        metadata = found.get("metadatas") or []
        return bool(metadata) and all(item and item.get("source_fingerprint") == fingerprint for item in metadata)

    def persist(self) -> None:
        """Chroma PersistentClient grava automaticamente cada mutação."""
