"""Índice FAISS local com vetores normalizados e persistência segura."""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import faiss
import numpy as np

from config import Settings, get_settings
from rag.vector_store import BaseVectorStore, EmbeddingProvider, VectorDocument
from utils.logger import get_logger

logger = get_logger(__name__)
_MANIFEST_VERSION = 2


class InMemoryVectorStore(BaseVectorStore):
    """Busca exata por cosseno, adequada ao corpus local pequeno.

    Documentos e vetores são serializados como JSON/NPY, nunca com pickle:
    carregar um diretório de índice não executa código arbitrário.
    """

    def __init__(self, embedding_dim: int, settings: Settings | None = None,
                 embedding_provider: EmbeddingProvider | None = None) -> None:
        self._settings = settings or get_settings()
        self.embedding_dim = embedding_dim
        self._embedding_provider = embedding_provider
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents_by_index: list[VectorDocument] = []
        self._vectors = np.empty((0, embedding_dim), dtype=np.float32)
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._load_if_exists()

    def _path(self, name: str) -> Path:
        return self._settings.vector_store_path / name

    def _load_if_exists(self) -> None:
        manifest_path, documents_path, vectors_path = (
            self._path("manifest.json"), self._path("documents.json"), self._path("vectors.npy")
        )
        if not manifest_path.exists():
            return
        if not documents_path.exists() or not vectors_path.exists():
            logger.warning("Índice incompleto em %s; será reconstruído.", self._settings.vector_store_path)
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("version") != _MANIFEST_VERSION or manifest.get("embedding_dim") != self.embedding_dim:
                logger.warning("Índice incompatível (modelo/dimensão alterado); será reconstruído.")
                return
            raw_documents = json.loads(documents_path.read_text(encoding="utf-8"))
            vectors = np.load(vectors_path, allow_pickle=False).astype(np.float32)
            if vectors.ndim != 2 or vectors.shape != (len(raw_documents), self.embedding_dim):
                raise ValueError("documentos e vetores têm tamanhos incompatíveis")
            self.documents_by_index = [VectorDocument(**doc) for doc in raw_documents]
            self._vectors = vectors
            if len(vectors):
                self.index.add(vectors)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Não foi possível carregar o índice local: %s", exc)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents_by_index, self._vectors = [], np.empty((0, self.embedding_dim), dtype=np.float32)

    def create_embedding(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Não é possível criar embedding para texto vazio.")
        cached = self._query_cache.get(text)
        if cached is not None:
            self._query_cache.move_to_end(text)
            return cached
        provider = self._embedding_provider
        if provider is not None:
            vector = provider.create_embedding(text)
        elif BaseVectorStore.create_embedding.__module__ != "rag.vector_store":
            # Mantém a superfície facilmente mockável em testes de unidade.
            vector = BaseVectorStore.create_embedding(self, text)
        else:
            from agent.llm_client import GeminiLLMClient
            vector = GeminiLLMClient(settings=self._settings).create_embedding(text)
        if len(vector) != self.embedding_dim:
            raise ValueError(f"Embedding com dimensão {len(vector)}; esperado {self.embedding_dim}.")
        self._query_cache[text] = vector
        if len(self._query_cache) > self._settings.embedding_cache_size:
            self._query_cache.popitem(last=False)
        return vector

    def _normalize(self, embedding: list[float] | np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding com dimensão {vector.shape[1]}; esperado {self.embedding_dim}.")
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Embedding inválido: vetor nulo ou não finito.")
        return vector / norm

    def store_embedding(self, embedding: list[float], document: VectorDocument) -> None:
        if not document.id or not document.text.strip():
            raise ValueError("Todo chunk precisa de id e texto não vazio.")
        self.delete([document.id])
        vector = self._normalize(embedding)
        self.index.add(vector)
        self._vectors = np.vstack((self._vectors, vector))
        self.documents_by_index.append(document.with_score(None))

    def similarity_search(self, query: str, top_k: int = 5) -> list[VectorDocument]:
        if top_k <= 0 or self.index.ntotal == 0:
            return []
        query_vector = self._normalize(self.create_embedding(query))
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
        return [self.documents_by_index[int(index)].with_score(float((score + 1.0) / 2.0))
                for score, index in zip(scores[0], indices[0]) if index >= 0]

    def delete(self, document_ids: list[str]) -> None:
        ids = set(document_ids)
        keep = [i for i, document in enumerate(self.documents_by_index) if document.id not in ids]
        if len(keep) == len(self.documents_by_index):
            return
        self.documents_by_index = [self.documents_by_index[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else np.empty((0, self.embedding_dim), dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        if len(self._vectors):
            self.index.add(self._vectors)

    def delete_by_source(self, source: str) -> None:
        self.delete([doc.id for doc in self.documents_by_index if doc.metadata.get("source") == source])

    def is_source_current(self, source: Path, fingerprint: str) -> bool:
        source_docs = [doc for doc in self.documents_by_index if doc.metadata.get("source") == str(source)]
        return bool(source_docs) and all(doc.metadata.get("source_fingerprint") == fingerprint for doc in source_docs)

    def persist(self) -> None:
        directory = self._settings.vector_store_path
        directory.mkdir(parents=True, exist_ok=True)
        self._atomic_json(self._path("documents.json"), [
            {"id": doc.id, "text": doc.text, "metadata": doc.metadata} for doc in self.documents_by_index
        ])
        vectors_tmp = self._path("vectors.npy.tmp")
        with vectors_tmp.open("wb") as file:
            np.save(file, self._vectors, allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(vectors_tmp, self._path("vectors.npy"))
        self._atomic_json(self._path("manifest.json"), {
            "version": _MANIFEST_VERSION, "embedding_dim": self.embedding_dim,
            "documents": len(self.documents_by_index),
        })

    @staticmethod
    def _atomic_json(path: Path, value: object) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, separators=(",", ":"))
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
