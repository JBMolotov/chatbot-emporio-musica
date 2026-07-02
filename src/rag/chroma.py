# """Implementação do vector store usando Chroma.
# """

# from __future__ import annotations

# from abc import ABC, abstractmethod
# from dataclasses import dataclass, field
# from typing import Any

# from vector_store import BaseVectorStore, VectorDocument
# import chromadb
# from chromadb import Client #Não precisa persistir
# #from chromadb import ClientPersist #Precisa persistir
# from chromadb.config import Settings as ChromaSettings


# class ChromaVectorStore(BaseVectorStore):
#     """Implementação do vector store usando Chroma."""

#     def __init__(self, collection_name: str):
#         self.collection_name = collection_name
#         # Inicialize a conexão com o Chroma aqui (exemplo fictício)
#         self.chroma_client = self._initialize_chroma_client()

#     def _initialize_chroma_client(self):
#         # Lógica para inicializar o cliente Chroma
#         return Client(settings=ChromaSettings())

#     def store_embedding(self, embedding: list[float], document: VectorDocument) -> None:
#         """Armazena o embedding no vector store associado ao documento."""
        
#         # try:
#         #     self.chroma_client.get_collection(self.collection_name)
#         # except Exception:
#         #     self.chroma_client.create_collection(self.collection_name)

#         self.chroma_client.add(
#             collection_name=self.collection_name,
#             documents=[document.text],
#             metadatas=[document.metadata],
#             ids=[document.id],
#             embeddings=[embedding]
#         )

#     def similarity_search(self, query: str, top_k: int = 5) -> list[VectorDocument]:
#         """Retorna os `top_k` documentos mais similares à consulta."""
#         # Lógica para realizar a busca de similaridade no Chroma
#         pass

#     def delete(self, document_ids: list[str]) -> None:
#         """Remove documentos do índice pelos seus IDs."""
#         # Lógica para deletar documentos do Chroma
#         pass

#     def persist(self) -> None:
#         """Persiste o estado do vector store."""
#         # Lógica para persistir o estado do Chroma
#         pass

