"""Composição única da aplicação, compartilhada por CLI e API."""
from __future__ import annotations

import hashlib

from agent.core import AgentDependencies, EmporioMusicaAgent
from agent.llm_client import GeminiLLMClient
from config import Settings
from memory.json_history_store import JsonConversationHistoryStore
from rag.in_memory import InMemoryVectorStore
from rag.pdf_indexer import PdfPolicyIndexer
from rag.retriever import Retriever
from tabular_data.pandas_service import PandasTabularDataService
from tools.catalog_tools import (CheckOrderStatusTool, CheckStockTool, GetActivePromotionsTool,
                                 GetCustomerOrdersTool, SearchProductsTool)
from tools.policy_tools import SearchStorePoliciesTool


def build_agent(settings: Settings) -> EmporioMusicaAgent:
    """Monta o grafo de dependências e atualiza o índice se a fonte mudou."""
    embedding_client = GeminiLLMClient(settings)
    vector_store = InMemoryVectorStore(settings.gemini_embedding_dim, settings, embedding_client)
    fingerprint = hashlib.sha256(settings.policies_pdf_path.resolve().read_bytes()).hexdigest()
    if not vector_store.is_source_current(settings.policies_pdf_path.resolve(), fingerprint):
        indexer = PdfPolicyIndexer(vector_store, settings.rag_chunk_size, settings.rag_chunk_overlap)
        indexer.index_documents(settings.policies_pdf_path)
        vector_store.persist()
    retriever = Retriever(vector_store, settings.rag_min_similarity, settings.rag_top_k)
    data = PandasTabularDataService(settings)
    tools = [
        SearchProductsTool(data), CheckOrderStatusTool(data), CheckStockTool(data),
        GetActivePromotionsTool(data), GetCustomerOrdersTool(data), SearchStorePoliciesTool(retriever),
    ]
    return EmporioMusicaAgent(AgentDependencies(
        llm_client=embedding_client, retriever=retriever, tabular_data_service=data,
        history_store=JsonConversationHistoryStore(str(settings.conversation_history_path)),
        tools=tools, settings=settings,
    ))
