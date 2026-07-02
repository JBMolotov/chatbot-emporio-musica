"""
Ferramentos para auxiliar na criação de políticas.
"""

from __future__ import annotations

from typing import Any

from rag.retriever import Retriever
from tools.base import BaseTool

class SearchStorePoliciesTool(BaseTool):
    """
    Ferramenta para pesquisar políticas de loja.
    """

    name = "search_store_policies"
    description = "Pesquisa políticas de loja por palavra-chave."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Termo de pesquisa para políticas de loja."},
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def run(self, **kwargs: Any) -> Any:
        query = kwargs.get("query")
        if not query:
            raise ValueError("O parâmetro 'query' é obrigatório.")
        documents = self.retriever.retrieve(query)
        return [
            {"text": doc.text, "metadata": doc.metadata, "score": doc.score}
            for doc in documents
        ]