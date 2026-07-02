"""
Ferramentas para manipulação de catálogos de produtos, incluindo funções para carregar dados de produtos, promoções e pedidos a partir de arquivos CSV.
"""

from __future__ import annotations

from typing import Any

from tools.base import BaseTool

class SearchProductsTool(BaseTool):
    """
    Ferramenta para pesquisar produtos no catálogo.
    """

    name = "search_products"
    description = "Pesquisa produtos no catálogo por nome ou descrição."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Termo de pesquisa para produtos."},
        },
        "required": ["query"],
    }

    def __init__(self, tabular_data_service):
        self.tabular_data_service = tabular_data_service

    def run(self, **kwargs: Any) -> Any:
        query = kwargs.get("query")
        if not query:
            raise ValueError("O parâmetro 'query' é obrigatório.")
        results = self.tabular_data_service.search_products(query)
        return results.where(results.notna(), None).to_dict(orient="records")
    
class CheckOrderStatusTool(BaseTool):
    """
    Ferramenta para verificar o status de um pedido.
    """

    name = "check_order_status"
    description = "Verifica o status de um pedido pelo ID."
    parameters_schema = {
        "type": "object",
        "properties": {
            "order_id": {"type": "integer", "description": "ID do pedido a ser verificado."},
        },
        "required": ["order_id"],
    }

    def __init__(self, tabular_data_service):
        self.tabular_data_service = tabular_data_service

    def run(self, **kwargs: Any) -> Any:
        order_id = kwargs.get("order_id")
        if order_id is None:
            raise ValueError("O parâmetro 'order_id' é obrigatório.")
        return self.tabular_data_service.get_order_status(order_id)