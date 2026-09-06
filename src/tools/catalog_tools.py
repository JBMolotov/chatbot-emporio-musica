"""Tools de catálogo com validação de parâmetros e resultados JSON seguros."""
from __future__ import annotations

from typing import Any

from tabular_data.service import BaseTabularDataService
from tools.base import BaseTool


class _CatalogTool(BaseTool):
    def __init__(self, tabular_data_service: BaseTabularDataService) -> None:
        self.tabular_data_service = tabular_data_service


class SearchProductsTool(_CatalogTool):
    name = "search_products"
    description = "Pesquisa produtos por nome ou descrição e pode filtrar categoria e preço em reais."
    parameters_schema = {"type": "object", "properties": {
        "query": {"type": "string", "description": "Termo que identifica o produto."},
        "category": {"type": "string", "description": "Categoria opcional."},
        "min_price": {"type": "number", "description": "Preço mínimo em BRL."},
        "max_price": {"type": "number", "description": "Preço máximo em BRL."},
    }, "required": ["query"]}

    def run(self, **kwargs: Any) -> list[dict[str, Any]]:
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("O parâmetro 'query' é obrigatório.")
        return self.tabular_data_service.search_products(
            query, category=kwargs.get("category"), min_price=kwargs.get("min_price"), max_price=kwargs.get("max_price")
        )


class CheckOrderStatusTool(_CatalogTool):
    name = "check_order_status"
    description = "Verifica status, rastreio e previsão de entrega pelo ID do pedido."
    parameters_schema = {"type": "object", "properties": {"order_id": {"type": "integer"}}, "required": ["order_id"]}

    def run(self, **kwargs: Any) -> dict[str, Any] | None:
        if kwargs.get("order_id") is None:
            raise ValueError("O parâmetro 'order_id' é obrigatório.")
        return self.tabular_data_service.get_order_status(int(kwargs["order_id"]))


class CheckStockTool(_CatalogTool):
    name = "check_stock"
    description = "Consulta a quantidade atual em estoque de um produto pelo ID."
    parameters_schema = {"type": "object", "properties": {"product_id": {"type": "integer"}}, "required": ["product_id"]}

    def run(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("product_id") is None:
            raise ValueError("O parâmetro 'product_id' é obrigatório.")
        product_id = int(kwargs["product_id"])
        return {"product_id": product_id, "stock_quantity": self.tabular_data_service.check_stock(product_id)}


class GetActivePromotionsTool(_CatalogTool):
    name = "get_active_promotions"
    description = "Lista promoções ativas; product_id é opcional."
    parameters_schema = {"type": "object", "properties": {"product_id": {"type": "integer"}}}

    def run(self, **kwargs: Any) -> list[dict[str, Any]]:
        product_id = kwargs.get("product_id")
        return self.tabular_data_service.get_active_promotions(int(product_id) if product_id is not None else None)


class GetCustomerOrdersTool(_CatalogTool):
    name = "get_customer_orders"
    description = "Lista pedidos e itens de um cliente autenticado pelo ID informado pelo contexto da sessão."
    parameters_schema = {"type": "object", "properties": {"customer_id": {"type": "integer"}}, "required": ["customer_id"]}

    def run(self, **kwargs: Any) -> list[dict[str, Any]]:
        if kwargs.get("customer_id") is None:
            raise ValueError("O parâmetro 'customer_id' é obrigatório.")
        return self.tabular_data_service.get_customer_orders(int(kwargs["customer_id"]))
