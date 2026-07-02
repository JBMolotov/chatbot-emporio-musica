"""Abstração de consulta aos dados tabulares.
"""

from __future__ import annotations

from typing import Any


class BaseTabularDataService():
    """Interface para consultas sobre produtos, clientes, pedidos e promoções."""

    def search_products(self, query: str, **filters: Any) -> list[dict[str, Any]]:
        """Busca produtos por nome/descrição/categoria, com filtros opcionais."""
    

    def get_product_by_id(self, product_id: int) -> dict[str, Any] | None:
        """Retorna os dados de um produto específico."""

    def check_stock(self, product_id: int) -> int:
        """Retorna a quantidade em estoque de um produto."""

    def get_active_promotions(self, product_id: int | None = None) -> list[dict[str, Any]]:
        """Retorna promoções ativas, opcionalmente filtradas por produto."""

    def get_customer_orders(self, customer_id: int) -> list[dict[str, Any]]:
        """Retorna o histórico de pedidos de um cliente."""

    def get_order_status(self, order_id: int) -> dict[str, Any] | None:
        """Retorna status, rastreio e previsão de entrega de um pedido."""
