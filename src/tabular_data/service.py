"""Abstração de consulta aos dados tabulares.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTabularDataService(ABC):
    """Interface para consultas sobre produtos, clientes, pedidos e promoções."""

    @abstractmethod
    def search_products(self, query: str, **filters: Any) -> list[dict[str, Any]]:
        """Busca produtos por nome/descrição/categoria, com filtros opcionais."""
        raise NotImplementedError

    @abstractmethod
    def get_product_by_id(self, product_id: int) -> dict[str, Any] | None:
        """Retorna os dados de um produto específico."""
        raise NotImplementedError

    @abstractmethod
    def check_stock(self, product_id: int) -> int:
        """Retorna a quantidade em estoque de um produto."""
        raise NotImplementedError

    @abstractmethod
    def get_active_promotions(self, product_id: int | None = None) -> list[dict[str, Any]]:
        """Retorna promoções ativas, opcionalmente filtradas por produto."""
        raise NotImplementedError

    @abstractmethod
    def get_customer_orders(self, customer_id: int) -> list[dict[str, Any]]:
        """Retorna o histórico de pedidos de um cliente."""
        raise NotImplementedError

    @abstractmethod
    def get_order_status(self, order_id: int) -> dict[str, Any] | None:
        """Retorna status, rastreio e previsão de entrega de um pedido."""
        raise NotImplementedError
