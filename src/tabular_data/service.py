"""Contrato de acesso aos dados de negócio, sempre serializável em JSON."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTabularDataService(ABC):
    @abstractmethod
    def search_products(self, query: str, **filters: Any) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_product_by_id(self, product_id: int) -> dict[str, Any] | None: ...

    @abstractmethod
    def check_stock(self, product_id: int) -> int | None: ...

    @abstractmethod
    def get_active_promotions(self, product_id: int | None = None) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_customer_orders(self, customer_id: int) -> list[dict[str, Any]]: ...

    @abstractmethod
    def get_customer_by_contact(self, contact: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def get_order_status(self, order_id: int) -> dict[str, Any] | None: ...
