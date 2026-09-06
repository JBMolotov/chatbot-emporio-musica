"""Implementação local, defensiva e JSON-first do catálogo CSV."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd
from pandas import DataFrame

from config import Settings
from tabular_data import schema
from tabular_data.service import BaseTabularDataService


class PandasTabularDataService(BaseTabularDataService):
    """Carrega os CSVs uma vez e nunca vaza DataFrames pela fronteira pública."""

    def __init__(self, settings: Settings) -> None:
        self.data_dir = settings.data_dir
        self.products_df = self._read(schema.PRODUCTS_FILE, schema.PRODUCTS_COLUMNS)
        self.orders_df = self._read(schema.ORDERS_FILE, schema.ORDERS_COLUMNS)
        self.order_items_df = self._read(schema.ORDER_ITEMS_FILE, schema.ORDER_ITEMS_COLUMNS)
        self.customers_df = self._read(schema.CUSTOMERS_FILE, schema.CUSTOMERS_COLUMNS)
        self.categories_df = self._read(schema.CATEGORIES_FILE, schema.CATEGORIES_COLUMNS)
        self.promotions_df = self._read(schema.PROMOTIONS_FILE, schema.PROMOTIONS_COLUMNS)

    def _read(self, filename: str, columns: list[str]) -> DataFrame:
        path = self.data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de dados ausente: {path}")
        frame = pd.read_csv(path)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"CSV inválido ({path}): colunas ausentes {sorted(missing)}")
        return frame

    @staticmethod
    def _normalize(value: object) -> str:
        value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
        value = re.sub(r"[^a-z0-9]+", " ", value).strip()
        # Stemming leve cobre violão/violões sem introduzir dependência NLP pesada.
        def singular(word: str) -> str:
            # Após remover acentos, "violões" vira "violoes".
            if word.endswith("oes") and len(word) > 4:
                return f"{word[:-3]}ao"
            if word.endswith("es") and len(word) > 4:
                return word[:-2]
            if word.endswith("s") and len(word) > 3:
                return word[:-1]
            return word
        return " ".join(singular(word) for word in value.split())

    @classmethod
    def _matches(cls, series: pd.Series, query: str) -> pd.Series:
        normalized_query = cls._normalize(query)
        if not normalized_query:
            raise ValueError("A busca deve conter ao menos um caractere alfanumérico.")
        terms = normalized_query.split()
        normalized_series = series.fillna("").map(cls._normalize)
        return normalized_series.map(lambda text: all(term in text for term in terms))

    @staticmethod
    def _records(frame: DataFrame) -> list[dict[str, Any]]:
        clean = frame.astype(object).where(frame.notna(), None)
        return clean.to_dict(orient="records")

    def search_products(self, query: str, **filters: Any) -> list[dict[str, Any]]:
        if not isinstance(query, str) or len(query.strip()) > 200:
            raise ValueError("A consulta de produto deve ter entre 1 e 200 caracteres.")
        categories = self.categories_df[["category_id", "name"]].rename(columns={"name": "category"})
        products = self.products_df.merge(categories, on="category_id", how="left")
        mask = self._matches(products["name"], query) | self._matches(products["description"], query)
        category = filters.get("category")
        if category:
            mask &= self._matches(products["category"], str(category))
        for name, comparator in (("min_price", lambda price, value: price >= value),
                                 ("max_price", lambda price, value: price <= value)):
            value = filters.get(name)
            if value is not None:
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name} deve ser numérico.") from exc
                if numeric < 0:
                    raise ValueError(f"{name} não pode ser negativo.")
                mask &= comparator(products["price_brl"], numeric)
        results = products.loc[mask].sort_values(["stock_quantity", "price_brl"], ascending=[False, True])
        records = self._records(results)
        promotions = {p["product_id"]: p for p in self.get_active_promotions()}
        for product in records:
            if product["product_id"] in promotions:
                product["promotion"] = promotions[product["product_id"]]
        return records

    def get_product_by_id(self, product_id: int) -> dict[str, Any] | None:
        records = self._records(self.products_df.loc[self.products_df["product_id"] == int(product_id)])
        return records[0] if records else None

    def check_stock(self, product_id: int) -> int | None:
        product = self.get_product_by_id(product_id)
        return int(product["stock_quantity"]) if product else None

    def get_active_promotions(self, product_id: int | None = None) -> list[dict[str, Any]]:
        promotions = self.promotions_df.loc[self.promotions_df["is_active"].astype(bool)]
        if product_id is not None:
            promotions = promotions.loc[promotions["product_id"] == int(product_id)]
        return self._records(promotions)

    def get_customer_orders(self, customer_id: int) -> list[dict[str, Any]]:
        orders = self.orders_df.loc[self.orders_df["customer_id"] == int(customer_id)]
        if orders.empty:
            return []
        items = self.order_items_df.merge(
            self.products_df[["product_id", "name", "price_brl"]], on="product_id", how="left"
        )
        grouped = {int(order_id): self._records(group[["product_id", "name", "quantity", "price_brl"]])
                   for order_id, group in items.groupby("order_id")}
        result = self._records(orders)
        for order in result:
            order["items"] = grouped.get(int(order["order_id"]), [])
        return result

    def get_customer_by_contact(self, contact: str) -> dict[str, Any] | None:
        normalized = self._normalize(contact)
        if not normalized:
            return None
        matches = self.customers_df.loc[
            self.customers_df["email"].map(self._normalize).eq(normalized)
            | self.customers_df["phone"].map(self._normalize).eq(normalized)
        ]
        records = self._records(matches)
        return records[0] if len(records) == 1 else None

    def get_order_status(self, order_id: int) -> dict[str, Any] | None:
        records = self._records(self.orders_df.loc[self.orders_df["order_id"] == int(order_id)])
        return records[0] if records else None
