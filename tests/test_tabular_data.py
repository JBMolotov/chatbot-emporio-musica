"""Testes do serviço de dados tabulares (`PandasTabularDataService`).

As expectativas são derivadas dos próprios DataFrames carregados (em vez de
valores fixos), para não quebrar se os CSVs de `data/` mudarem.
"""

from __future__ import annotations

import pytest

from config import get_settings
from tabular_data.pandas_service import PandasTabularDataService


@pytest.fixture(scope="module")
def service() -> PandasTabularDataService:
    return PandasTabularDataService(settings=get_settings())


class TestSearchProducts:
    def test_finds_by_name_case_insensitive(self, service: PandasTabularDataService) -> None:
        query = service.products_df.iloc[0]["name"].split()[0].lower()

        results = service.search_products(query)

        assert not results.empty
        matches_name = results["name"].str.contains(query, case=False)
        matches_description = results["description"].str.contains(query, case=False, na=False)
        assert (matches_name | matches_description).all()

    def test_finds_by_description_when_name_does_not_match(self, service: PandasTabularDataService) -> None:
        # "Violão" não aparece em nenhum `name` (os produtos são nomeados por
        # marca/modelo), só em `description` — cobre a busca combinada.
        results = service.search_products("violão")

        assert not results.empty
        assert results["description"].str.contains("violão", case=False, na=False).all()

    def test_no_match_returns_empty(self, service: PandasTabularDataService) -> None:
        results = service.search_products("xxxxxxxxxxxxnaoexiste")

        assert results.empty


class TestGetProductById:
    def test_returns_dict_for_existing_product(self, service: PandasTabularDataService) -> None:
        product_id = int(service.products_df.iloc[0]["product_id"])

        product = service.get_product_by_id(product_id)

        assert product is not None
        assert product["product_id"] == product_id

    def test_returns_none_for_missing_product(self, service: PandasTabularDataService) -> None:
        assert service.get_product_by_id(-1) is None


class TestCheckStock:
    def test_matches_stock_quantity_column(self, service: PandasTabularDataService) -> None:
        row = service.products_df.iloc[0]

        stock = service.check_stock(int(row["product_id"]))

        assert stock == int(row["stock_quantity"])

    def test_returns_none_for_missing_product(self, service: PandasTabularDataService) -> None:
        assert service.check_stock(-1) is None


class TestGetActivePromotions:
    def test_only_returns_active_promotions(self, service: PandasTabularDataService) -> None:
        promotions = service.get_active_promotions()

        assert not promotions.empty
        assert (promotions["is_active"] == 1).all()

    def test_filters_by_product_id(self, service: PandasTabularDataService) -> None:
        active_row = service.promotions_df[service.promotions_df["is_active"] == 1].iloc[0]
        product_id = int(active_row["product_id"])

        promotions = service.get_active_promotions(product_id=product_id)

        assert not promotions.empty
        assert (promotions["product_id"] == product_id).all()


class TestGetCustomerOrders:
    def test_filters_by_customer_id(self, service: PandasTabularDataService) -> None:
        customer_id = int(service.orders_df.iloc[0]["customer_id"])

        orders = service.get_customer_orders(customer_id)

        assert not orders.empty
        assert (orders["customer_id"] == customer_id).all()


class TestGetOrderStatus:
    def test_returns_dict_for_existing_order(self, service: PandasTabularDataService) -> None:
        order_id = int(service.orders_df.iloc[0]["order_id"])

        order = service.get_order_status(order_id)

        assert order is not None
        assert order["order_id"] == order_id

    def test_returns_none_for_missing_order(self, service: PandasTabularDataService) -> None:
        assert service.get_order_status(-1) is None
