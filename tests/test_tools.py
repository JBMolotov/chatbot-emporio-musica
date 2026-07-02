"""Testes das ferramentas de function calling (`tools`)."""

from __future__ import annotations

import json

import pytest

from config import get_settings
from tabular_data.pandas_service import PandasTabularDataService
from tools.base import BaseTool
from tools.catalog_tools import CheckOrderStatusTool, SearchProductsTool


@pytest.fixture(scope="module")
def service() -> PandasTabularDataService:
    return PandasTabularDataService(settings=get_settings())


class TestBaseTool:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]

    def test_subclass_without_run_cannot_be_instantiated(self) -> None:
        class Incompleta(BaseTool):
            name = "incompleta"
            description = "sem run"
            parameters_schema: dict = {}

        with pytest.raises(TypeError):
            Incompleta()  # type: ignore[abstract]


class TestSearchProductsTool:
    def test_run_returns_json_serializable_list(self, service: PandasTabularDataService) -> None:
        tool = SearchProductsTool(service)

        results = tool.run(query="violão")

        assert isinstance(results, list)
        assert len(results) > 0
        json.dumps(results, allow_nan=False)  # não deve levantar (sem NaN/Infinity)

    def test_run_finds_matches_in_description_too(self, service: PandasTabularDataService) -> None:
        # "Violão" só existe em `description`, não em nenhum `name`.
        tool = SearchProductsTool(service)

        results = tool.run(query="violão")

        assert all("violão" in r["description"].lower() for r in results)

    def test_run_without_query_raises(self, service: PandasTabularDataService) -> None:
        tool = SearchProductsTool(service)

        with pytest.raises(ValueError):
            tool.run()

    def test_to_gemini_function_declaration(self, service: PandasTabularDataService) -> None:
        tool = SearchProductsTool(service)

        declaration = tool.to_gemini_function_declaration()

        assert declaration["name"] == "search_products"
        assert declaration["parameters"]["required"] == ["query"]


class TestCheckOrderStatusTool:
    def test_run_returns_json_serializable_dict_without_nan(self, service: PandasTabularDataService) -> None:
        order_id = int(service.orders_df.iloc[0]["order_id"])
        tool = CheckOrderStatusTool(service)

        result = tool.run(order_id=order_id)

        assert result["order_id"] == order_id
        json.dumps(result, allow_nan=False)  # confirma que NaN virou None

    def test_run_missing_order_returns_none(self, service: PandasTabularDataService) -> None:
        tool = CheckOrderStatusTool(service)

        assert tool.run(order_id=-1) is None

    def test_run_without_order_id_raises(self, service: PandasTabularDataService) -> None:
        tool = CheckOrderStatusTool(service)

        with pytest.raises(ValueError):
            tool.run()
