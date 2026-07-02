"""
Serviço de dados tabulares usando pandas para manipulação de DataFrames.
"""

from typing import Any, Dict, Optional

import pandas as pd
from pandas import DataFrame

from tabular_data.service import BaseTabularDataService


class PandasTabularDataService(BaseTabularDataService):
    """
    Serviço de dados tabulares usando pandas para manipulação de DataFrames.
    """

    def __init__(self, settings: Any):
        """
        Inicializa o serviço carregando os CSVs em DataFrames.
        """
        self.data_dir = settings.data_dir
        self.products_df = pd.read_csv(f"{self.data_dir}/products.csv")
        self.orders_df = pd.read_csv(f"{self.data_dir}/orders.csv")
        self.order_items_df = pd.read_csv(f"{self.data_dir}/order_items.csv")
        self.customers_df = pd.read_csv(f"{self.data_dir}/customers.csv")
        self.categories_df = pd.read_csv(f"{self.data_dir}/categories.csv")
        self.promotions_df = pd.read_csv(f"{self.data_dir}/promotions.csv")

    def search_products(self, query: str, **filters) -> DataFrame:
        """
        Pesquisa produtos por nome/descrição e aplica filtros opcionais.
        """
        results = self.products_df[self.products_df["name"].str.contains(query, case=False, na=False)]
        return results

    def get_product_by_id(self, product_id: int) -> Optional[Dict]:
        """
        Retorna o produto pelo ID.
        """
        product = self.products_df[self.products_df["product_id"] == product_id]
        return product.to_dict(orient="records")[0] if not product.empty else None

    def check_stock(self, product_id: int) -> Optional[int]:
        """
        Verifica a quantidade em estoque do produto.
        """
        product = self.get_product_by_id(product_id)
        return product.get("stock_quantity") if product else None

    def get_active_promotions(self, product_id: Optional[int] = None) -> DataFrame:
        """
        Retorna promoções ativas, opcionalmente filtradas por produto.
        """
        promotions = self.promotions_df[self.promotions_df["is_active"] == 1]
        if product_id is not None:
            promotions = promotions[promotions["product_id"] == product_id]
        return promotions

    def get_customer_orders(self, customer_id: int) -> DataFrame:
        """
        Retorna pedidos do cliente com detalhes dos itens.
        """
        customer_orders = self.orders_df[self.orders_df["customer_id"] == customer_id]
        return customer_orders

    def get_order_status(self, order_id: int) -> Optional[Dict]:
        """
        Retorna o status do pedido pelo ID.
        """
        order = self.orders_df[self.orders_df["order_id"] == order_id]
        return order.to_dict(orient="records")[0] if not order.empty else None