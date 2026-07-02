"""Dicionário de dados dos arquivos brutos do desafio (`data/*.csv`).
"""

from __future__ import annotations

# Nomes dos arquivos brutos, relativos a `Settings.data_dir`.
CATEGORIES_FILE = "categories.csv"
CUSTOMERS_FILE = "customers.csv"
ORDER_ITEMS_FILE = "order_items.csv"
ORDERS_FILE = "orders.csv"
PRODUCTS_FILE = "products.csv"
PROMOTIONS_FILE = "promotions.csv"

# Colunas de cada tabela, na ordem em que aparecem no CSV de origem.
CATEGORIES_COLUMNS = ["category_id", "name", "description"]
CUSTOMERS_COLUMNS = ["customer_id", "name", "phone", "email", "city"]
ORDER_ITEMS_COLUMNS = ["order_id", "quantity", "product_id"]
ORDERS_COLUMNS = [
    "order_id",
    "customer_id",
    "order_date",
    "status",
    "total_brl",
    "payment_method",
    "tracking_code",
    "estimated_delivery",
    "notes",
]
PRODUCTS_COLUMNS = [
    "product_id",
    "price_brl",
    "name",
    "category_id",
    "description",
    "stock_quantity",
    "status",
    "specs",
    "created_at",
]
PROMOTIONS_COLUMNS = [
    "promotion_id",
    "product_id",
    "discount_percent",
    "description",
    "is_active",
]
