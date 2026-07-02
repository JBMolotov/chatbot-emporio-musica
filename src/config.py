"""Configuração central da aplicação.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações da aplicação, carregadas de variáveis de ambiente / `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Gemini API ----------------------------------------------------------
    google_api_key: str = Field(default="", description="Chave de API do Google AI (Gemini)")
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description=(
            "ID do modelo Gemini usado pelo agente. Conferir os modelos "
            "disponíveis em https://ai.google.dev/gemini-api/docs/models."
        ),
    )
    gemini_max_output_tokens: int = Field(
        default=1024,
        description="Máximo de tokens de saída por resposta do agente",
    )
    gemini_embedding_dim: int = Field(
        default=768,
        description=(
            "Dimensão dos vetores gerados pelo modelo de embedding configurado "
            "(conferir o valor correto para o modelo escolhido em "
            "https://ai.google.dev/gemini-api/docs/embeddings)."
        ),
    )

    # --- Aplicação -----------------------------------------------------------
    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # --- Dados / consulta a dados tabulares ----------------------------------
    # Diretório com os arquivos brutos do desafio: categorias, clientes,
    # itens de pedido, pedidos, produtos e promoções (CSV), além do PDF de
    # políticas da loja usado pelo RAG.
    data_dir: Path = Field(default=Path("data"))
    policies_pdf_path: Path = Field(default=Path("data/políticas.pdf"))

    # --- RAG (futuro) ----------------------------------------------------------
    vector_store_path: Path = Field(default=Path("storage/vector_store"))

    # --- Histórico de conversas (futuro) -----------------------------------------
    conversation_history_path: Path = Field(default=Path("storage/conversation_history"))


def get_settings() -> Settings:
    """Ponto único de acesso às configurações (facilita mock em testes)."""
    return Settings()
