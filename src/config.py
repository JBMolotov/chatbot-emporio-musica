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
        default=1024, ge=64, le=8192,
        description="Máximo de tokens de saída por resposta do agente",
    )
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001",
        description="ID do modelo Gemini usado para gerar embeddings (RAG).",
    )
    gemini_embedding_dim: int = Field(
        default=768, ge=128, le=3072,
        description=(
            "Dimensão solicitada dos vetores de embedding via "
            "`output_dimensionality` (o modelo `gemini-embedding-001` "
            "aceita valores menores que o padrão de 3072, ex.: 768 ou 1536)."
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

    # --- RAG -----------------------------------------------------------------
    vector_store_path: Path = Field(default=Path("storage/vector_store"))
    rag_chunk_size: int = Field(default=900, ge=200, le=4000)
    rag_chunk_overlap: int = Field(default=120, ge=0, le=1000)
    rag_top_k: int = Field(default=4, ge=1, le=20)
    rag_min_similarity: float = Field(default=0.55, ge=0.0, le=1.0)
    embedding_cache_size: int = Field(default=256, ge=0, le=10000)

    # --- Histórico de conversas ----------------------------------------------
    conversation_history_path: Path = Field(default=Path("storage/conversation_history"))
    max_history_messages: int = Field(default=20, ge=0, le=200)
    max_message_characters: int = Field(default=8000, ge=100, le=50000)

    # --- API -----------------------------------------------------------------
    api_key: str = Field(default="", repr=False)


def get_settings() -> Settings:
    """Ponto único de acesso às configurações (facilita mock em testes)."""
    return Settings()
