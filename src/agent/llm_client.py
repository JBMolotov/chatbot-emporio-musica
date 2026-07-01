"""Wrapper fino sobre o SDK do Google Gemini (`google-genai`).
"""

from __future__ import annotations

from typing import Any

from google import genai

from config import Settings, get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLLMClient:
    """Encapsula a criação e o uso do client oficial do Google Gemini."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = genai.Client(api_key=self._settings.google_api_key)

    def generate_response(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Envia uma conversa ao modelo e retorna a resposta.
        """
        raise NotImplementedError(
            "Implementar a chamada ao modelo Gemini "
            f"(model={self._settings.gemini_model!r})."
        )
