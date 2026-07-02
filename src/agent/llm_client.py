"""Wrapper fino sobre o SDK do Google Gemini (`google-genai`).

Mantém toda a interação direta com `genai.Client` isolada em um único
lugar, para que `agent.core` dependa apenas desta interface.

TODO (implementação futura):
    - Implementar `generate_response`, decidindo entre chamada simples
      (`client.models.generate_content`) e streaming
      (`client.models.generate_content_stream`) conforme o tamanho
      esperado da resposta.
    - Definir `generation_config` (temperature, max_output_tokens etc.) e,
      se necessário, `safety_settings`.
    - Tratar `finish_reason` (`STOP`, `MAX_TOKENS`, `SAFETY`, chamadas de
      function calling, etc.).
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

        Ponto de extensão: aqui entra a chamada real a
        `self._client.models.generate_content(...)` (ou
        `.generate_content_stream(...)`), a instrução de sistema final e o
        eventual loop de function calling.
        """
        raise NotImplementedError(
            "Implementar a chamada ao modelo Gemini "
            f"(model={self._settings.gemini_model!r})."
        )
