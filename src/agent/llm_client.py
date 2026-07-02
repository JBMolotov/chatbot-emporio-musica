"""Wrapper fino sobre o SDK do Google Gemini (`google-genai`).

Mantém toda a interação direta com `genai.Client` isolada em um único
lugar, para que `agent.core` dependa apenas desta interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from google import genai
from google.genai import types

from config import Settings, get_settings
from utils.logger import get_logger

if TYPE_CHECKING:
    from tools.base import BaseTool

logger = get_logger(__name__)

# Limite de idas-e-voltas de function calling numa mesma pergunta, para
# evitar loop infinito caso o modelo insista em chamar tools.
_MAX_FUNCTION_CALL_ROUNDS = 5

class EmbeddingModel(str):
    """Enumeração de modelos de embedding do Gemini.

    Usar `str` como superclasse permite passar diretamente para
    `genai.Client.models.embed_content(...)`.
    """

    GEMINI_EMBEDDING_001 = "gemini-embedding-001"
    GEMINI_EMBEDDING_2 = "gemini-embedding-2"
    GEMINI_EMBEDDING_2_PREVIEW = "gemini-embedding-2-preview"

class GeminiLLMClient:
    """Encapsula a criação e o uso do client oficial do Google Gemini."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = genai.Client(api_key=self._settings.google_api_key)

    def create_embedding(self, text: str, model: EmbeddingModel | None = None) -> list[float]:
        """Cria um embedding para o texto fornecido.

        Args:
            text: Texto a ser transformado em embedding.
            model: Modelo de embedding a ser usado (opcional, padrão do settings).

        Returns:
            Embedding como lista de floats.
        """
        model_to_use = model or self._settings.gemini_embedding_model
        response = self._client.models.embed_content(
            model=model_to_use,
            contents=text,
            config={"output_dimensionality": self._settings.gemini_embedding_dim},
        )
        return response.embeddings[0].values

    def generate_response(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list["BaseTool"] | None = None,
    ) -> str:
        """Envia uma conversa ao modelo e retorna o texto da resposta final.

        Recebe as `tools` de verdade (não só as declarações) porque, quando o
        modelo pede uma function call, é preciso executá-la (`tool.run(...)`)
        e devolver o resultado para o modelo, que então gera a resposta final.
        """
        contents: list[types.Content] = [
            types.Content(role=m["role"], parts=[types.Part(text=m["content"])]) for m in messages
        ]
        tools_by_name = {tool.name: tool for tool in (tools or [])}
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self._settings.gemini_max_output_tokens,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**tool.to_gemini_function_declaration()) for tool in tools
                    ]
                )
            ]
            if tools
            else None,
        )

        for _ in range(_MAX_FUNCTION_CALL_ROUNDS):
            response = self._client.models.generate_content(
                model=self._settings.gemini_model,
                contents=contents,
                config=config,
            )
            candidate_content = response.candidates[0].content
            function_calls = [part.function_call for part in candidate_content.parts if part.function_call]

            if not function_calls:
                return response.text

            contents.append(candidate_content)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=call.name,
                                response={"result": self._run_tool(tools_by_name, call)},
                            )
                        )
                        for call in function_calls
                    ],
                )
            )

        raise RuntimeError(
            f"O modelo excedeu o limite de {_MAX_FUNCTION_CALL_ROUNDS} chamadas de "
            "ferramentas para uma única mensagem."
        )

    @staticmethod
    def _run_tool(tools_by_name: dict[str, "BaseTool"], call: types.FunctionCall) -> Any:
        tool = tools_by_name.get(call.name)
        if tool is None:
            return {"error": f"Ferramenta '{call.name}' não existe."}
        try:
            return tool.run(**(call.args or {}))
        except Exception as exc:  # ferramenta com input inválido, dado inexistente etc.
            logger.warning("Ferramenta '%s' falhou com args=%r: %s", call.name, call.args, exc)
            return {"error": str(exc)}
