"""Interface base para ferramentas (tools) usadas pelo agente.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Interface para uma ferramenta invocável pelo agente via function calling."""

    #: Nome da ferramenta, conforme exposto ao modelo.
    name: str

    #: Descrição usada pelo modelo para decidir quando chamar a ferramenta.
    description: str

    #: Schema (estilo OpenAPI/JSON Schema) dos parâmetros de entrada da ferramenta.
    parameters_schema: dict[str, Any]

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Executa a ferramenta com os argumentos fornecidos pelo modelo."""
        raise NotImplementedError

    def to_gemini_function_declaration(self) -> dict[str, Any]:
        """Converte a ferramenta para o formato de `FunctionDeclaration` do Gemini."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }
