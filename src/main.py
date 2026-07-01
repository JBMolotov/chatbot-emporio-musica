"""Ponto de entrada da aplicação (CLI).

Uso:
    emporio-chatbot chat        # após `pip install -e .`
    python src/main.py chat     # execução direta em desenvolvimento
"""

from __future__ import annotations

import typer

from agent.core import AgentDependencies, EmporioMusicaAgent
from agent.llm_client import GeminiLLMClient
from cli.interface import run_chat_loop
from config import get_settings

app = typer.Typer(help="Agente de atendimento (CLI) da loja Empório da Música.")


@app.callback()
def _main() -> None:
    """Agente de atendimento (CLI) da loja Empório da Música."""


@app.command()
def chat() -> None:
    """Inicia uma sessão de atendimento interativa no terminal."""
    settings = get_settings()

    dependencies = AgentDependencies(
        llm_client=GeminiLLMClient(settings=settings),
    )
    agent = EmporioMusicaAgent(dependencies=dependencies)

    run_chat_loop(agent)


if __name__ == "__main__":
    app()
