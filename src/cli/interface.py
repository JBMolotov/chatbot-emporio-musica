"""Loop interativo do chat via terminal.

Interface via terminal. Uma futura interface web/API pode reutilizar `agent.core.EmporioMusicaAgent`.
"""

from __future__ import annotations

import uuid

from rich.console import Console

from agent.core import EmporioMusicaAgent
from utils.logger import get_logger

logger = get_logger(__name__)

_EXIT_COMMANDS = {"sair", "exit", "quit"}


def run_chat_loop(agent: EmporioMusicaAgent, console: Console | None = None) -> None:
    """Executa o loop de conversa no terminal até o usuário encerrar.
    """
    console = console or Console()
    session_id = str(uuid.uuid4())

    console.print(
        "[bold]Empório da Música[/bold] - agente de atendimento (CLI)\n"
        "Digite sua mensagem ou 'sair' para encerrar.",
    )

    while True:
        try:
            user_message = console.input("[bold cyan]Você:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nAté logo!")
            break

        if not user_message:
            continue
        if user_message.lower() in _EXIT_COMMANDS:
            console.print("Até logo!")
            break

        try:
            response = agent.handle_message(session_id=session_id, user_message=user_message)
        except NotImplementedError as exc:
            # Placeholder enquanto o pipeline do agente não é implementado.
            logger.debug("Pipeline do agente ainda não implementado: %s", exc)
            console.print(
                "[yellow]O agente ainda não está implementado nesta versão "
                "inicial do projeto.[/yellow]",
            )
            continue

        console.print(f"[bold green]Agente:[/bold green] {response}")
