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
from rag.in_memory import InMemoryVectorStore
from rag.pdf_indexer import PdfPolicyIndexer
from rag.retriever import Retriever

app = typer.Typer(help="Agente de atendimento (CLI) da loja Empório da Música.")


@app.callback()
def _main() -> None:
    """Agente de atendimento (CLI) da loja Empório da Música."""




@app.command()
def chat() -> None:
    """Inicia uma sessão de atendimento interativa no terminal."""
    settings = get_settings()

    vector_store = InMemoryVectorStore(embedding_dim=settings.gemini_embedding_dim, settings=settings)
    if not vector_store.documents_by_index:
        indexer = PdfPolicyIndexer(vector_store)
        indexer.index_documents(settings.policies_pdf_path)
        vector_store.persist()
    retriever = Retriever(vector_store)

    dependencies = AgentDependencies(
        llm_client=GeminiLLMClient(settings=settings),
        retriever=retriever,
    )
    agent = EmporioMusicaAgent(dependencies=dependencies)

    run_chat_loop(agent)


if __name__ == "__main__":
    app()
