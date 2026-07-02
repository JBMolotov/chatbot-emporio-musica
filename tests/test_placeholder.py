"""Teste de fumaça: garante que o pacote e a estrutura básica carregam.
"""

from src import __version__
from src.agent.core import AgentDependencies, EmporioMusicaAgent
from src.agent.llm_client import GeminiLLMClient
from src.config import Settings


def test_package_has_version() -> None:
    assert __version__


def test_agent_handle_message_not_implemented_yet() -> None:
    settings = Settings(google_api_key="test-key")
    dependencies = AgentDependencies(llm_client=GeminiLLMClient(settings=settings))
    agent = EmporioMusicaAgent(dependencies=dependencies)

    try:
        agent.handle_message(session_id="s1", user_message="olá")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Pipeline do agente ainda deveria estar sem implementação.")
