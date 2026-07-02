"""Teste de fumaça: garante que o pacote e a estrutura básica carregam.
"""

from src import __version__


def test_package_has_version() -> None:
    assert __version__
