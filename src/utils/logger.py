"""Configuração de logging da aplicação."""

from __future__ import annotations

import logging

from config import get_settings

_CONFIGURED = False


def _configure_root_logger() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger configurado para o módulo `name`."""
    _configure_root_logger()
    return logging.getLogger(name)
