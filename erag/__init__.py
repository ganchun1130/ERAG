"""ERAG: thesis-aligned one-model RAG teaching assistant."""

__version__ = "2.2.0"

from .engine import ERAGEngine
from .settings import Settings

__all__ = ["ERAGEngine", "Settings"]
