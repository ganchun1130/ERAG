"""ERAG: thesis-aligned one-model RAG teaching assistant."""

__version__ = "2.4.0"

from .engine import ERAGEngine
from .extensions import Extensions
from .schemas import Chunk
from .settings import Settings

__all__ = ["ERAGEngine", "Settings", "Extensions", "Chunk"]
