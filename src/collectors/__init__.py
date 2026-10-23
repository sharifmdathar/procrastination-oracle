"""Collectors that fill the Procrastination Oracle event log."""

from .synthetic import generate_and_store, generate_tasks

__all__ = ["generate_tasks", "generate_and_store"]
