"""Backward-compatible facade for the refactored structure service."""
from __future__ import annotations

from .structure import (
    StructureCache,
    StructureContext,
    StructureService,
    StructureServiceError,
    StructureSource,
)

__all__ = [
    "StructureService",
    "StructureServiceError",
    "StructureCache",
    "StructureContext",
    "StructureSource",
]
