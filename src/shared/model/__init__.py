from __future__ import annotations

from enum import Enum


class DocumentOperation(str, Enum):
    """Document operation types."""
    ADD = 'add'
    UPDATE = 'update'
    DELETE = 'delete'
    GET = 'get'
