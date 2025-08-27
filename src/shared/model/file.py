from __future__ import annotations

from enum import Enum


class UnitCategory(str, Enum):
    """Unit categories for file classification."""
    FINANCE = 'finance'
    HR = 'hr'
    OPERATIONS = 'operations'
    MARKETING = 'marketing'
    TECHNICAL = 'technical'
    GENERAL = 'general'


class UploadFileType(str, Enum):
    """Supported file types for upload."""
    EXCEL = 'excel'
    CSV = 'csv'
    PDF = 'pdf'
    WORD = 'word'
    TEXT = 'text'
