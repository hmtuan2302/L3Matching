from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List

from shared.base import BaseModel
from shared.model import DocumentOperation
from shared.model.file import UnitCategory
from shared.model.file import UploadFileType


class DocumentApplicationInput(BaseModel):
    """Input for document application."""
    operation: DocumentOperation
    files: List[Any]  # List of UploadFile objects
    document_type: UploadFileType
    unit: UnitCategory


class DocumentApplicationOutput(BaseModel):
    """Output from document application."""
    processed_files: List[str]
    total_rows_processed: int
    processing_status: str
    unit: str
    document_type: str
    data: Dict[str, Any] = {}


class SequentiallyAddDocumentApplication:
    """Application for sequentially processing document uploads."""
    
    def __init__(self, settings: Any):
        self.settings = settings
        
    def run(self, inputs: DocumentApplicationInput) -> DocumentApplicationOutput:
        """Process the document upload.
        
        Args:
            inputs: Input parameters for processing
            
        Returns:
            Processing results
        """
        # Simulate processing
        processed_files = []
        total_rows = 0
        
        for file in inputs.files:
            if hasattr(file, 'filename'):
                processed_files.append(file.filename)
                # Simulate row count
                total_rows += 50
        
        return DocumentApplicationOutput(
            processed_files=processed_files,
            total_rows_processed=total_rows,
            processing_status='completed',
            unit=inputs.unit.value if hasattr(inputs.unit, 'value') else str(inputs.unit),
            document_type=inputs.document_type.value if hasattr(inputs.document_type, 'value') else str(inputs.document_type),
        )
