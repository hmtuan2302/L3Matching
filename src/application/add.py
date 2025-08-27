from __future__ import annotations

import io
from functools import cached_property
from typing import Tuple
from typing import Type
from uuid import uuid4
from fastapi import UploadFile
from shared.utils import get_file_extension
from shared.logging import get_logger
from shared.utils import profile

from .base import DocumentApplication
from .base import DocumentApplicationInput
from .base import DocumentApplicationOutput
from domain.preprocess import PreProcessor
from domain.preprocess import PreProcessorInput
logger = get_logger(__name__)


class AddDocumentApplication(DocumentApplication):

    @cached_property
    def preprocess(self) -> PreProcessor:
        return PreProcessor()

    @profile
    def run(self, inputs: DocumentApplicationInput) -> DocumentApplicationOutput:
        """Run whole pipeline (phase 1 + 2)

        Args:
            inputs (DocumentApplicationInput): list of files as s3 bucket object path

        Returns:
            DocumentApplicationOutput: Results
        """
        results = self.preprocess.process(PreProcessorInput(file=inputs.file))

        return DocumentApplicationOutput(
            
            result="processed",
        )
