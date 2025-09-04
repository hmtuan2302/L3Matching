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
from domain.date_generation import DateGenerationProcessor
from domain.date_generation import DateGenerationInput
from domain.date_generation import DateGenerationOutput
from typing import List
from shared.base import BaseModel
import polars as pl

class AddApplicationInput(DocumentApplicationInput):
    """Input for document application."""
    files: List[UploadFile]  # List of UploadFile objects


class AddApplicationOutput(DocumentApplicationOutput):
    """Output for document application."""
    model_config = {'arbitrary_types_allowed': True}
    result: pl.DataFrame

class AddApplication(DocumentApplication):

    @cached_property
    def preprocess(self) -> PreProcessor:
        return PreProcessor()
    
    @cached_property
    def dategen(self) -> DateGenerationProcessor:
        return DateGenerationProcessor()

    @profile
    def run(self, inputs: AddApplicationInput) -> AddApplicationOutput:
        """Run whole pipeline (phase 1 + 2)

        Args:
            inputs (DocumentApplicationInput): list of files as s3 bucket object path

        Returns:
            DocumentApplicationOutput: Results
        """

        history = PreProcessorInput(file=inputs.files[0], file_type='mdl_historical')
        testing = PreProcessorInput(file=inputs.files[1], file_type='mdl_input_testing')
        l3 = PreProcessorInput(file=inputs.files[2], file_type='l3')
        results = self.preprocess.process_multiple([history, testing, l3])
        print(results)
        
        dategen_result = self.dategen.process(results)
        print(dategen_result.output_df)
        
        return AddApplicationOutput(
            result=dategen_result.output_df
        )
