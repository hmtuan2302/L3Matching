from __future__ import annotations

from shared.utils import get_logger
from shared.utils import profile
from fastapi import UploadFile
import pandas as pd
from shared.base import BaseModel

logger = get_logger(__name__)


class PreProcessorInput(BaseModel):
    file: UploadFile


class PreProcessorOutput(BaseModel):
    file: str
    
class PreProcessor(BaseModel):
    @profile
    def process(self, input: PreProcessorInput) -> PreProcessorOutput:
        return PreProcessorOutput(file="test")