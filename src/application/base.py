from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Sequence

from fastapi import UploadFile
from shared.base import BaseModel
from shared.model import DocumentOperation


class DocumentApplicationInput(BaseModel):
    """Base Application Input, All ConcreteApplicationInput must inherited this if have extra fields"""

    file: UploadFile


class DocumentApplicationOutput(BaseModel):
    """Base Application Output, All ConcreteApplicationOutput must inherited this if have extra fields"""

    result: str


class DocumentApplication(BaseModel, ABC):
    """This is an application interface.\n
    Required run method.\n
    Application is in charged of execute a flow with provided
    error_handler (for tracking and handle errors), in a planned manner
    (sequentially, concurrently, through median like rabbitmq or kafka v.v.).
    """

    @abstractmethod
    def run(self, inputs: DocumentApplicationInput) -> DocumentApplicationOutput:
        """Run a document operation with provided error_handler.

        Args:
            inputs (ApplicationInput): application input

        Returns:
            ApplicationOutput: application output
        """
        raise NotImplementedError()
