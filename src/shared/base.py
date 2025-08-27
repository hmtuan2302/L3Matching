from __future__ import annotations
from abc import ABC
from abc import abstractmethod
from typing import Any
from pydantic import BaseModel


class CustomBaseModel(BaseModel):
    class Config:
        """Configuration of the Pydantic Object"""

        # Allowing arbitrary types for class validation
        arbitrary_types_allowed = True


class BaseService(ABC, CustomBaseModel):
    @abstractmethod
    def process(self, inputs: Any) -> Any:
        raise NotImplementedError()


class AsyncBaseService(ABC, CustomBaseModel):
    @abstractmethod
    async def process(self, inputs: Any) -> Any:
        raise NotImplementedError()
