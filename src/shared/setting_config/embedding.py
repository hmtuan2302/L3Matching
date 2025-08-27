from __future__ import annotations

from pydantic import HttpUrl
from shared.base import BaseModel


class AzureEmbeddingSetting(BaseModel):
    endpoint: HttpUrl
    key: str
    deployment_name: str
    api_version: str
    dimensions: int
    max_tries: int
