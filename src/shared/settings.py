from __future__ import annotations

from dotenv import find_dotenv
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from .setting_config import AzureEmbeddingSetting
# test in local
load_dotenv(find_dotenv('.env'), override=True)


class Settings(BaseSettings):
    embedding: AzureEmbeddingSetting

    class Config:
        env_nested_delimiter = '__'
