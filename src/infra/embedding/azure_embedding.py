from __future__ import annotations

from openai import AsyncAzureOpenAI
from openai import AzureOpenAI
from shared.logging import get_logger
from shared.setting_config.embedding import AzureEmbeddingSetting
from shared.base import BaseService
from shared.base import BaseModel


logger = get_logger(__name__)


class AzureEmbeddingInput(BaseModel):
    context: str


class AzureEmbeddingOutput(BaseModel):
    vector: list


class AzureEmbeddingBaseService(BaseService):
    embedding_setting: AzureEmbeddingSetting

    @property
    def client(self) -> AzureOpenAI:
        return AzureOpenAI(
            api_key=self.embedding_setting.key,
            azure_endpoint=str(self.embedding_setting.endpoint),
            api_version=self.embedding_setting.api_version,
        )

    @property
    def async_client(self) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            api_key=self.embedding_setting.key,
            azure_endpoint=str(self.embedding_setting.endpoint),
            api_version=self.embedding_setting.api_version,
        )

    def process(self, inputs: AzureEmbeddingInput) -> AzureEmbeddingOutput | None:
        """
        Processes the input context to generate an embedding vector using Azure OpenAI.

        Parameters:
        inputs (AzureEmbeddingInput): An instance of AzureEmbeddingInput containing the context to be embedded.

        Returns:
        AzureEmbeddingOutput: An instance of AzureEmbeddingOutput containing the generated embedding vector.
        If an error occurs during the process, an empty list is returned as the embedding vector.
        """
        for i in range(
            self.embedding_setting.max_tries,
        ):
            try:
                logger.info(
                    f'Generating embedding vector in {i+1} time',
                    dimension=self.embedding_setting.dimensions,
                )
                response = self.client.embeddings.create(
                    input=inputs.context,
                    model=self.embedding_setting.deployment_name,
                    dimensions=self.embedding_setting.dimensions,
                )
                embed_vector = response.data[0].embedding
                return AzureEmbeddingOutput(vector=embed_vector)
            except Exception as e:
                if i < self.embedding_setting.max_tries:
                    logger.warning(
                        f'Retrying to get embedding vector: {e} in {i+1} time',
                        extra={
                            'inputs': inputs,
                        },
                    )
                    continue
                else:
                    logger.exception(
                        f'Some error when get embedding vector: {e}',
                        extra={
                            'inputs': inputs,
                        },
                    )
                    raise e
        return None

    async def aprocess(
        self,
        inputs: AzureEmbeddingInput,
    ) -> AzureEmbeddingOutput | None:
        """
        Processes async the input context to generate an embedding vector using Azure OpenAI.

        Parameters:
        inputs (AzureEmbeddingInput): An instance of AzureEmbeddingInput containing the context to be embedded.

        Returns:
        AzureEmbeddingOutput: An instance of AzureEmbeddingOutput containing the generated embedding vector.
        If an error occurs during the process, an empty list is returned as the embedding vector.
        """
        for i in range(
            self.embedding_setting.max_tries,
        ):
            try:
                logger.info(
                    f'Generating embedding vector in {i+1} time',
                    dimension=self.embedding_setting.dimensions,
                )
                response = await self.async_client.embeddings.create(
                    input=inputs.context,
                    model=self.embedding_setting.deployment_name,
                    dimensions=self.embedding_setting.dimensions,
                )
                embed_vector = response.data[0].embedding
                return AzureEmbeddingOutput(vector=embed_vector)
            except Exception as e:
                if i < self.embedding_setting.max_tries:
                    logger.warning(
                        f'Retrying to get embedding vector: {e} in {i+1} time',
                        extra={
                            'inputs': inputs,
                        },
                    )
                    continue
                else:
                    logger.exception(
                        f'Some error when get embedding vector: {e}',
                        extra={
                            'inputs': inputs,
                        },
                    )
                    raise e
        return None
