from __future__ import annotations


from typing import List

from fastembed import TextEmbedding
from shared.utils import get_logger

logger = get_logger(__name__)


class EmbeddingProcessor:
    """Processor for generating embeddings from text."""

    def __init__(self, model_name: str = 'BAAI/bge-small-en-v1.5'):
        self.model_name = model_name
        self._embedding_model = None

    @property
    def embedding_model(self) -> TextEmbedding:
        """Lazy initialization of embedding model."""
        if self._embedding_model is None:
            logger.info(f"Initializing embedding model: {self.model_name}")
            self._embedding_model = TextEmbedding(model_name=self.model_name)
            logger.info(f"Embedding model {self.model_name} is ready to use.")
        return self._embedding_model

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            # Filter out empty/null texts
            valid_texts = []
            valid_indices = []

            for i, text in enumerate(texts):
                if text and str(text).strip():
                    valid_texts.append(str(text).strip())
                    valid_indices.append(i)

            if not valid_texts:
                logger.warning('No valid texts found for embedding generation')
                return [[0.0] * 384] * len(texts)  # Return zero vectors

            logger.info(
                f"Generating embeddings for {len(valid_texts)} texts...",
            )

            # Generate embeddings
            embeddings_generator = self.embedding_model.embed(valid_texts)
            embeddings_list = list(embeddings_generator)

            # Create full embedding list with zeros for invalid texts
            full_embeddings = []
            valid_idx = 0

            for i in range(len(texts)):
                if i in valid_indices:
                    full_embeddings.append(embeddings_list[valid_idx].tolist())
                    valid_idx += 1
                else:
                    # Use zero vector for empty/null texts
                    embedding_dim = len(
                        embeddings_list[0],
                    ) if embeddings_list else 384
                    full_embeddings.append([0.0] * embedding_dim)

            logger.info(
                f"Successfully generated {len(full_embeddings)} embeddings",
            )
            return full_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero vectors as fallback
            embedding_dim = 384  # Default for BAAI/bge-small-en-v1.5
            return [[0.0] * embedding_dim] * len(texts)

    def embedding_to_string(self, embedding: List[float]) -> str:
        """Convert embedding vector to string for storage."""
        return ','.join(map(str, embedding))

    def string_to_embedding(self, embedding_str: str) -> List[float]:
        """Convert string back to embedding vector."""
        try:
            return [float(x) for x in embedding_str.split(',')]
        except Exception:
            return [0.0] * 384  # Return zero vector if parsing fails