from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import List
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from fastapi import UploadFile
from fastembed import TextEmbedding
from shared.base import BaseModel
from shared.utils import get_logger
from shared.utils import profile

logger = get_logger(__name__)


class PreProcessorInput(BaseModel):
    file: UploadFile
    file_type: Literal[
        'mdl_input_testing',
        'mdl_historical', 'l3',
    ] = 'mdl_input_testing'
    generate_embeddings: bool = True  # New parameter to embedding generation


class PreProcessorOutput(BaseModel):
    file_path: str
    file_type: str
    validation_status: str
    processed_rows: int
    columns_found: List[str]
    title_processing_applied: bool
    embeddings_generated: bool  # New field to track embedding generation
    embedding_model_used: str  # Track which model was used
    file_hash: str


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


class MDLFileValidator:
    """Validator for MDL file types."""

    # Required columns for different file types
    MDL_INPUT_TESTING_COLUMNS = [
        'Document No',
        'Title',
        'Plan - First Date',
        'Plan - Final Date',
    ]

    MDL_HISTORICAL_COLUMNS = [
        'Document No',
        'Title',
        'Plan - First Date',
        'Plan - Final Date',
        'Actual - First Date',
        'Actual - Final Date',
    ]

    DATE_COLUMNS_INPUT = [
        'Plan - First Date',
        'Plan - Final Date',
    ]

    DATE_COLUMNS_HISTORICAL = [
        'Plan - First Date',
        'Plan - Final Date',
        'Actual - First Date',
        'Actual - Final Date',
    ]

    def validate_columns(self, df: pl.DataFrame, file_type: str) -> None:
        """Validate required columns are present."""
        if file_type == 'mdl_input_testing':
            required_columns = self.MDL_INPUT_TESTING_COLUMNS
        elif file_type == 'mdl_historical':
            required_columns = self.MDL_HISTORICAL_COLUMNS
        else:
            # L3 files - skip validation for now
            # TODO: validation for L3
            return

        missing_columns = [
            col for col in required_columns if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for {file_type}: {missing_columns}",
            )

        logger.info(f"All required columns present for {file_type}")

    def validate_date_columns(
            self,
            df: pl.DataFrame,
            file_type: str,
    ) -> pl.DataFrame:
        """Validate and convert date columns."""
        if file_type == 'l3':
            return df  # Skip date validation for L3 files

        date_columns = (
            self.DATE_COLUMNS_INPUT if file_type == 'mdl_input_testing'
            else self.DATE_COLUMNS_HISTORICAL
        )

        validated_df = df.clone()

        for date_col in date_columns:
            if date_col not in df.columns:
                continue

            # Check for null values
            null_count = validated_df[date_col].null_count()
            if null_count > 0:
                raise ValueError(
                    f"Found {null_count} null values in date column "
                    f"'{date_col}'. All date columns "
                    f"must have valid values.",
                )

            # If already datetime, skip conversion
            if pl.datatypes.is_datetime(validated_df.schema[date_col]):
                logger.info(f"Column '{date_col}' is already datetime type")
                continue

            # Convert to datetime
            try:
                validated_df = validated_df.with_columns([
                    pl.col(date_col).str.to_datetime(
                        format='%Y-%m-%d', strict=False,
                    ).alias(date_col),
                ])
                logger.info(f"Successfully converted '{date_col}' to datetime")
            except Exception:
                try:
                    validated_df = validated_df.with_columns([
                        pl.col(date_col).str.to_datetime().alias(date_col),
                    ])
                    logger.info(
                        f"Successfully converted '{date_col}' "
                        f"to datetime with auto-parsing",
                    )
                except Exception as e:
                    raise ValueError(
                        f"Cannot convert '{date_col}' to datetime: {str(e)}",
                    )

        return validated_df


class TitleProcessor:
    """Processor for cleaning and normalizing title text."""

    @staticmethod
    def trim_and_case(text: str) -> str:
        """Trim whitespace and normalize case."""
        return text.strip().title()

    @staticmethod
    def whitespace_collapse(text: str) -> str:
        """Collapse multiple whitespaces into single space."""
        return re.sub(r'\s+', ' ', text)

    @staticmethod
    def bracket_stripping(text: str) -> str:
        """Remove content within brackets and the brackets themselves."""
        # Remove content within various types of brackets
        text = re.sub(r'\[.*?\]', '', text)  # Square brackets
        text = re.sub(r'\(.*?\)', '', text)  # Round brackets
        text = re.sub(r'\{.*?\}', '', text)  # Curly brackets
        text = re.sub(r'<.*?>', '', text)    # Angle brackets
        return text

    @staticmethod
    def delimiter_normalise(text: str) -> str:
        """Normalize delimiters to standard format."""
        # Replace various delimiters with standard ones
        text = re.sub(r'[–—―]', '-', text)    # Em/en dashes to hyphen
        text = re.sub(r'[''‚‛]', "'", text)   # Various apostrophes
        text = re.sub(r'[""„‟]', '"', text)   # Various quotes
        text = re.sub(r'[…]', '...', text)    # Ellipsis
        return text

    @staticmethod
    def prefix_suffix_rules(text: str) -> str:
        """Apply prefix/suffix normalization rules."""
        # Remove common prefixes
        prefixes = [r'^(the|a|an)\s+', r'^(project|task|work)\s+']
        for prefix in prefixes:
            text = re.sub(prefix, '', text, flags=re.IGNORECASE)

        # Remove common suffixes
        suffixes = [r'\s+(inc|ltd|corp|llc)\.?$', r'\s+(project|task|work)$']
        for suffix in suffixes:
            text = re.sub(suffix, '', text, flags=re.IGNORECASE)

        return text

    @classmethod
    def process_title(cls, title: str) -> str:
        """Apply all title processing techniques."""
        if pd.isna(title) or title == '':
            return ''

        # Convert to string if not already
        title = str(title)

        # Apply processing steps in order
        title = cls.trim_and_case(title)
        title = cls.whitespace_collapse(title)
        title = cls.bracket_stripping(title)
        title = cls.delimiter_normalise(title)
        title = cls.prefix_suffix_rules(title)
        title = cls.whitespace_collapse(title)  # Final cleanup
        title = title.strip()

        return title

    @classmethod
    def generate_title_hash(cls, title: str) -> str:
        """Generate hash for processed title."""
        processed_title = cls.process_title(title)
        return hashlib.md5(processed_title.lower().encode()).hexdigest()[:8]


class PreProcessor(BaseModel):

    def __init__(self):
        self.validator = MDLFileValidator()
        self.title_processor = TitleProcessor()
        self.embedding_processor = EmbeddingProcessor()

    def _save_upload_file(self, upload_file: UploadFile) -> str:
        """Save uploaded file to temporary location."""
        try:
            # Create temporary file with original extension
            suffix = Path(
                upload_file.filename,
            ).suffix if upload_file.filename else '.xlsx'
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp_file:
                content = upload_file.file.read()
                tmp_file.write(content)
                temp_path = tmp_file.name

            upload_file.file.seek(0)  # Reset file pointer
            return temp_path

        except Exception as e:
            logger.error(f"Error saving upload file: {str(e)}")
            raise

    def _read_excel_file(self, file_path: str) -> pl.DataFrame:
        """Read Excel file using Polars."""
        try:
            # Check file exists and has valid extension
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if path.suffix.lower() not in ['.xlsx', '.xls']:
                raise ValueError(f"Invalid file format: {path.suffix}")

            # Read with Polars
            df = pl.read_excel(file_path)
            logger.info(
                f"Successfully read Excel file: {df.height} rows, "
                f"{df.width} columns",
            )

            return df

        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise

    def _process_title_column(self, df: pl.DataFrame) -> pl.DataFrame:
        """Process title column with cleaning techniques."""
        if 'Title' not in df.columns:
            logger.warning('Title column not found, skipping title processing')
            return df

        try:
            # Process titles and create hash column
            processed_df = df.with_columns([
                pl.col('Title').map_elements(
                    self.title_processor.process_title,
                    return_dtype=pl.Utf8,
                ).alias('Title_Processed'),
                pl.col('Title').map_elements(
                    self.title_processor.generate_title_hash,
                    return_dtype=pl.Utf8,
                ).alias('Title_Hash'),
            ])

            logger.info('Successfully processed title column')
            return processed_df

        except Exception as e:
            logger.error(f"Error processing title column: {str(e)}")
            raise

    def _add_title_embeddings(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add embedding column for titles."""
        if 'Title' not in df.columns:
            logger.warning(
                'Title column not found, skipping embedding generation',
            )
            return df

        try:
            # Extract titles for embedding
            titles = df.select(pl.col('Title')).to_series().to_list()

            # Use processed titles if available, otherwise use original
            if 'Title_Processed' in df.columns:
                processed_titles = df.select(
                    pl.col('Title_Processed'),
                ).to_series().to_list()
                texts_to_embed = [
                    title if title else orig for title, orig in zip(
                        processed_titles, titles,
                    )
                ]
            else:
                texts_to_embed = titles

            # Generate embeddings
            embeddings = self.embedding_processor.generate_embeddings(
                texts_to_embed,
            )

            # Convert embeddings to strings for storage
            embedding_strings = [
                self.embedding_processor.embedding_to_string(emb)
                for emb in embeddings
            ]

            # Add embedding column
            df_with_embeddings = df.with_columns([
                pl.Series('Title_Embedding', embedding_strings),
            ])

            logger.info(
                f"Successfully generated embeddings for "
                f"{len(embeddings)} titles",
            )
            return df_with_embeddings

        except Exception as e:
            logger.error(f"Error generating title embeddings: {str(e)}")
            # Return original dataframe if embedding fails
            return df

    def _generate_file_hash(self, file_path: str) -> str:
        """Generate hash for the file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()[:16]
        except Exception as e:
            logger.warning(f"Could not generate file hash: {str(e)}")
            return 'unknown'

    def _save_processed_file(
        self,
        df: pl.DataFrame,
        original_path: str,
        file_type: str,
    ) -> str:
        """Save processed DataFrame to new Excel file."""
        try:
            # Create output filename
            original_name = Path(original_path).stem
            output_dir = Path('processed_files')
            output_dir.mkdir(exist_ok=True)

            output_path = output_dir / \
                f"{original_name}_{file_type}_processed.xlsx"

            # Save as Excel
            df.write_excel(str(output_path))
            logger.info(f"Saved processed file to: {output_path}")

            return str(output_path)

        except Exception as e:
            logger.error(f"Error saving processed file: {str(e)}")
            raise

    @profile
    def process(self, input: PreProcessorInput) -> PreProcessorOutput:
        """Main processing pipeline."""
        temp_file_path = None
        embeddings_generated = False
        embedding_model_used = ''

        try:
            # Save uploaded file temporarily
            temp_file_path = self._save_upload_file(input.file)

            # Read Excel file
            df = self._read_excel_file(temp_file_path)

            # Validate columns based on file type
            if input.file_type in ['mdl_input_testing', 'mdl_historical']:
                self.validator.validate_columns(df, input.file_type)

                # Validate and convert date columns
                df = self.validator.validate_date_columns(df, input.file_type)

            # Process title column (for all file types that have Title column)
            title_processing_applied = False
            if 'Title' in df.columns:
                df = self._process_title_column(df)
                title_processing_applied = True

                # Generate embeddings if requested
                if input.generate_embeddings:
                    df = self._add_title_embeddings(df)
                    embeddings_generated = True
                    embedding_model_used = self.embedding_processor.model_name

            # Generate file hash
            file_hash = self._generate_file_hash(temp_file_path)

            # Save processed file
            output_path = self._save_processed_file(
                df, temp_file_path, input.file_type,
            )

            return PreProcessorOutput(
                file_path=output_path,
                file_type=input.file_type,
                validation_status='success',
                processed_rows=df.height,
                columns_found=df.columns,
                title_processing_applied=title_processing_applied,
                embeddings_generated=embeddings_generated,
                embedding_model_used=embedding_model_used,
                file_hash=file_hash,
            )

        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return PreProcessorOutput(
                file_path='',
                file_type=input.file_type,
                validation_status=f"failed: {str(e)}",
                processed_rows=0,
                columns_found=[],
                title_processing_applied=False,
                embeddings_generated=False,
                embedding_model_used='',
                file_hash='',
            )

        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Could not clean up temp file: {str(e)}")


# Utility function to extract embeddings from processed file
def extract_embeddings_from_file(
    file_path: str,
) -> tuple[np.ndarray, List[str]]:
    """
    Extract embeddings and titles from a processed Excel file.

    Args:
        file_path: Path to the processed Excel file

    Returns:
        Tuple of (embeddings_array, titles_list)
    """
    try:
        df = pl.read_excel(file_path)

        if 'Title_Embedding' not in df.columns:
            raise ValueError('No embedding column found in file')

        # Extract embeddings and convert back to numpy array
        embedding_processor = EmbeddingProcessor()
        embedding_strings = df.select(
            pl.col('Title_Embedding'),
        ).to_series().to_list()
        embeddings = [
            embedding_processor.string_to_embedding(emb_str)
            for emb_str in embedding_strings
        ]

        # Extract titles
        title_column = (
            'Title_Processed'
            if 'Title_Processed' in df.columns
            else 'Title'
        )
        titles = df.select(pl.col(title_column)).to_series().to_list()

        return np.array(embeddings), titles

    except Exception as e:
        logger.error(f"Error extracting embeddings from file: {str(e)}")
        raise


# Example usage
if __name__ == '__main__':
    processor = PreProcessor()

    # Example processing with embeddings
    # result = processor.process(PreProcessorInput(
    #     file=upload_file,
    #     file_type="mdl_historical",
    #     generate_embeddings=True
    # ))
    # print(result)
