from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Literal

import polars as pl
from domain.preprocess.embedding import EmbeddingProcessor
from domain.preprocess.file_validation import MDLFileValidator
from domain.preprocess.title_processor import TitleProcessor
from fastapi import UploadFile
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


class PreProcessor:

    def __init__(self):
        self.validator = MDLFileValidator()
        self.title_processor = TitleProcessor()
        self.embedding_processor = EmbeddingProcessor()

    def _save_upload_file(self, upload_file: UploadFile) -> str:
        """Save uploaded file to temporary location."""
        try:
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
            logger.error(f'Error saving upload file: {str(e)}')
            raise

    def _read_excel_file(self, file_path: str) -> pl.DataFrame:
        """Read Excel file using Polars."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f'File not found: {file_path}')

            if path.suffix.lower() not in ['.xlsx', '.xls']:
                raise ValueError(f'Invalid file format: {path.suffix}')

            df = pl.read_excel(file_path)
            logger.info(
                f'Successfully read Excel file: {df.height} rows, '
                f'{df.width} columns',
            )

            return df

        except Exception as e:
            logger.error(f'Error reading Excel file: {str(e)}')
            raise

    def _process_title_column(self, df: pl.DataFrame) -> pl.DataFrame:
        """Process title column with cleaning techniques."""
        if 'Title' not in df.columns:
            logger.warning('Title column not found, skipping title processing')
            return df

        try:
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
            logger.error(f'Error processing title column: {str(e)}')
            raise

    def _add_title_embeddings(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add embedding column for titles."""
        if 'Title' not in df.columns:
            logger.warning(
                'Title column not found, skipping embedding generation',
            )
            return df

        try:
            titles = df.select(pl.col('Title')).to_series().to_list()

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

            embeddings = self.embedding_processor.generate_embeddings(
                texts_to_embed,
            )

            embedding_strings = [
                self.embedding_processor.embedding_to_string(emb)
                for emb in embeddings
            ]

            df_with_embeddings = df.with_columns([
                pl.Series('Title_Embedding', embedding_strings),
            ])

            logger.info(
                f'Successfully generated embeddings for '
                f'{len(embeddings)} titles',
            )
            return df_with_embeddings

        except Exception as e:
            logger.error(f'Error generating title embeddings: {str(e)}')
            return df

    def _generate_file_hash(self, file_path: str) -> str:
        """Generate hash for the file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.md5(content).hexdigest()[:16]
        except Exception as e:
            logger.warning(f'Could not generate file hash: {str(e)}')
            return 'unknown'

    def _add_range_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add Range columns for First and Final dates, handling pre-converted
            datetime columns."""
        try:
            logger.info('Starting range column calculations...')

            date_cols = [
                'Plan - First Date', 'Actual - First Date',
                'Plan - Final Date', 'Actual - Final Date',
            ]

            column_info = {}
            for col in date_cols:
                dtype = df.select(pl.col(col)).dtypes[0]
                column_info[col] = dtype
                logger.info(f"Column '{col}' has dtype: {dtype}")

            datetime_conversion_map = {}

            for col in date_cols:
                if column_info[col] in [pl.Date, pl.Datetime]:
                    if column_info[col] == pl.Date:
                        datetime_conversion_map[col] = pl.col(
                            col,
                        ).cast(pl.Datetime)
                    else:
                        datetime_conversion_map[col] = pl.col(col)
                else:
                    datetime_conversion_map[col] = pl.col(
                        col,
                    ).str.to_datetime(strict=False)

            df_with_dates = df.with_columns([
                datetime_conversion_map['Plan - First Date'].alias(
                    'Plan_First_Date_dt',
                ),
                datetime_conversion_map['Actual - First Date'].alias(
                    'Actual_First_Date_dt',
                ),
                datetime_conversion_map['Plan - Final Date'].alias(
                    'Plan_Final_Date_dt',
                ),
                datetime_conversion_map['Actual - Final Date'].alias(
                    'Actual_Final_Date_dt',
                ),
            ])

            temp_datetime_cols = [
                'Plan_First_Date_dt', 'Actual_First_Date_dt',
                'Plan_Final_Date_dt', 'Actual_Final_Date_dt',
            ]

            for temp_col in temp_datetime_cols:
                null_count = df_with_dates.select(
                    pl.col(temp_col).is_null().sum(),
                ).item()
                if null_count > 0:
                    logger.warning(
                        f'Column {temp_col} has {null_count} null'
                        f' values after conversion',
                    )

            df_with_ranges = df_with_dates.with_columns([
                (pl.col('Actual_First_Date_dt') - pl.col('Plan_First_Date_dt'))
                .dt.total_days()
                .alias('Range - First Date'),

                (pl.col('Actual_Final_Date_dt') - pl.col('Plan_Final_Date_dt'))
                .dt.total_days()
                .alias('Range - Final Date'),
            ])

            final_df = df_with_ranges.drop([
                'Plan_First_Date_dt', 'Actual_First_Date_dt',
                'Plan_Final_Date_dt', 'Actual_Final_Date_dt',
            ])

            logger.info(
                'Successfully added Range columns',
            )
            return final_df

        except Exception as e:
            logger.error(f'Error adding range columns: {str(e)}')
            import traceback
            traceback.print_exc()
            return df

    @profile
    def process_multiple(
        self,
        inputs: list[PreProcessorInput],
    ) -> list[pl.DataFrame]:
        """
        Process multiple files and return a list of processed DataFrames.
        Each output is a processed DataFrame (or None if failed).
        """
        processed_dfs = []
        for input in inputs:
            temp_file_path = None
            try:
                temp_file_path = self._save_upload_file(input.file)
                df = self._read_excel_file(temp_file_path)

                if input.file_type in ['mdl_input_testing', 'mdl_historical']:
                    self.validator.validate_columns(df, input.file_type)
                    df = self.validator.validate_date_columns(
                        df, input.file_type,
                    )
                    df = self._add_range_columns(df)

                if 'Title' in df.columns:
                    df = self._process_title_column(df)
                    if input.generate_embeddings:
                        df = self._add_title_embeddings(df)

                processed_dfs.append(df)

            except Exception as e:
                logger.error(
                    'Error processing file %s: %s',
                    getattr(input.file, 'filename', 'unknown'),
                    str(e),
                )
                processed_dfs.append(None)

            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception as e:
                        logger.warning(
                            f'Could not clean up temp file: {str(e)}',
                        )

        return processed_dfs
