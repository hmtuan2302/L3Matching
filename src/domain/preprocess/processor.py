from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Literal

import polars as pl
from domain.preprocess.mdl_processor import MDLPreprocessor
from domain.preprocess.l3_processor import L3Processor
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


class PreProcessor:

    def __init__(self):
        self.mdl_preprocessor = MDLPreprocessor()

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

    
    @profile
    def process_multiple(
        self,
        inputs: list[PreProcessorInput],
    ) -> dict:
        """
        Process multiple files and return a dict of processed DataFrames.
        The L3 file is processed first to extract the NTP date, which is then
        applied to the processing of MDL files.
        """
        processed_dfs = {}
        ntp_date = None

        # Process the L3 file first
        for input in inputs:
            if input.file_type == 'l3':
                try:
                    logger.info("Processing L3 file to extract NTP date...")
                    _ = self._read_excel_file(input.file.file)
                    l3_processor = L3Processor()
                    l3_clean_df = l3_processor.load_data(input.file.file)
                    ntp_date = l3_processor.find_ntp_date(l3_clean_df)
                    processed_dfs['l3'] = l3_clean_df
                    logger.info(f"NTP date extracted: {ntp_date}")
                except Exception as e:
                    logger.error(
                        f"Error processing L3 file {getattr(input.file, 'filename', 'unknown')}: {str(e)}"
                    )
                    processed_dfs['l3'] = None
                break

        if ntp_date is None:
            raise ValueError("NTP date could not be extracted from the L3 file.")

        # Process the MDL files
        for input in inputs:
            if input.file_type in ['mdl_input_testing', 'mdl_historical']:
                temp_file_path = None
                try:
                    logger.info(f"Processing {input.file_type} file...")
                    temp_file_path = self._save_upload_file(input.file)
                    df = self._read_excel_file(temp_file_path)

                    ## TODO: fill in this part
                    mdl_processor = MDLPreprocessor()

                    processed_dfs[input.file_type] = df

                except Exception as e:
                    logger.error(
                        f"Error processing file {getattr(input.file, 'filename', 'unknown')}: {str(e)}"
                    )
                    processed_dfs[input.file_type] = None

                finally:
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except Exception as e:
                            logger.warning(f"Could not clean up temp file: {str(e)}")

        return processed_dfs

