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
    model_config = {'arbitrary_types_allowed': True}
    file: UploadFile
    file_type: Literal[
        'mdl_input_testing',
        'mdl_historical', 'l3',
    ] = 'mdl_input_testing'

class PreProcessorOutput(BaseModel):
    model_config = {'arbitrary_types_allowed': True}
    processed_data: dict[str, pl.DataFrame] # Keys: 'mdl_input_testing', 'mdl_historical', 'l3'

class PreProcessor:

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
        output_dir: str = "../processed_outputs"
    ) -> PreProcessorOutput:
        """
        Process multiple files and return a dict of processed DataFrames.
        Also saves each processed DataFrame to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_dfs = {}
        ntp_date = None

        # Process the L3 file first
        for input in inputs:
            if input.file_type == 'l3':
                try:
                    logger.info("Processing L3 file to extract NTP date...")
                    l3_processor = L3Processor()
                    l3_clean_df = l3_processor.load_data(input.file.file)
                    ntp_date = l3_processor.find_ntp_date(l3_clean_df)
                    processed_dfs['l3'] = l3_clean_df

                    # Save L3
                    out_path = Path(output_dir) / "l3_processed.xlsx"
                    l3_clean_df.to_pandas().to_excel(out_path, index=False)
                    logger.info(f"L3 processed file saved to {out_path}")

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

                    mdl_processor = MDLPreprocessor(
                        list_excel_file_path=[input.file.file],
                        list_NTP=[ntp_date]
                    )
                    mdl_processor.load_data()
                    df = mdl_processor.preprocess()
                    df = df.with_columns([
                        pl.lit(ntp_date).alias("start_date")
                    ])
                    processed_dfs[input.file_type] = df

                    # Save MDL
                    out_path = Path(output_dir) / f"{input.file_type}_processed.xlsx"
                    df.to_pandas().to_excel(out_path, index=False)
                    logger.info(f"{input.file_type} processed file saved to {out_path}")

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
    

## TODO: remove later - example usage
from pathlib import Path
from fastapi import UploadFile

# Helper: turn local file path → UploadFile
def to_uploadfile(path: str) -> UploadFile:
    return UploadFile(filename=Path(path).name, file=open(path, "rb"))

# Example input files
inputs = [
    PreProcessorInput(file=to_uploadfile("../data/Grati_L3.xlsx"), file_type="l3"),
    PreProcessorInput(file=to_uploadfile("../data/Grati_MDL_test.xlsx"), file_type="mdl_input_testing"),
    PreProcessorInput(file=to_uploadfile("../data/Grati_MDL_train.xlsx"), file_type="mdl_historical"),
]

# Run processor
processor = PreProcessor()
outputs = processor.process_multiple(inputs, output_dir="../processed_results")