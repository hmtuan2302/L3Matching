import polars as pl
import datetime
from typing import Union, BinaryIO
import io
from shared.utils import get_logger
logger = get_logger(__name__)


class L3Processor:
    def load_data(self, data_source: Union[str, bytes, BinaryIO]) -> pl.DataFrame:
        """
        Load Excel data from either a file path, binary data, or file-like object.
        
        Args:
            data_source: Can be a file path (str), binary data (bytes), or file-like object
        """
        if isinstance(data_source, str):
            # File path
            l3_df = pl.read_excel(data_source)
        elif isinstance(data_source, bytes):
            # Binary data
            l3_df = pl.read_excel(io.BytesIO(data_source))
        elif hasattr(data_source, 'read'):
            # File-like object (including UploadFile.file)
            try:
                data_source.seek(0)  # Reset file pointer to beginning
            except (AttributeError, io.UnsupportedOperation):
                pass  # Some file-like objects don't support seek
            content = data_source.read()
            l3_df = pl.read_excel(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
            
        clean_df = l3_df.filter(
            (pl.col("Activity Name").is_not_null()) & (pl.col("Activity Name") != "")
        )
        return clean_df
    
    def find_ntp_date(self, l3_df: pl.DataFrame):
        # Find the NTP date from the L3 DataFrame
        ntp_row = l3_df.filter(
            pl.col("Activity Name").str.to_lowercase() == "ntp"
        )
        if ntp_row.is_empty():
            logger.error("NTP row not found in L3 data.")
            raise ValueError("NTP row not found in L3 data.")
        
        ntp_date = ntp_row.select(pl.col("Start").cast(pl.Datetime)).item()
        if not isinstance(ntp_date, datetime.datetime):
            logger.error("NTP date is not a valid datetime object.")
            raise ValueError("NTP date is not a valid datetime object.")
        
        return ntp_date
    

