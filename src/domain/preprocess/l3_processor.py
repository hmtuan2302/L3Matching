import polars as pl
import datetime
from shared.utils import get_logger
logger = get_logger(__name__)


class L3Processor:
    def load_data(self, data_path: str) -> pl.DataFrame:
        l3_df = pl.read_excel(data_path)
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
        if not isinstance(ntp_date, datetime.datetime) and not isinstance(ntp_date, pl.datetime):
            logger.error("NTP date is not a valid datetime object.")
            raise ValueError("NTP date is not a valid datetime object.")
        
        return ntp_date
    

