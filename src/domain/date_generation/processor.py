from __future__ import annotations

from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional, Literal
from typing import Tuple
from domain.preprocess import PreProcessorOutput
from domain.date_generation.basic_approach import basic_date_gen
import numpy as np
import polars as pl
from domain.date_generation.metrics import iou_1_sample
from domain.date_generation.metrics import iou_mean
from domain.date_generation.metrics import mae
# from domain.preprocess.mdl_processor import EmbeddingProcessor
from shared.base import BaseModel
from shared.utils import get_logger
from shared.utils import profile

logger = get_logger(__name__)


class DateGenerationInput(BaseModel):
    """Input for date generation processing."""
    model_config = {'arbitrary_types_allowed': True}
    preprocessed_data: PreProcessorOutput
    basic_cal_method: Literal['weighted_avg', 'avg'] = 'weighted_avg'
    max_k: int = 15

class DateGenerationOutput(BaseModel):
    """Output from date generation processing."""
    model_config = {'arbitrary_types_allowed': True}
    output_df: pl.DataFrame
    iou: Optional[float] = None
    mae_first_date: Optional[float] = None
    mae_final_date: Optional[float] = None


class DateGenerationProcessor:
    """Processor for generating date predictions based on historical data."""

    def calculate_fa_fc(self, df: pl.DataFrame) -> pl.DataFrame:
        ## add predicted_FA and predicted_FC columns
        df = df.with_columns([
            (pl.col("start_date") + pl.duration(days=pl.col("predicted_NTP_to_FA"))).alias("predicted_FA")
        ])

        # predicted_FC = predicted_FA + predicted_FA_to_FC (as days)
        df = df.with_columns([
            (pl.col("predicted_FA") + pl.duration(days=pl.col("predicted_FA_to_FC"))).alias("predicted_FC")
        ])
        return df

    
    @profile
    def process(
        self,
        input_data: DateGenerationInput,
    ) -> DateGenerationOutput:
        """Batch date generation pipeline for all rows in input_df."""

        # Extract inputs
        input_df = input_data.preprocessed_data.processed_data["mdl_historical"]
        hist_df = input_data.preprocessed_data.processed_data["mdl_input_testing"]

        logger.info(
            f"Running date generation with method={input_data.basic_cal_method}, max_k={input_data.max_k}"
        )

        # Step 1: Run basic date generation
        pred_df = basic_date_gen(
            input_data=input_df,
            hist_data=hist_df,
            max_k=input_data.max_k,
            method=input_data.basic_cal_method,
        )
        pred_df = self.calculate_fa_fc(pred_df)
        # Step 2: Compute evaluation metrics if ground truth available
        iou_score = None
        mae_first = None
        mae_final = None

        if "NTP_to_FA" in pred_df.columns and "FA_to_FC" in pred_df.columns:
            try:
                iou_score = iou_mean(
                    list(
                        zip(
                            pred_df["predicted_FA"].to_list(),
                            pred_df["FA"].to_list(),
                        )
                    ),
                    list(
                        zip(
                            pred_df["predicted_FC"].to_list(),
                            pred_df["FC"].to_list(),
                        )
                    ),
                )

                mae_first = mae(
                    pred_df["predicted_FA"].to_list(),
                    pred_df["FA"].to_list(),
                )

                mae_final = mae(
                    pred_df["predicted_FC"].to_list(),
                    pred_df["FC"].to_list(),
                )

                logger.info(
                    f"Metrics computed: IoU={iou_score}, "
                    f"MAE_first={mae_first}, MAE_final={mae_final}"
                )

            except Exception as e:
                logger.warning(f"Could not compute metrics: {e}")

        return DateGenerationOutput(
            output_df=pred_df,
            iou=iou_score,
            mae_first_date=mae_first,
            mae_final_date=mae_final,
        )