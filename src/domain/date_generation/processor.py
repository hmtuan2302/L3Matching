from __future__ import annotations

from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import polars as pl
from domain.date_generation.match_title import exact_match
from domain.date_generation.match_title import find_similar_embeddings
from domain.date_generation.match_title import fuzzy_search
from domain.date_generation.metrics import iou_1_sample
from domain.date_generation.metrics import iou_mean
from domain.date_generation.metrics import mae
from domain.preprocess.mdl_processor import EmbeddingProcessor
from shared.base import BaseModel
from shared.utils import get_logger
from shared.utils import profile

logger = get_logger(__name__)


class DateGenerationInput(BaseModel):
    """Input for date generation processing."""
    model_config = {'arbitrary_types_allowed': True}
    pass

class DateGenerationOutput(BaseModel):
    """Output from date generation processing."""
    model_config = {'arbitrary_types_allowed': True}
    output_df: pl.DataFrame
    iou: Optional[float] = None
    mae_first_date: Optional[float] = None
    mae_final_date: Optional[float] = None


class DateGenerationProcessor:
    """Processor for generating date predictions based on historical data."""

    @profile
    def process(
        self,
        data: Dict[str, pl.DataFrame],
    ) -> DateGenerationOutput:
        """Batch date generation pipeline for all rows in input_df."""
        pass