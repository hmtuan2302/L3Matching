from __future__ import annotations
import polars as pl

import numpy as np
import polars as pl
from typing import List, Literal
from shared.utils import get_logger

logger = get_logger(__name__)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of A and each row of B."""
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T

def find_elbow(similarities: np.ndarray, max_k: int = 15) -> int:
    """
    Heuristic elbow detection:
    - Sort similarities
    - Look at 2nd derivative to find elbow point
    """
    sims = np.sort(similarities)[::-1][:max_k]  # top max_k
    diffs = np.diff(sims)
    if len(diffs) < 2:
        return min(len(sims), 3)  # fallback
    second_diff = np.diff(diffs)
    elbow_idx = np.argmin(second_diff) + 1
    return elbow_idx + 1  # k = index+1

def basic_date_gen(
    input_data: pl.DataFrame,
    hist_data: pl.DataFrame,
    max_k: int,
    method: Literal["weighted_avg", "avg"] = "weighted_avg"
) -> pl.DataFrame:
    A = np.array(input_data["concat_embed"].to_list())
    B = np.array(hist_data["concat_embed"].to_list())
    sim_matrix = cosine_similarity_matrix(A, B)

    NTP_to_FA_pred = predict_date_range("NTP_to_FA", sim_matrix, hist_data, method, max_k)
    NTP_to_FC_pred = predict_date_range("NTP_to_FC", sim_matrix, hist_data, method, max_k)

    input_data = input_data.with_columns([
        pl.Series("predicted_NTP_to_FA", NTP_to_FA_pred),
        pl.Series("predicted_NTP_to_FC", NTP_to_FC_pred),
    ])

    return input_data


def predict_date_range(
    column_name: str,
    sim_matrix: np.ndarray,
    hist_data: pl.DataFrame,
    method: Literal["weighted_avg", "avg"] = "weighted_avg",
    max_k: int = 15
) -> List[int]:
    predictions: List[int] = []
    hist_dates = np.array(hist_data[column_name].to_list())
    for _, sims in enumerate(sim_matrix):
        # Find elbow -> k
        k = find_elbow(sims, max_k=min(max_k, len(hist_dates)))
        top_idx = np.argsort(sims)[::-1][:k]
        
        top_sims = sims[top_idx]
        top_dates = hist_dates[top_idx]

        if method == "avg":
            weighted_avg = top_dates.mean()
        else:
            if top_sims.sum() > 0:
                weighted_avg = np.dot(top_sims, top_dates) / top_sims.sum()
            else:
                weighted_avg = top_dates.mean()

        predictions.append(int(round(weighted_avg)))
    return predictions