from __future__ import annotations
import numpy as np
import polars as pl
from typing import Literal
import re
from shared.utils import get_logger

logger = get_logger(__name__)

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of A and each row of B (vectorized)."""
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A_norm @ B_norm.T


def find_elbow_vectorized(sims: np.ndarray, max_k: int = 15) -> int:
    """
    Fast elbow detection using 2nd derivative.
    Works on a 1D similarity vector.
    """
    sims = np.sort(sims)[::-1][:max_k]
    diffs = np.diff(sims)
    if len(diffs) < 2:
        return min(len(sims), 3)
    second_diff = np.diff(diffs)
    elbow_idx = np.argmin(second_diff) + 1
    return elbow_idx + 1


def predict_date_range(
    sim_matrix: np.ndarray,
    hist_data: np.ndarray,
    method: Literal["weighted_avg", "avg"] = "weighted_avg",
    max_k: int = 15,
) -> np.ndarray:
    """
    Vectorized prediction of date ranges.
    Returns a NumPy array of predictions for all rows.
    """
    n_queries = sim_matrix.shape[0]
    predictions = np.zeros(n_queries, dtype=int)

    for i in range(n_queries):
        sims = sim_matrix[i]

        # elbow-based k
        k = find_elbow_vectorized(sims, max_k=min(max_k, len(hist_data)))

        # get top-k indices (fast selection, no full sort)
        top_idx = np.argpartition(-sims, k)[:k]
        top_sims = sims[top_idx]
        top_dates = hist_data[top_idx]

        if method == "avg" or top_sims.sum() == 0:
            weighted_avg = top_dates.mean()
        else:
            weighted_avg = np.dot(top_sims, top_dates) / top_sims.sum()

        predictions[i] = int(round(weighted_avg))

    return predictions

def parse_embedding(x: str) -> list[float]:
    """
    Convert a string like "[0. 1.23 4.56]" into a list of floats.
    Handles extra commas/newlines/spaces.
    """
    if isinstance(x, (list, np.ndarray)):
        return [float(v) for v in x]
    if not isinstance(x, str):
        return []
    try:
        # Remove brackets and normalize whitespace/commas
        cleaned = re.sub(r"[\[\]\n\r]", " ", x)
        cleaned = re.sub(r"[,\s]+", " ", cleaned).strip()
        if not cleaned:
            return []
        return [float(v) for v in cleaned.split(" ")]
    except Exception as e:
        logger.error(f"Failed to parse embedding: {x[:80]}... ({e})")
        return []

def basic_date_gen(
    input_data: pl.DataFrame,
    hist_data: pl.DataFrame,
    max_k: int,
    method: Literal["weighted_avg", "avg"] = "weighted_avg",
) -> pl.DataFrame:
    """Main entry point: generates predicted date ranges."""
    input_data = input_data.with_columns(
        pl.col("concat_embed").map_elements(parse_embedding).alias("concat_embed")
    )
    hist_data = hist_data.with_columns(
        pl.col("concat_embed").map_elements(parse_embedding).alias("concat_embed")
    )   

    # Convert to NumPy arrays
    A = np.stack(input_data["concat_embed"].to_numpy())
    B = np.stack(hist_data["concat_embed"].to_numpy())

    sim_matrix = cosine_similarity_matrix(A, B)

    hist_NTP_FA = hist_data["NTP_to_FA"].to_numpy()
    hist_FA_FC = hist_data["FA_to_FC"].to_numpy()

    # Predictions
    NTP_to_FA_pred = predict_date_range(sim_matrix, hist_NTP_FA, method, max_k)
    FA_to_FC_pred = predict_date_range(sim_matrix, hist_FA_FC, method, max_k)

    return input_data.with_columns([
        pl.Series("predicted_NTP_to_FA", NTP_to_FA_pred),
        pl.Series("predicted_FA_to_FC", FA_to_FC_pred),
    ])
