from __future__ import annotations

from datetime import date
from typing import List
from typing import Tuple
from typing import Union

import numpy as np


def iou_1_sample(
    predicted_values: Tuple[Union[str, date], Union[str, date]],
    actual_values: Tuple[Union[str, date], Union[str, date]],
) -> float:
    """
    Calculate IoU for two date ranges.

    Args:
        predicted_values: tuple (predicted_start, predicted_end)
        actual_values: tuple (actual_start, actual_end)

    Returns:
        IoU score (float between 0 and 1)
    """
    pred_start, pred_end = predicted_values
    actual_start, actual_end = actual_values

    # Convert strings to date
    if isinstance(pred_start, str):
        pred_start = date.fromisoformat(pred_start)
        pred_end = date.fromisoformat(pred_end)
    if isinstance(actual_start, str):
        actual_start = date.fromisoformat(actual_start)
        actual_end = date.fromisoformat(actual_end)

    # Ensure proper order
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start
    if actual_end < actual_start:
        actual_start, actual_end = actual_end, actual_start

    # Intersection
    intersection_start = max(pred_start, actual_start)
    intersection_end = min(pred_end, actual_end)
    intersection = max(
        0, (intersection_end - intersection_start).days + 1,
    )  # inclusive

    # Union
    union_start = min(pred_start, actual_start)
    union_end = max(pred_end, actual_end)
    union = (union_end - union_start).days + 1

    return intersection / union if union > 0 else 0.0


def iou_mean(
    predicted_ranges: List[Tuple[Union[str, date], Union[str, date]]],
    actual_ranges: List[Tuple[Union[str, date], Union[str, date]]],
) -> float:
    """
    Calculate mean IoU for multiple date ranges.

    Args:
        predicted_ranges: list of (predicted_start, predicted_end)
        actual_ranges: list of (actual_start, actual_end)

    Returns:
        Mean IoU score (float)
    """
    if len(predicted_ranges) != len(actual_ranges):
        raise ValueError(
            'Predicted and actual ranges must have the same length.',
        )
    if not predicted_ranges:
        return 0.0

    scores = [
        iou_1_sample(pred, actual)
        for pred, actual in zip(predicted_ranges, actual_ranges)
    ]
    return float(np.mean(scores))


def mae(
    predicted_dates: List[Union[str, date]],
    actual_dates: List[Union[str, date]],
) -> float:
    """
    Calculate MAE (mean absolute error) in days for a list of single dates.

    Args:
        predicted_dates: list of predicted dates (str or date)
        actual_dates: list of actual dates (same length)

    Returns:
        MAE in days
    """
    if len(predicted_dates) != len(actual_dates):
        raise ValueError(
            'Predicted and actual lists must have the same length.',
        )
    if not predicted_dates:
        return 0.0

    errors = []
    for pred, actual in zip(predicted_dates, actual_dates):
        if isinstance(pred, str):
            pred = date.fromisoformat(pred)
        if isinstance(actual, str):
            actual = date.fromisoformat(actual)
        errors.append(abs((pred - actual).days))

    return float(np.mean(errors))
