"""Full-grid metrics and event-equal aggregation used in the paper tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def exact_top_mask(scores: np.ndarray, valid: np.ndarray, fraction: float) -> np.ndarray:
    valid_ids = np.flatnonzero(valid)
    count = max(1, int(np.ceil(fraction * len(valid_ids))))
    selected = valid_ids[np.argpartition(scores[valid_ids], -count)[-count:]]
    mask = np.zeros(len(scores), dtype=bool)
    mask[selected] = True
    return mask


def evaluate_grid(density: np.ndarray, scores: np.ndarray,
                  pga_bins: np.ndarray | None = None) -> dict:
    density = np.asarray(density, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if density.shape != scores.shape:
        raise ValueError("Density and score arrays must have the same shape")
    valid = np.isfinite(density) & np.isfinite(scores)
    positive = valid & (density > 0)
    if not positive.any() or (~positive & valid).sum() == 0:
        raise ValueError("Both positive and negative valid grids are required")
    y = positive[valid].astype(int)
    p = scores[valid]
    threshold = float(np.quantile(density[positive], 0.90))
    high_density = positive & (density >= threshold)
    top10 = exact_top_mask(scores, valid, 0.10)
    result = {
        "N_Test": int(valid.sum()),
        "Positive_Test": int(positive.sum()),
        "AUC": float(roc_auc_score(y, p)),
        "AP": float(average_precision_score(y, p)),
        "High_Density_Threshold": threshold,
        "N_High_Density": int(high_density.sum()),
        "Predicted_Top10_N": int(top10.sum()),
        "High_Density_Hits_At10": int((top10 & high_density).sum()),
        "HDR10": float((top10 & high_density).sum() / high_density.sum()),
    }
    if pga_bins is not None:
        bins = np.asarray(pga_bins)[valid]
        q4 = bins == bins.max()
        result["PGA_Q4_N"] = int(q4.sum())
        result["PGA_Q4_Positive"] = int(y[q4].sum())
        result["PGA_Q4_AUC"] = (
            float(roc_auc_score(y[q4], p[q4]))
            if len(np.unique(y[q4])) == 2 else float("nan")
        )
    return result


def equal_event_mean(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"Model", "Setting", "Test_Event", "Repeat", "AUC", "HDR10"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    if metrics.duplicated(["Model", "Setting", "Test_Event", "Repeat"]).any():
        raise ValueError("Duplicate model/setting/event/repeat result")
    event = metrics.groupby(["Model", "Setting", "Test_Event"], as_index=False).agg(
        Repeats=("Repeat", "nunique"), AUC=("AUC", "mean"),
        HDR10=("HDR10", "mean"),
    )
    return event.groupby(["Model", "Setting"], as_index=False).agg(
        Events=("Test_Event", "nunique"), Min_Repeats=("Repeats", "min"),
        Mean_AUC=("AUC", "mean"), Mean_HDR10=("HDR10", "mean"),
    )
