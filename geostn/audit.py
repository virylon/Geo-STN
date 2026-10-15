"""Check published aggregation or recompute metrics from private grid scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import equal_event_mean, evaluate_grid


def audit_recorded(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    summary = equal_event_mean(rows)
    if not summary.Events.eq(8).all() or not summary.Min_Repeats.eq(10).all():
        raise ValueError("Published rows require eight events and ten repeats per setting")
    if not np.allclose(rows.HDR10, rows.High_Density_Hits_At10 / rows.N_High_Density):
        raise ValueError("HDR@10 numerator/denominator mismatch")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return summary


def audit_scores(grid_path: Path, score_path: Path, expected_path: Path | None,
                 model: str | None, setting: str | None,
                 event: str | None, repeat: int | None) -> dict:
    with np.load(grid_path) as grids, np.load(score_path) as predictions:
        density = grids["density"]
        scores = predictions["scores"]
        result = evaluate_grid(density, scores)
    if expected_path is not None:
        if None in (model, setting, event, repeat):
            raise ValueError("Reference comparison requires model, setting, event and repeat")
        reference = pd.read_csv(expected_path)
        match = reference.loc[
            reference.Model.eq(model) & reference.Setting.eq(setting)
            & reference.Test_Event.eq(event) & reference.Repeat.eq(repeat)
        ]
        if len(match) != 1:
            raise ValueError("Expected exactly one matching reference row")
        row = match.iloc[0]
        for field in ("N_Test", "Positive_Test", "N_High_Density",
                      "Predicted_Top10_N", "High_Density_Hits_At10"):
            if int(result[field]) != int(row[field]):
                raise AssertionError(f"{field}: {result[field]} != {row[field]}")
        for field in ("AUC", "AP", "HDR10"):
            if not np.isclose(result[field], row[field], atol=1e-5, rtol=0):
                raise AssertionError(f"{field}: {result[field]} != {row[field]}")
        print("Reference row reproduced from full-grid scores.")
    for key, value in result.items():
        print(f"{key}: {value}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("recorded")
    summary.add_argument("--metrics", type=Path, required=True)
    scores = sub.add_parser("scores")
    scores.add_argument("--grid", type=Path, required=True,
                        help="NPZ with a density array in original full-grid row order")
    scores.add_argument("--scores", type=Path, required=True,
                        help="NPZ with scores in the same row order")
    scores.add_argument("--expected", type=Path)
    scores.add_argument("--model")
    scores.add_argument("--setting")
    scores.add_argument("--event")
    scores.add_argument("--repeat", type=int)
    args = parser.parse_args()
    if args.command == "recorded":
        audit_recorded(args.metrics)
    else:
        audit_scores(args.grid, args.scores, args.expected,
                     args.model, args.setting, args.event, args.repeat)


if __name__ == "__main__":
    main()
