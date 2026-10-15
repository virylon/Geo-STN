"""Strip local paths from original per-repeat logs and combine AUC/AP/HDR@10."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import equal_event_mean


MODEL_NAMES = {
    "GeoTab-STN v2": "Geo-STN",
    "Random Forest": "RF",
    "HistGradientBoosting": "HGB",
    "Regularized MLP": "MLP",
    "FTTransformer": "FTTrans",
    "PatchCNN": "PatchCNN",
    "Mean-neighbor GraphSAGE": "GraphSAGE",
}
KEYS = ["Model", "Setting", "Test_Event", "Repeat"]


def export_results(geo_path: Path, geo_hdr_path: Path,
                   baseline_path: Path, baseline_hdr_path: Path,
                   output_path: Path) -> pd.DataFrame:
    geo = pd.read_csv(geo_path)
    geo_hdr = pd.read_csv(geo_hdr_path)
    base = pd.read_csv(baseline_path)
    base_hdr = pd.read_csv(baseline_hdr_path)
    if not geo.Model.eq("GeoTab-STN v2").all():
        raise ValueError("Geo-STN log contains unexpected models")
    geo_hdr["Model"] = "GeoTab-STN v2"
    base = base.loc[base.Model.ne("GeoTab-STN v2")].copy()
    base_hdr = base_hdr.loc[base_hdr.Model.ne("GeoTab-STN v2")].copy()
    base_hdr["Setting"] = "Domain-z"
    combined = pd.concat([geo, base], ignore_index=True)
    high_risk = pd.concat([geo_hdr, base_hdr], ignore_index=True)
    if combined.duplicated(KEYS).any() or high_risk.duplicated(KEYS).any():
        raise ValueError("Duplicate model/setting/event/repeat results")
    keep = KEYS + ["N_Valid", "N_Positive", "N_High_Density",
                   "High_Density_Threshold", "Predicted_Top10_N",
                   "HighDensity_Recall_At10", "AUC_Check", "AP_Check"]
    rows = combined.merge(high_risk[keep], on=KEYS, validate="one_to_one")
    if len(rows) != 880:
        raise ValueError(f"Expected 400 Geo-STN and 480 baseline rows, got {len(rows)}")
    if not (rows.N_Test.eq(rows.N_Valid) & rows.Positive_Test.eq(rows.N_Positive)).all():
        raise ValueError("Full-grid population mismatch")
    for reported, check in (("AUC", "AUC_Check"), ("AP", "AP_Check")):
        available = rows[check].notna()
        if not np.allclose(rows.loc[available, reported], rows.loc[available, check],
                           atol=1e-5, rtol=0):
            raise ValueError(f"Checkpoint verification mismatch: {reported}")
    hits = rows.N_High_Density * rows.HighDensity_Recall_At10
    if not np.allclose(hits, np.round(hits), atol=1e-6, rtol=0):
        raise ValueError("Non-integral high-density hit count")
    rows["High_Density_Hits_At10"] = np.round(hits).astype(int)
    rows["HDR10"] = rows.HighDensity_Recall_At10
    rows["Model"] = rows.Model.map(MODEL_NAMES)
    if rows.Model.isna().any():
        raise ValueError("Unknown model name")
    columns = KEYS + ["Seed", "N_Train", "N_Validation", "N_Test",
                      "Positive_Test", "Source_Val_AUC", "Best_Epoch", "Rank_Pairs",
                      "AUC", "AP", "PGA_Q4_AUC", "High_Density_Threshold",
                      "N_High_Density", "Predicted_Top10_N",
                      "High_Density_Hits_At10", "HDR10"]
    public = rows[columns].sort_values(KEYS).reset_index(drop=True)
    counts = public.groupby(["Model", "Setting"]).size()
    if not counts.eq(80).all():
        raise ValueError("Every released setting must contain 8 x 10 paired runs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(output_path, index=False)
    equal_event_mean(public).to_csv(output_path.with_name("event_equal_means.csv"),
                                    index=False)
    return public


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geostn", type=Path, required=True)
    parser.add_argument("--geostn-hdr", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--baselines-hdr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = export_results(args.geostn, args.geostn_hdr,
                          args.baselines, args.baselines_hdr, args.out)
    print(f"Wrote {len(rows)} path-free repeat records to {args.out}")


if __name__ == "__main__":
    main()
