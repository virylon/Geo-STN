"""Small implementation checks; synthetic values are not research results."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from geostn.metrics import evaluate_grid
from geostn.model import GeoTabSTNV2
from geostn.protocol import DOMAIN_FEATURES, EVENTS, load_events, stable_seed
from geostn.train import pga_zone_rank_loss, run_one


class ReleaseTests(unittest.TestCase):
    def test_monotonic_pga_with_fixed_environment(self):
        torch.manual_seed(3)
        model = GeoTabSTNV2(list(DOMAIN_FEATURES), monotonic_pga=True).eval()
        center = torch.zeros(5, len(DOMAIN_FEATURES))
        center[:, -1] = torch.linspace(-2, 2, 5)
        neighbors = torch.zeros_like(center)
        with torch.no_grad():
            output = model(center, neighbors)
        self.assertTrue(torch.all(output[1:] >= output[:-1]).item())

    def test_full_grid_metrics_and_hit_count(self):
        density = np.array([0.0] * 18 + [0.1, 0.9])
        scores = np.arange(20, dtype=float)
        result = evaluate_grid(density, scores)
        self.assertEqual(result["N_Test"], 20)
        self.assertEqual(result["Positive_Test"], 2)
        self.assertEqual(result["Predicted_Top10_N"], 2)
        self.assertEqual(result["N_High_Density"], 1)
        self.assertEqual(result["High_Density_Hits_At10"], 1)
        self.assertEqual(result["HDR10"], 1.0)
        self.assertEqual(result["AUC"], 1.0)

    def test_rank_loss_preserves_group_boundaries(self):
        logit = torch.tensor([0.0, 0.0, 0.0, 0.0])
        target = torch.tensor([0.0, 1.0, 0.0, 1.0])
        event = torch.tensor([0, 0, 1, 1])
        pga_bin = torch.tensor([0, 0, 1, 1])
        loss, pairs = pga_zone_rank_loss(logit, target, event, pga_bin)
        self.assertEqual(pairs, 2)
        self.assertAlmostEqual(loss.item(), 0.05)

    def test_seed_is_stable(self):
        self.assertEqual(stable_seed(0, "Barkam_20220609_M59"), 734)

    def test_source_only_training_smoke(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            for event_i, event in enumerate(EVENTS):
                rows = []
                for y in range(12):
                    for x in range(12):
                        rows.append({
                            "lon": 100 + event_i + x * 0.0015,
                            "lat": 30 + event_i + y * 0.0015,
                            "Density": 0.2 if (x + y + event_i) % 5 == 0 else 0.0,
                            "Elevation": 1800 + 15 * x + event_i * 20,
                            "Slope": 4 + y, "Aspect_sin": np.sin(x),
                            "Aspect_cos": np.cos(x), "TWI": 2 + y / 10,
                            "Riverdis": x * 0.2, "NDVI": y / 20,
                            "Faultdis": y * 0.3, "Lithology": x % 3,
                            "Landcover": y % 2,
                            "USGS_PGA": 0.05 + x * 0.002 + y * 0.003,
                        })
                pd.DataFrame(rows).to_csv(folder / f"{event}.csv", index=False)
            regions = load_events(folder)
            with patch("geostn.train.EPOCHS", 2):
                result = run_one(regions, EVENTS[0], 0, "Geo-STN", "Full",
                                 folder, torch.device("cpu"), False)
            self.assertEqual(result["N_Test"], 144)
            self.assertGreater(result["N_Train"], 0)
            self.assertTrue(0 <= result["AUC"] <= 1)


if __name__ == "__main__":
    unittest.main()
