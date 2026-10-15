"""Source-only LORO training on user-supplied, pre-clipped full-grid tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .baselines import NEURAL_BASELINES, tree_baseline
from .metrics import evaluate_grid
from .model import GeoTabSTNV2
from .protocol import (EVENTS, FoldData, build_fold, load_events,
                       sample_sources, source_train_val_split, stable_seed)


EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 512
LR = 7e-4
WEIGHT_DECAY = 1e-3
RANK_MARGIN = 0.05
RANK_MIN_TARGET_DIFF = 1e-4
MAX_PAIRS_PER_GROUP = 512
GEO_RANK_WEIGHT = 0.225
GEO_SPATIAL_WEIGHT = 0.30
BASELINE_RANK_WEIGHT = 0.15
SETTINGS = {
    "Raw": ("raw", False, False),
    "Domain-z": ("domain", False, False),
    "Domain-z+Rank": ("domain", True, False),
    "Domain-z+MonoPGA": ("domain", False, True),
    "Full": ("domain", True, True),
}


def pga_zone_rank_loss(logit: torch.Tensor, target: torch.Tensor,
                       event: torch.Tensor, pga_bin: torch.Tensor):
    losses = []
    n_pairs = 0
    for e in event.unique():
        for b in pga_bin[event == e].unique():
            idx = torch.where((event == e) & (pga_bin == b))[0]
            if idx.numel() < 2:
                continue
            hi, lo = torch.where((target[idx, None] - target[None, idx])
                                 > RANK_MIN_TARGET_DIFF)
            if hi.numel() == 0:
                continue
            if hi.numel() > MAX_PAIRS_PER_GROUP:
                chosen = torch.randperm(hi.numel(), device=logit.device)[:MAX_PAIRS_PER_GROUP]
                hi, lo = hi[chosen], lo[chosen]
            losses.append(torch.relu(RANK_MARGIN - (logit[idx[hi]]
                                                    - logit[idx[lo]])).mean())
            n_pairs += int(hi.numel())
    loss = torch.stack(losses).mean() if losses else logit.new_tensor(0.0)
    return loss, n_pairs


def _soft_targets(data: FoldData) -> np.ndarray:
    return np.clip(data.sample.Density.to_numpy(dtype=float), 0, 1).astype(np.float32)


def _forward(model, name: str, center, neigh8, patch):
    if name == "Geo-STN":
        return model(center, neigh8.mean(dim=1))
    if name == "PatchCNN":
        return model(patch)
    if name == "GraphSAGE":
        return model(center, neigh8.mean(dim=1))
    return model(center)


def _predict_arrays(model, name: str, arrays, device, batch_size=2048) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(arrays[0]), batch_size):
            items = [torch.from_numpy(a[start:start + batch_size]).to(device)
                     for a in arrays]
            chunks.append(torch.sigmoid(_forward(model, name, *items)).cpu().numpy())
    return np.concatenate(chunks)


def fit_neural(name: str, data: FoldData, setting: str,
               seed: int, device: torch.device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, use_rank, monotonic = SETTINGS[setting]
    model = (GeoTabSTNV2(data.features, monotonic)
             if name == "Geo-STN" else NEURAL_BASELINES[name](len(data.features)))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y = _soft_targets(data)
    event_codes = pd.Categorical(data.sample.train_event.astype(str)).codes.astype(np.int64)
    pga_bins = data.sample.PGA_Bin.to_numpy(dtype=np.int64)
    if name == "Geo-STN":
        center, neigh8, _ = data.train_arrays
        ds = TensorDataset(
            torch.from_numpy(center),
            torch.from_numpy(neigh8.mean(axis=1).astype(np.float32)),
            torch.from_numpy(y[data.fit_idx]),
            torch.from_numpy(event_codes[data.fit_idx]),
            torch.from_numpy(pga_bins[data.fit_idx]),
        )
    else:
        ds = TensorDataset(
            *(torch.from_numpy(x) for x in data.train_arrays),
            torch.from_numpy(y[data.fit_idx]),
            torch.from_numpy(event_codes[data.fit_idx]),
            torch.from_numpy(pga_bins[data.fit_idx]),
        )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_auc, best_epoch, bad, best_state, best_pairs = -np.inf, 0, 0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_pairs = 0
        for batch in loader:
            if name == "Geo-STN":
                center_b, neigh_b, y_b, event_b, pga_b = [x.to(device) for x in batch]
                parts = model(center_b, neigh_b, return_parts=True)
                logit = parts["logit"]
                loss = nn.functional.binary_cross_entropy_with_logits(logit, y_b)
                loss = loss + GEO_SPATIAL_WEIGHT * parts["spatial_delta"].square().mean()
                rank_weight = GEO_RANK_WEIGHT
            else:
                center_b, neigh_b, patch_b, y_b, event_b, pga_b = [x.to(device) for x in batch]
                logit = _forward(model, name, center_b, neigh_b, patch_b)
                loss = nn.functional.binary_cross_entropy_with_logits(logit, y_b)
                rank_weight = BASELINE_RANK_WEIGHT
            if use_rank:
                rank_loss, pairs = pga_zone_rank_loss(logit, y_b, event_b, pga_b)
                loss = loss + rank_weight * rank_loss
                epoch_pairs += pairs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_scores = _predict_arrays(model, name, data.val_arrays, device,
                                     batch_size=4096 if name == "Geo-STN" else 2048)
        val_labels = y[data.val_idx] > 0
        val_auc = float(roc_auc_score(val_labels, val_scores))
        if val_auc > best_auc + 1e-5:
            best_auc, best_epoch, bad, best_pairs = val_auc, epoch, 0, epoch_pairs
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("No valid source-selected checkpoint")
    model.load_state_dict(best_state)
    return model, best_auc, best_epoch, best_pairs


def predict_target(model, name: str, cache, device) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(cache.frame), 2048):
            ids = np.arange(start, min(start + 2048, len(cache.frame)))
            center, neigh8, patch = cache.arrays(ids)
            arrays = [torch.from_numpy(a).to(device) for a in (center, neigh8, patch)]
            logits = _forward(model, name, *arrays)
            chunks.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(chunks)


def run_one(regions, target_event: str, repeat: int, model_name: str,
            setting: str, output_dir: Path, device: torch.device,
            save_checkpoint: bool) -> dict:
    seed = stable_seed(repeat, target_event)
    sample = sample_sources(regions, target_event, seed)
    fit_idx, val_idx = source_train_val_split(sample, seed)
    representation, _, monotonic = SETTINGS[setting]
    data = build_fold(regions, sample, fit_idx, val_idx, target_event, representation)
    if model_name in {"RF", "HGB"}:
        model = tree_baseline(model_name, seed)
        model.fit(sample.iloc[fit_idx][data.features], _soft_targets(data)[fit_idx])
        val_scores = np.clip(model.predict(sample.iloc[val_idx][data.features]), 0, 1)
        val_auc = float(roc_auc_score(_soft_targets(data)[val_idx] > 0, val_scores))
        best_epoch, rank_pairs = 0, 0
        scores = np.clip(model.predict(data.target_cache.frame[data.features]), 0, 1)
    else:
        model, val_auc, best_epoch, rank_pairs = fit_neural(
            model_name, data, setting, seed, device)
        scores = predict_target(model, model_name, data.target_cache, device)
    if save_checkpoint:
        folder = output_dir / "checkpoints" / model_name / setting / target_event
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"repeat_{repeat:02d}.{'joblib' if model_name in {'RF', 'HGB'} else 'pt'}"
        if model_name in {"RF", "HGB"}:
            joblib.dump({"model": model, "features": data.features}, path)
        else:
            torch.save({"state_dict": {k: v.detach().cpu()
                                       for k, v in model.state_dict().items()},
                        "features": data.features, "imputer": data.imputer,
                        "scaler": data.scaler, "monotonic_pga": monotonic}, path)
    metrics = evaluate_grid(data.target_cache.frame.Density.to_numpy(), scores,
                            data.target_cache.frame.PGA_Bin.to_numpy())
    return {"Model": model_name, "Setting": setting, "Test_Event": target_event,
            "Repeat": repeat, "Seed": seed, "N_Train": len(fit_idx),
            "N_Validation": len(val_idx), "N_Features": len(data.features),
            "Source_Val_AUC": val_auc, "Best_Epoch": best_epoch,
            "Rank_Pairs": rank_pairs, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("Geo-STN", "RF", "HGB", *NEURAL_BASELINES),
                        default="Geo-STN")
    parser.add_argument("--setting", choices=tuple(SETTINGS), default="Full")
    parser.add_argument("--event", choices=EVENTS)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--reference-metrics", type=Path,
                        help="Check event grid and positive counts against published results")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.model in {"RF", "HGB"} and args.setting not in {"Raw", "Domain-z"}:
        parser.error("Tree models were evaluated only with Raw or Domain-z")
    if args.model != "Geo-STN" and args.setting in {"Domain-z+MonoPGA", "Full"}:
        parser.error("MonoPGA is a Geo-STN architecture constraint")
    events = (args.event,) if args.event else EVENTS
    regions = load_events(args.data_dir)
    if args.reference_metrics is not None:
        reference = pd.read_csv(args.reference_metrics)
        for event in EVENTS:
            rows = reference.loc[reference.Test_Event.eq(event)]
            if rows.empty or rows.N_Test.nunique() != 1 or rows.Positive_Test.nunique() != 1:
                raise ValueError(f"Inconsistent reference population for {event}")
            observed = (len(regions[event]), int(regions[event].Density.gt(0).sum()))
            expected = (int(rows.N_Test.iloc[0]), int(rows.Positive_Test.iloc[0]))
            if observed != expected:
                raise ValueError(f"{event}: grid/positive counts {observed} != {expected}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = args.output_dir / "metrics.csv"
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not existing.empty and existing.duplicated(
            ["Model", "Setting", "Test_Event", "Repeat"]).any():
        raise ValueError("Existing metrics contain duplicate keys")
    completed = (set(map(tuple, existing[["Model", "Setting", "Test_Event", "Repeat"]]
                               .itertuples(index=False, name=None))) if not existing.empty else set())
    for repeat in range(args.repeats):
        for event in events:
            key = (args.model, args.setting, event, repeat)
            if key in completed:
                continue
            row = run_one(regions, event, repeat, args.model, args.setting,
                          args.output_dir, device, args.save_checkpoints)
            pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
            print(f"{event} repeat {repeat}: AUC={row['AUC']:.4f}, "
                  f"HDR10={row['HDR10']:.4f}, source epoch={row['Best_Epoch']}", flush=True)


if __name__ == "__main__":
    main()
