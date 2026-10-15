"""Input contract and source-only sampling for the eight-event LORO experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


EVENTS = (
    "Barkam_20220609_M59", "Jiuzhaigou_20170808_M65",
    "Luding_20220905_M66", "Menyuan_20220107_M66",
    "Milin_20171117_M64", "Minxian_20130722_M59",
    "Nepal_20150425_M78_Valagussa", "Wenchuan_20080512_M79_Li",
)
STATIC_FEATURES = (
    "Elevation", "Slope", "Aspect_sin", "Aspect_cos", "TWI",
    "Riverdis", "NDVI", "Faultdis", "Lithology", "Landcover",
)
CATEGORICAL = {"Lithology", "Landcover"}
RAW_FEATURES = (*STATIC_FEATURES, "USGS_PGA")
DOMAIN_FEATURES = tuple(
    name if name in CATEGORICAL else f"{name}_domain_z"
    for name in STATIC_FEATURES
) + ("USGS_PGA",)
ROW_ID = "_region_row_id"


def stable_seed(repeat: int, event: str) -> int:
    digest = hashlib.sha256(event.encode("utf-8")).digest()
    return 42 + 100 * repeat + int.from_bytes(digest[:4], "little") % 1000


def load_events(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read pre-clipped, factor-extracted grid tables; no inventory is shipped."""
    regions = {}
    required = (*RAW_FEATURES, "Density", "lon", "lat")
    for event in EVENTS:
        path = data_dir / f"{event}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{event}: missing columns {sorted(missing)}")
        for col in required:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        frame = frame.reset_index(drop=True)
        if frame.empty or frame.Density.lt(0).any() or frame.Density.gt(1).any():
            raise ValueError(f"{event}: empty event or Density outside [0, 1]")
        if frame.duplicated(["lon", "lat"]).any():
            raise ValueError(f"{event}: duplicate grid coordinates")
        for col in STATIC_FEATURES:
            if col in CATEGORICAL:
                continue
            values = frame[col]
            median = values.median(skipna=True)
            iqr = values.quantile(0.75) - values.quantile(0.25)
            scale = iqr if np.isfinite(iqr) and abs(iqr) >= 1e-6 else values.std(skipna=True)
            if not np.isfinite(scale) or scale <= 1e-6:
                scale = 1.0
            frame[f"{col}_domain_z"] = (values - median) / scale
        pga = frame["USGS_PGA"]
        if pga.nunique(dropna=True) <= 1:
            frame["PGA_Bin"] = np.int8(0)
        else:
            frame["PGA_Bin"] = pd.qcut(
                pga.rank(method="average"), q=4, labels=False,
                duplicates="drop",
            ).fillna(0).astype(np.int8)
        regions[event] = frame
    return regions


def spatial_block_labels(frame: pd.DataFrame) -> pd.Series:
    lon_bin = pd.cut(frame.lon, 8, labels=False, include_lowest=True)
    lat_bin = pd.cut(frame.lat, 8, labels=False, include_lowest=True)
    return (lon_bin.fillna(-1).astype(int).astype(str) + "_"
            + lat_bin.fillna(-1).astype(int).astype(str))


def spatial_proportional_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    blocks = spatial_block_labels(frame)
    counts = blocks.value_counts(sort=False)
    expected = counts.to_numpy(dtype=float) * (n / len(frame))
    quotas = np.floor(expected).astype(int)
    remainder = n - int(quotas.sum())
    order = np.argsort(-(expected - quotas), kind="stable")
    quotas[order[:remainder]] += 1
    parts = []
    for block, count in zip(counts.index, quotas):
        if count:
            block_seed = int.from_bytes(
                hashlib.sha256(str(block).encode()).digest()[:4], "little")
            parts.append(frame.loc[blocks == block].sample(
                n=int(count), random_state=seed + block_seed))
    return pd.concat(parts).sample(frac=1, random_state=seed)


def sample_sources(regions: dict[str, pd.DataFrame], target_event: str,
                   seed: int) -> pd.DataFrame:
    parts = []
    sources = [name for name in EVENTS if name != target_event]
    for event_i, event in enumerate(sources):
        frame = regions[event].copy()
        frame[ROW_ID] = np.arange(len(frame), dtype=np.int64)
        frame["Label"] = frame.Density.gt(0).astype(int)
        pos = frame.loc[frame.Label.eq(1)]
        neg = frame.loc[frame.Label.eq(0)]
        sampled_pos = spatial_proportional_sample(pos, 5000, seed + event_i * 1000)
        sampled_neg = spatial_proportional_sample(
            neg, 5 * len(sampled_pos), seed + event_i * 1000 + 1)
        sampled = pd.concat([sampled_pos, sampled_neg]).sample(
            frac=1, random_state=seed + event_i * 1000).reset_index(drop=True)
        sampled["train_event"] = event
        parts.append(sampled)
    train = pd.concat(parts, ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)
    thinned = []
    for group_i, (_, group) in enumerate(train.groupby(["train_event", "Label"], sort=True)):
        n = max(1, round(len(group) * 0.20))
        thinned.append(spatial_proportional_sample(group, n, seed + group_i * 1000))
    return pd.concat(thinned, ignore_index=True).sample(
        frac=1, random_state=seed).reset_index(drop=True)


def source_train_val_split(train: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = np.empty(len(train), dtype=object)
    for event, index in train.groupby("train_event").groups.items():
        sub = train.loc[index]
        lon, lat = sub.lon, sub.lat
        x = np.floor((lon - lon.min()) / max(float(lon.max() - lon.min()), 1e-9)
                     * 5).clip(0, 4).astype(int)
        y = np.floor((lat - lat.min()) / max(float(lat.max() - lat.min()), 1e-9)
                     * 5).clip(0, 4).astype(int)
        groups[np.asarray(list(index), dtype=int)] = [
            f"{event}:{a}:{b}" for a, b in zip(x, y)]
    labels = train.Label.to_numpy(dtype=int)
    splitter = GroupShuffleSplit(n_splits=20, test_size=0.20, random_state=seed)
    for fit_idx, val_idx in splitter.split(np.zeros(len(train)), labels, groups):
        if len(np.unique(labels[fit_idx])) == 2 and len(np.unique(labels[val_idx])) == 2:
            return fit_idx, val_idx
    raise ValueError("No source-only spatial-block split contains both classes")


def build_patch_indices(frame: pd.DataFrame) -> np.ndarray:
    lon = pd.to_numeric(frame.lon, errors="coerce").round(6)
    lat = pd.to_numeric(frame.lat, errors="coerce").round(6)
    lon_values = np.sort(lon.dropna().unique())
    lat_values = np.sort(lat.dropna().unique())
    lon_to_ix = {value: i for i, value in enumerate(lon_values)}
    lat_to_iy = {value: i for i, value in enumerate(lat_values)}
    lookup = {}
    coordinates = []
    for row, (x, y) in enumerate(zip(lon, lat)):
        grid = (lon_to_ix[x], lat_to_iy[y])
        lookup[grid] = row
        coordinates.append(grid)
    patches = np.empty((len(frame), 9), dtype=np.int64)
    for row, (ix, iy) in enumerate(coordinates):
        patches[row] = [lookup.get((ix + dx, iy + dy), row)
                        for dy in (1, 0, -1) for dx in (-1, 0, 1)]
    return patches


@dataclass
class GridCache:
    frame: pd.DataFrame
    features: np.ndarray
    patches: np.ndarray

    def arrays(self, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center = self.features[row_ids].astype(np.float32)
        patch = self.features[self.patches[row_ids]].reshape(len(row_ids), 3, 3, -1)
        neigh8 = np.delete(patch.reshape(len(row_ids), 9, -1), 4, axis=1)
        patch_chw = np.transpose(patch, (0, 3, 1, 2))
        return center, neigh8.astype(np.float32), patch_chw.astype(np.float32)


@dataclass
class FoldData:
    sample: pd.DataFrame
    fit_idx: np.ndarray
    val_idx: np.ndarray
    features: list[str]
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray]
    val_arrays: tuple[np.ndarray, np.ndarray, np.ndarray]
    target_cache: GridCache
    imputer: SimpleImputer
    scaler: StandardScaler


def build_fold(regions: dict[str, pd.DataFrame], sample: pd.DataFrame,
               fit_idx: np.ndarray, val_idx: np.ndarray,
               target_event: str, representation: str) -> FoldData:
    features = list(RAW_FEATURES if representation == "raw" else DOMAIN_FEATURES)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    scaler.fit(imputer.fit_transform(sample.iloc[fit_idx][features]))
    caches = {}
    for event in (*[e for e in EVENTS if e != target_event], target_event):
        frame = regions[event]
        x = scaler.transform(imputer.transform(frame[features])).astype(np.float32)
        caches[event] = GridCache(frame, x, build_patch_indices(frame))
    arrays = None
    for event, index in sample.groupby("train_event").groups.items():
        positions = np.asarray(list(index), dtype=np.int64)
        row_ids = sample.loc[positions, ROW_ID].to_numpy(dtype=np.int64)
        part = caches[event].arrays(row_ids)
        if arrays is None:
            arrays = tuple(np.empty((len(sample), *a.shape[1:]), dtype=np.float32)
                           for a in part)
        for output, item in zip(arrays, part):
            output[positions] = item
    if arrays is None:
        raise ValueError("No source samples")
    return FoldData(sample, fit_idx, val_idx, features,
                    tuple(a[fit_idx] for a in arrays),
                    tuple(a[val_idx] for a in arrays),
                    caches[target_event], imputer, scaler)
