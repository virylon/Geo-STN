# Geo-STN: minimal cross-event reproduction package

This folder contains the final Geo-STN model, the eight-event leave-one-event-out
(LORO) training protocol, six public comparison models, full-grid evaluation,
and path-free per-repeat results. It intentionally excludes raw inventories,
factor rasters, grid tables, trained weights, exploratory experiments and
discussion figures.

## What is released

- `geostn/model.py`: the Geo-STN architecture (historical class name
  `GeoTabSTNV2`, retained for checkpoint compatibility).
- `geostn/protocol.py`: feature normalization, 3 x 3 neighborhood construction,
  source sampling and spatial-block source validation.
- `geostn/train.py`: source-selected LORO training and full-target prediction.
- `geostn/baselines.py`: RF, HGB, MLP, FTTrans, PatchCNN and mean-neighbor
  GraphSAGE comparison implementations. The last model concatenates center
  features with the mean of eight immediate neighbors; it does not learn edge
  weights or use a PyG graph.
- `geostn/metrics.py` and `geostn/audit.py`: AUC, AP, high-density recall at
  the highest-scoring 10% of grids (HDR@10), and event-equal aggregation.
- `results/repeat_metrics.csv`: 880 published-protocol model/setting/event/repeat
  records without local paths or predictions.

## Input contract

The private input directory is **not** part of this repository. It must
contain eight CSV files named after the event identifiers in
`geostn/protocol.py`. Each file represents the complete set of valid
prediction grid cells **already clipped to that event's study boundary**.
The formal prediction grids have a `0.0015` degree coordinate spacing in
both longitude and latitude; the separate 30 m landslide-mask analysis is
not part of this training protocol.
The required columns are:

`lon, lat, Density, Elevation, Slope, Aspect_sin, Aspect_cos, TWI,
Riverdis, NDVI, Faultdis, Lithology, Landcover, USGS_PGA`

`lon` and `lat` are WGS84 grid-center coordinates. `Density` is the mapped
landslide area fraction in `[0, 1]`; a cell is positive for AUC/AP when
`Density > 0`. Lithology and land-cover codes retain their numeric categories.
The CSV row order defines the order of any exported score array. The upstream
GIS extraction and inventory files are not released here; input grid counts
and positive counts should match `results/repeat_metrics.csv` before comparing
numbers. No target-event density value is used for normalization, fitting,
early stopping or checkpoint selection.

## Formal protocol

- Eight continuous static factors use within-event
  `(value - median) / IQR` on all factor-valid grid cells, with the original
  standard-deviation fallback when IQR is degenerate. Lithology, land cover
  and USGS PGA are not event-normalized. A pooled `StandardScaler` for neural
  inputs is then fitted on **source training rows only**.
- For each held-out event and repeat, each of the seven source events contributes
  at most 5,000 positive cells and up to five negatives per positive, sampled
  proportionally over spatial blocks. Each source event/class is then thinned
  to 20% of that sample.
- Source training and validation use a 5 x 5 within-event block partition;
  `GroupShuffleSplit` holds out 20% of blocks. The target event is never in
  this split. The selected checkpoint maximizes source-validation AUC.
- Neural models use AdamW, learning rate `7e-4`, weight decay `1e-3`, batch
  size 512, at most 40 epochs and patience 6. Training uses soft `Density`
  targets with binary cross entropy. Trees regress the same soft target.
- The final Geo-STN Full configuration uses Domain-z, PGA-conditioned ranking
  weight `0.225`, spatial-residual L2 weight `0.30`, and the monotonic PGA
  branch. The ranking loss groups source training cells by event and four
  event-wise PGA quantile bins. Public Table 1 baselines use Domain-z without
  ranking or MonoPGA.
- AUC/AP are calculated over **all** factor-valid target grids, not a sampled
  test set. HDR@10 is the fraction of positive cells in the highest density
  decile captured by the highest-scoring 10% of all valid target cells. Each
  event's ten-repeat mean is calculated first; the paper mean then weights the
  eight events equally.

## Commands

Tested locally with Python 3.9 and the versions in `requirements.txt`.

```bash
python -m geostn.audit recorded --metrics results/repeat_metrics.csv
python -m unittest discover -s tests -v
python -m geostn.train --data-dir /path/to/prepared-grids --output-dir output --model Geo-STN --setting Full --repeats 10 --save-checkpoints --reference-metrics results/repeat_metrics.csv
```

For a single fold, pass `--event Barkam_20220609_M59 --repeats 1`. The same
trainer accepts `--model RF`, `HGB`, `MLP`, `FTTrans`, `PatchCNN` or `GraphSAGE`
with `--setting Domain-z`. Geo-STN also accepts the four other ablation
settings in `geostn/train.py`.

Given a private full-grid density array and checkpoint score array, recompute
one published row independently:

```bash
python -m geostn.audit scores --grid grid_coordinates.npz --scores model_scores.npz --expected results/repeat_metrics.csv --model Geo-STN --setting Full --event Barkam_20220609_M59 --repeat 3
```

The grid NPZ must contain `density`; the score NPZ must contain `scores`, both
in the same full-grid row order. The package includes recorded fold metrics,
not the private pixel data or predictions needed to independently reproduce
every fold from scratch. This distinction is intentional.

`VERIFICATION.md` records checks performed against the original private
experiment before this folder was prepared. Those checks do not substitute
for independent retraining on the unavailable grid data.

## Release boundary

No private GIS data, trained checkpoints, absolute local paths, NDSHA
exploration, oracle-selected experiments, or figure-generation scripts belong
in this folder. Before publishing, review data and software licensing for any
separate materials you choose to add.
