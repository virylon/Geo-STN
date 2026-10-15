# Local verification record

These checks were performed against the original experiment files before
preparing this path-free public folder. Private data, scores and checkpoints
are not included.

1. The released Geo-STN class loaded a source-validation-selected Full
   checkpoint without missing or unexpected keys. With identical test tensors,
   its logits matched the original class exactly (maximum difference `0.0`).
   All four released neural comparison classes loaded their formal checkpoints.
2. On the Barkam held-out fold at repeat 0, the original and released sampling
   routines selected the same 28,071 source rows in the same order. Both
   produced the same 22,162 training and 5,909 validation row indices. The
   3 x 3 neighborhood indices also matched in the checked grid subset.
3. Using the original full-grid density and saved per-repeat score arrays,
   AUC, AP and HDR@10 were independently recomputed for all 80 Geo-STN Full
   runs (8 events x 10 repeats). Grid counts, positive counts, high-density
   denominators and high-density hits also matched every published row.
4. The path-free result file has 880 unique records: 400 Geo-STN ablation
   runs and 480 public-baseline runs. Every model/setting has 8 events x 10
   repeats. Event-equal aggregation gives Geo-STN Full mean AUC `0.737412`
   and mean HDR@10 `0.350845`.

This release makes the analysis logic inspectable. Raw-data ingestion from
landslide polygons and the private full-grid inputs are outside its scope;
therefore the released metric CSV alone cannot independently establish the
correctness of every reported prediction.
