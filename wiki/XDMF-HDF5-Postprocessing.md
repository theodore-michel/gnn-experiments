# XDMF-HDF5 Postprocessing

The active pipeline uses `meshio` time-series XDMF readers and writers. Each `.xdmf` file has a matching `.h5` sidecar that stores the heavy data arrays.

## Input Layout

The active stages discover raw prediction files by scanning `pred_dir` for `prediction_base_name + *.xdmf`.

Each raw XDMF series is expected to provide:

| Field family | Active semantic use |
|---|---|
| `x0`, `x1`, `x2` | Velocity x, velocity y, and pressure predictions |
| `x3` | Levelset, when `levelset_source = prediction` |
| Highest `x*` index | Nodetype, via `latest_x_feature_key(feature_map)` |
| `y0`, `y1`, `y2` | Target velocity x, target velocity y, and target pressure |

If `levelset_source = dataset`, or if no `feature_map` entry maps to `levelset`, the code loads levelset from a truth XDMF under `dataset_dir` instead.

## Active Standardized Output

`postprocess_xdmf.py` rewrites each case to `output_dir/model_name/xdmf/<case>.xdmf` with these point-data keys:

| Key | Shape | Meaning |
|---|---|---|
| `vx` | `(N,)` | Prediction velocity x |
| `vy` | `(N,)` | Prediction velocity y |
| `v_pred` | `(N, 3)` | Stacked prediction velocity vector with zero z-component |
| `p` | `(N,)` | Prediction pressure |
| `levelset` | `(N,)` | Levelset field |
| `nodetype` | `(N,)` | Node classification |
| `vx_targ` | `(N,)` | Target velocity x |
| `vy_targ` | `(N,)` | Target velocity y |
| `v_targ` | `(N, 3)` | Stacked target velocity vector with zero z-component |
| `p_targ` | `(N,)` | Target pressure |

The writer preserves the original mesh points and cells and writes one time step per aligned rollout step.

## HDF5 Sidecar Behavior

The writer temporarily changes into the output directory before calling `meshio.xdmf.TimeSeriesWriter`, because `meshio` writes relative `.h5` references. The result is that `case.xdmf` and `case.h5` sit next to each other in the same folder.

If the `.h5` sidecar is missing or deleted, the corresponding XDMF series is no longer readable by `meshio`.

## How Readers Use the Files

| Stage | Required fields |
|---|---|
| `extract_sensors.py` | `v_pred`, `v_targ`, `p`, `p_targ` |
| `compute_errors.py` | `v_pred`, `v_targ`, `p`, `p_targ` |
| `compute_forces.py` | `vx`, `vy`, `vx_targ`, `vy_targ`, `p`, `p_targ`, `levelset`, `nodetype` |
| `plot_results.py` | CSV outputs only; it does not reopen XDMF |

## Non-Standard Outputs

The repository contains one older preprocessing path, [postprocess/metrics/process_xdmf.py](../postprocess/metrics/process_xdmf.py), which writes a different set of field names such as `Vx_pred`, `Vy_pred`, `P_pred`, `V_vect_pred`, `V_pred`, `Vx_targ`, and `V_vect_targ` into `processed_xdmf/`.

That legacy layout is documented here because some historical workflows still refer to it, but the active root-level pipeline does not generate those names.

## Practical Rules

1. Keep `prediction_base_name` aligned with the actual prefix on disk, commonly `graph_`.
2. Keep the source and initializer meshes compatible if you use the dataset-editing scripts.
3. If you change `levelset_source`, delete stale outputs so the downstream CSVs are regenerated from the new field source.

---
Prev: [Error Metrics](Error-Metrics.md) | Next: [Plotting](Plotting.md)
