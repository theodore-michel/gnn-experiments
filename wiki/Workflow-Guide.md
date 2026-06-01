# Workflow Guide

This is the shortest complete route from raw prediction outputs to final figures and tables.

## Example Scenario

Use `configs/config.re2trunc_allnoise_vpln_l.json` as the working example. It points at raw predictions in `pred_dir`, truth data in `dataset_dir`, and writes results to `./results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l`.

## Step 1. Verify The Config

Check these keys before you run anything:

| Key | What must be true |
|---|---|
| `pred_dir` | The directory exists and contains `graph_*.xdmf` files |
| `dataset_dir` | Truth XDMFs exist if `levelset_source = dataset` |
| `configs_pool` | The pickle is readable and contains the case IDs |
| `model_name` | Matches the folder name you want under `output_dir` |
| `feature_map` | Contains the right `x*` fields for your prediction layout |

## Step 2. Standardize The XDMFs

```bash
python -m postprocess.postprocess_xdmf configs/config.re2trunc_allnoise_vpln_l.json
```

This step aligns prediction and target fields by `target_step_offset`, crops to `rollout_steps`, injects `levelset` when needed, and writes standardized XDMFs under `xdmf/`.

## Step 3. Extract Sensors

```bash
python -m postprocess.extract_sensors configs/config.re2trunc_allnoise_vpln_l.json
```

For `sensor_mode = auto`, the code reads geometry from `configs_pool` and places sensors in the wake. It then writes `sensors/sensor_data.csv` plus sensor-node mapping caches.

## Step 4. Compute RMSE Tables

```bash
python -m postprocess.compute_errors configs/config.re2trunc_allnoise_vpln_l.json
```

This produces per-case RMSE, cumulative RMSE, mean/std across cases, and summary statistics in `errors/`.

## Step 5. Compute Forces

```bash
python -m postprocess.compute_forces configs/config.re2trunc_allnoise_vpln_l.json
```

This integrates pressure and viscous traction over the obstacle boundary and writes `forces/forces_<case>.csv` plus `forces/forces_summary.csv`.

## Step 6. Render Figures

```bash
python -m postprocess.plot_results configs/config.re2trunc_allnoise_vpln_l.json
```

This reads the CSV outputs and creates sensor plots, force curves, force bars, rollout RMSE plots, and per-case RMSE bars.

## Step 7. Compare Models

Once you have several fully processed model folders, compare them:

```bash
python -m postprocess.compare_models configs/config.re2trunc_allnoise_vpln_l.json \
  --models \
    results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPN_l \
    results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l \
    results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_ld \
  --nicknames "VPN-l" "VPLN-l" "VPLN-ld" \
  --output_dir results_ReX_allnoise/comparison_Re2
```

## What You Should See

| Stage | Success signal |
|---|---|
| `postprocess_xdmf` | A non-empty `xdmf/` directory and a summary saying how many cases were written |
| `extract_sensors` | `sensor_data.csv` with rows for every sensor and timestep |
| `compute_errors` | `per_case_rmse.csv`, `cumulative_rmse_mean.csv`, `summary_statistics.csv` |
| `compute_forces` | `forces_summary.csv` plus one CSV per case |
| `plot_results` | PNG files under `figures/` |

## Common Failure Points

1. A wrong `prediction_base_name` means no cases are discovered.
2. A bad `configs_pool` path breaks auto sensor placement and Reynolds ordering.
3. If `target_step_offset` is too large, `postprocess_xdmf.py` writes zero processed cases.
4. If you change sensor placement, delete the cached `sensor_points_<case>.json` files before rerunning extraction.

---
Prev: [Model Comparison](Model-Comparison.md) | Next: [Extending the Codebase](Extending-the-Codebase.md)
