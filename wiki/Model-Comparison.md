# Model Comparison

The active multi-model comparison entry point is [postprocess/compare_models.py](../postprocess/compare_models.py).

## What It Compares

| Comparison family | Data source | Aggregation |
|---|---|---|
| Sensor traces | `sensors/sensor_data.csv` | Overlaid per sensor and per case |
| RMSE summary | `errors/per_case_rmse.csv`, `errors/cumulative_rmse_mean.csv` | Mean and standard deviation across cases |
| Force summaries | `forces/forces_summary.csv` and `forces/forces_<case>.csv` | Mean and standard deviation across cases |

## Required Inputs

Every model directory passed to `--models` must contain a full active pipeline output tree:

```text
model_root/
├── sensors/sensor_data.csv
├── errors/per_case_rmse.csv
├── errors/cumulative_rmse_mean.csv
└── forces/forces_summary.csv
```

The code checks that every model has the same case IDs. If the sets differ, it raises a `ValueError` before plotting.

## How Results Are Aggregated

### Sensors

For each case and each sensor, the comparison plot overlays the prediction traces from all models plus the ground truth trace taken from the first model’s sensor CSV. The sensor ordering follows the wake layout `p1` to `p9`, with an optional cropped 3x2 layout when `sensor_drop_last_column` is enabled.

### RMSE

The comparison code reads `rmse_total_mean`, `rmse_step1`, `rmse_50step`, and `rmse_all` from each model’s `per_case_rmse.csv`, then computes:

| Aggregate | Meaning |
|---|---|
| `rmse_1step_mean` / `std` | Mean and standard deviation across cases of `rmse_step1` |
| `rmse_50step_mean` / `std` | Mean and standard deviation across cases of `rmse_50step` |
| `rmse_all_mean` / `std` | Mean and standard deviation across cases of `rmse_all` |

These rows are written to `compare_rmse_summary.csv`.

### Forces

`forces_summary.csv` is used to build grouped bar charts for `Fx` and `Fy`. For each case and each model, the script plots the mean and standard deviation of the predicted and target forces.

## Adding A New Model To A Comparison Run

1. Run the full active pipeline for the model so that `sensors/`, `errors/`, and `forces/` exist.
2. Pass the model result directory to `--models`.
3. Keep the config argument aligned with the same `configs_pool` used by the other models.
4. Optionally pass `--nicknames` to override the displayed labels.

Example:

```bash
bash run_postprocess.sh --compare configs/config.re2trunc_allnoise_vpln_l.json \
  --models results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPN_l \
           results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l \
           results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_ld \
  --nicknames "VPN-l" "VPLN-l" "VPLN-ld" \
  --output_dir results_ReX_allnoise/comparison_Re2
```

## Caveats

1. Comparison plots do not recompute metrics; they only read existing CSVs.
2. The plotting code uses the first model’s sensor CSV as the source of the case/sensor layout.
3. `compare_models.py` is strict about case-set equality, so partial runs are not directly comparable.

---
Prev: [Batch Scripts](Batch-Scripts.md) | Next: [Workflow Guide](Workflow-Guide.md)
