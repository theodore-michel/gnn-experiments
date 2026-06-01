# Extending the Codebase

This repo is organized so that each stage has one narrow responsibility. Follow that pattern when adding new behavior.

## Add A New Error Metric

1. Start in [postprocess/compute_errors.py](../postprocess/compute_errors.py).
2. Add a helper function near the existing `_rmse`, `_cumulative_sum`, and `_cum_rmse_horizon` functions.
3. Wire the new metric into `run()` so it is written to a CSV alongside the existing outputs.
4. If the new metric is part of plots, extend [postprocess/plot_results.py](../postprocess/plot_results.py) and, if needed, [postprocess/compare_models.py](../postprocess/compare_models.py).

Suggested conventions:

| Convention | Reason |
|---|---|
| Keep metric names in CSV columns explicit | Downstream plotting code reads columns by name |
| Write one CSV per case plus one summary CSV | Matches the existing per-case and aggregated workflow |
| Preserve the current `model_name` and `case_id` columns | The comparison code depends on them |

## Add A New Plot Type

1. Add a function in [postprocess/plot_results.py](../postprocess/plot_results.py) or [postprocess/compare_models.py](../postprocess/compare_models.py).
2. Reuse `PLOT_CONFIG` and `_save()` so file naming stays consistent.
3. Add a CLI switch only if the new plot family can be turned on or off independently.
4. If the plot depends on new CSV columns, update the producing stage first.

## Add A New Postprocessing Stage

1. Create a new module under `postprocess/`.
2. Follow the existing pattern: `load_json(config)`, `run(config_path)`, `build_parser()`, `main()`.
3. Read and write through `output_dir/model_name/<new_stage>/`.
4. Add the stage to `run_postprocess.sh` if it belongs in the main pipeline.
5. Add a section in the wiki and, if useful, a line in `README.md`.

## Add A New Model To The Existing Workflow

1. Copy one of the `configs/*.json` files from the same Reynolds family.
2. Update `pred_dir`, `dataset_dir`, `configs_pool`, `model_name`, `model_shortname`, and `output_dir`.
3. Keep `prediction_base_name` aligned with the raw file prefix.
4. Keep `feature_map` consistent with the fields present in the XDMFs.
5. If you use a new sensor layout, update both `extract_sensors.py` and the plotting code.

## Files Most Likely To Need Edits

| Change | Files |
|---|---|
| New metric | `postprocess/compute_errors.py`, `postprocess/plot_results.py`, `postprocess/compare_models.py` |
| New force definition | `postprocess/compute_forces.py`, `postprocess/plot_results.py` |
| New XDMF field layout | `postprocess/postprocess_xdmf.py`, `postprocess/utils/xdmf_io.py` |
| New sensor geometry | `postprocess/extract_sensors.py`, `postprocess/utils/xdmf_io.py`, `postprocess/plot_results.py` |
| New batch workflow | `run_postprocess.sh`, `rerun_*.sh`, `run_full_postprocess_*.sh` |

## Practical Guardrails

1. Keep the active root-level pipeline and the legacy `postprocess/metrics` code separate unless you are intentionally migrating the old API.
2. Treat the cached sensor mappings and force CSVs as disposable outputs when changing geometry or physical parameters.
3. When in doubt, add a small smoke config and test the smallest stage that exercises your change.

---
Prev: [Workflow Guide](Workflow-Guide.md) | [Home](Home.md)