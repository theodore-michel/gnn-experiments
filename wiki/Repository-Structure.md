# Repository Structure

This is the repo-local tree that the wiki documents.

```text
gnn-experiments/
├── README.md
├── environment.yml
├── requirements.txt
├── run_postprocess.sh
├── batch_postprocess.sh
├── rerun_postprocess.sh
├── rerun_plots_only.sh
├── rerun_postprocess_init_iter.sh
├── rerun_postprocess_init_iter_slurm.sh
├── run_full_postprocess_slurm.sh
├── run_full_postprocess_1step.sh
├── run_full_postprocess_1step_slurm.sh
├── run_create_combined_init_iter_dataset.sh
├── postprocess/
│   ├── postprocess_xdmf.py
│   ├── extract_sensors.py
│   ├── compute_errors.py
│   ├── compute_forces.py
│   ├── compare_models.py
│   ├── plot_results.py
│   ├── config.example.json
│   ├── config.re2trunc_allnoise_vpln_full.json
│   ├── Re2_onecyl_VPLN_l.json
│   ├── utils/xdmf_io.py
│   ├── metrics/...
│   └── visualization/...
├── scripts/
│   ├── create_combined_init_iter_dataset.py
│   └── create_predict_edit_dataset.py
├── configs/
│   ├── config.re2_1step_VPLN.json
│   ├── config.re2_1step_VPLN_ld.json
│   ├── config.re2_1step_VPN.json
│   ├── config.re2_allnoise_vpln_ld_init-iter.json
│   ├── config.re2trunc_allnoise_vpln_l.json
│   ├── config.re2trunc_allnoise_vpln_ld.json
│   ├── config.re2trunc_allnoise_vpn_l.json
│   ├── config.re3_1step_VPLN.json
│   ├── config.re3_1step_VPLN_ld.json
│   ├── config.re3_1step_VPN.json
│   ├── config.re3_allnoise_vpln_ld_init-iter.json
│   ├── config.re3trunc_allnoise_vpln_l.json
│   ├── config.re3trunc_allnoise_vpln_ld.json
│   ├── config.re3trunc_allnoise_vpn_l.json
│   ├── config.re4_1step_VPLN.json
│   ├── config.re4_1step_VPLN_ld.json
│   ├── config.re4_1step_VPN.json
│   ├── config.re4_allnoise_vpln_ld_init-iter.json
│   ├── config.re4trunc_allnoise_vpln_l.json
│   ├── config.re4trunc_allnoise_vpln_ld.json
│   └── config.re4trunc_allnoise_vpn_l.json
├── results_ReX_1step/
├── results_ReX_allnoise/
└── results_ReX_allnoise_full/
```

## Top-Level Files

| File | Purpose |
|---|---|
| `README.md` | Quick overview and common commands |
| `environment.yml` | Conda environment for the active pipeline |
| `requirements.txt` | Minimal pip dependency list |
| `run_postprocess.sh` | Main dispatcher for full run, plots-only mode, and comparison mode |
| `batch_postprocess.sh` | Sequential local batch launcher |
| `rerun_postprocess.sh` | Rebuilds the all-noise Re2/Re3/Re4 outputs from scratch |
| `rerun_plots_only.sh` | Regenerates figures from existing CSV outputs |
| `rerun_postprocess_init_iter.sh` | Rebuilds init-iter outputs and trunc-vs-init comparisons |
| `rerun_postprocess_init_iter_slurm.sh` | SLURM wrapper for the init-iter rebuild |
| `run_full_postprocess_slurm.sh` | SLURM wrapper for the all-noise rebuild |
| `run_full_postprocess_1step.sh` | Rebuilds the 1-step model outputs |
| `run_full_postprocess_1step_slurm.sh` | SLURM wrapper for the 1-step rebuild |
| `run_create_combined_init_iter_dataset.sh` | Wrapper for dataset stitching |

## `postprocess/`

| Path | Purpose |
|---|---|
| `postprocess_xdmf.py` | Standardizes raw prediction XDMFs and writes aligned output series |
| `extract_sensors.py` | Samples sensor values from standardized XDMFs |
| `compute_errors.py` | Computes RMSE tables and cumulative error curves |
| `compute_forces.py` | Computes surface force time series and summaries |
| `plot_results.py` | Renders single-model figures from CSV outputs |
| `compare_models.py` | Renders multi-model comparison figures |
| `utils/xdmf_io.py` | Shared XDMF, geometry, and case-discovery helpers |
| `config.example.json` | Commented template for the active config schema |
| `config.re2trunc_allnoise_vpln_full.json` | Concrete example config used for smoke/full runs |
| `Re2_onecyl_VPLN_l.json` | Legacy nested-schema config example |
| `metrics/` | Legacy compatibility package |
| `visualization/` | Legacy compatibility plotting package |

## `results_*`

These directories are generated output trees or example caches. The active scripts write `xdmf/`, `sensors/`, `errors/`, `forces/`, and `figures/` under `output_dir/model_name`.

---
Prev: [Getting Started](Getting-Started.md) | Next: [Configuration](Configuration.md)
