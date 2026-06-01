# Batch Scripts

This page documents the shell launchers at the repo root plus the SLURM wrapper under `postprocess/`.

## Main Shell Launchers

| Script | Purpose | Configurable variables | Example |
|---|---|---|---|
| `run_postprocess.sh` | Dispatches full pipeline, plots-only reruns, or comparison runs | `PLOTS_ONLY`, `COMPARE`, `CONFIG`, `OUT_OVERRIDE`, `ONLY_PLOTS`, `SENSOR_DROP_LAST_COLUMN`, `MODELS`, `NICKNAMES` | `bash run_postprocess.sh configs/config.re2trunc_allnoise_vpln_l.json` |
| `batch_postprocess.sh` | Sequential local batch runner | Positional config list only | `bash batch_postprocess.sh config.A.json config.B.json` |
| `rerun_postprocess.sh` | Rebuilds all all-noise Re2/Re3/Re4 outputs and comparison figures | `CONFIGS` array only | `bash rerun_postprocess.sh` |
| `rerun_plots_only.sh` | Rebuilds figures from existing CSV outputs | `CONFIGS` array only | `bash rerun_plots_only.sh` |
| `rerun_postprocess_init_iter.sh` | Rebuilds init-iter outputs and trunc-vs-init comparisons | `RE_LIST` environment variable | `RE_LIST="2 4" bash rerun_postprocess_init_iter.sh` |
| `run_full_postprocess_1step.sh` | Rebuilds 1-step XDMFs and RMSE tables | `CONFIGS` array only | `bash run_full_postprocess_1step.sh` |
| `run_create_combined_init_iter_dataset.sh` | Wraps the dataset-combining Python script | `PYTHON_BIN`, `RES`, `SOURCE_TEMPLATE`, `INIT_TEMPLATE`, `OUTPUT_TEMPLATE`, `SOURCE_V_FIELD`, `SOURCE_P_FIELD`, `INIT_V_FIELD`, `INIT_P_FIELD`, `VERBOSE`, `STRICT_MISSING`, `STRICT_INCOMPATIBLE` | `bash run_create_combined_init_iter_dataset.sh` |

## SLURM Wrappers

| Script | Purpose | What to edit for a new cluster |
|---|---|---|
| `run_full_postprocess_slurm.sh` | SLURM wrapper around `rerun_postprocess.sh` | `#SBATCH --partition`, `--qos`, `--time`, `--chdir`, `--ntasks` |
| `run_full_postprocess_1step_slurm.sh` | SLURM wrapper around `run_full_postprocess_1step.sh` | Same `#SBATCH` fields |
| `rerun_postprocess_init_iter_slurm.sh` | SLURM wrapper around `rerun_postprocess_init_iter.sh` | Same `#SBATCH` fields |
| `postprocess/submit_postprocess.slurm.sh` | Serial SLURM job that runs `run_postprocess.sh` on each config passed on the command line | `#SBATCH` fields plus conda environment name |

## What To Change For New Data

| Location | Typical edit |
|---|---|
| `CONFIGS=(...)` in rerun scripts | Replace the hardcoded config list with the files you want to rebuild |
| `RE_LIST` in `rerun_postprocess_init_iter.sh` | Restrict the Reynolds regimes processed |
| `SOURCE_TEMPLATE`, `INIT_TEMPLATE`, `OUTPUT_TEMPLATE` | Point the dataset combiner at a new source/init/output directory tree |
| `#SBATCH --chdir` | Point SLURM jobs at the repo checkout on your cluster |

## Execution Notes

1. The scripts activate `gnnpostprocess` when possible, or fall back to `graph` in some launchers.
2. The full rebuild scripts delete existing `sensors/`, `errors/`, `forces/`, and figure directories before recomputing.
3. `run_postprocess.sh --plots-only` does not touch the earlier XDMF or CSV stages.
4. `run_postprocess.sh --compare` requires `--models` and `--output_dir`.

## Expected Layout

The active pipeline expects each model results directory to look like this:

```text
output_dir/
└── model_name/
    ├── xdmf/
    ├── sensors/
    ├── errors/
    ├── forces/
    └── figures/
```

If the directories already exist, the scripts generally reuse or overwrite their contents rather than rebuilding the whole tree from scratch.

---
Prev: [CLI Reference](CLI-Reference.md) | Next: [Model Comparison](Model-Comparison.md)
