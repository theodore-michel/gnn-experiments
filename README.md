# gnn-experiments — GNN Fluid Dynamics Experiment Tracker

Lightweight CLI-based experiment registry for **graph-physics** GNN training runs
on cylinder flow simulations. Every run is recorded as a self-contained YAML file
committed to this repository, making experiments fully traceable and reproducible.

---

## Repository layout

```
gnn-experiments/
├── run_manager.py          # CLI (click): register, update, relaunch, compare, sync-sheet
├── launch.sh               # Wrapper: register → sbatch → capture job ID
├── environment.yml         # Conda env spec for the "graph" environment
├── schema.yaml             # Documented YAML schema for a single run record
├── README.md               # This file
├── registry/               # One YAML file per run (git-tracked)
│   ├── re100_full_20260305_001.yaml
│   └── ...
└── postprocess/            # Post-processing scaffold (to be implemented)
    ├── __init__.py
    ├── metrics/            # Error statistics, custom metric computation
    │   ├── __init__.py
    │   └── README.md
    ├── visualization/      # Plotting, ParaView XDMF, video generation
    │   ├── __init__.py
    │   └── README.md
    └── utils/              # Shared utilities: data loading, XDMF parsing
        ├── __init__.py
        └── README.md
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Conda env `graph`** | All dependencies pre-installed. See `environment.yml` for the full spec. |
| **SLURM** | Job submission via `sbatch`. The `sacct` command is used for accounting. |
| **wandb** | Offline mode (`WANDB_MODE=offline`) is the default in `job.sh`. |
| **Clean graph-physics checkout** | Needed for source-diff computation. Set `GRAPH_PHYSICS_REF` env var. |

### One-time setup

```bash
# 1. Clone or ensure the clean graph-physics reference exists
#    (a checkout of the graph-physics repo at the version your runs are based on)
export GRAPH_PHYSICS_REF=/path/to/clean/graph-physics

# 2. Activate the conda env
conda activate graph

# 3. Install the few extra CLI deps (click, rich, pyyaml) if not already present
pip install click rich pyyaml

# 4. (Optional) Initialize this folder as a git repo for version-controlled records
cd gnn-experiments
git init && git add -A && git commit -m "Initial experiment tracker"
```

---

## Daily workflow

### 1. Launch a new training run

```bash
# Option A — use the launch wrapper (recommended)
./launch.sh /path/to/onecyl-T-Re2-VPLN-lgud training_config/onecyl_Re2.json \
    --notes "Baseline Re=100, physics loss, 20 epochs"

# Option B — register manually, then sbatch yourself
python run_manager.py register \
    --run-dir /path/to/onecyl-T-Re2-VPLN-lgud \
    --notes "Baseline Re=100"
# → prints: re100_full_20260305_001

cd /path/to/onecyl-T-Re2-VPLN-lgud
sbatch job.sh
```

`launch.sh` will:
1. Activate `conda activate graph`
2. Call `run_manager.py register` to snapshot all metadata (parameters JSON, job.sh,
   source diff, model config, loss config, noise, dataset path, …)
3. Submit via `sbatch job.sh`
4. Write the SLURM job ID back into the run record

### 2. Update metrics after training completes

```bash
python run_manager.py update --run-id re100_full_20260305_001
```

This reads:
- `wandb-summary.json` from the offline wandb folder → final metrics
- wandb Python API (if online) → best-across-epochs metrics + run URL
- `sacct` → wall-clock time, GPU type, partition

### 3. Compare runs

```bash
# All runs, sorted by val/rollout_rmse (default)
python run_manager.py compare

# Filter by dataset, show top 5
python run_manager.py compare --dataset re100 --top 5

# Sort by a different metric
python run_manager.py compare --sort-by val/1step_rmse --all-metrics
```

### 4. Commit the record

```bash
cd gnn-experiments
git add registry/
git commit -m "Add run re100_full_20260305_001"
```

---

## Relaunch workflow

Reproduce or re-submit a previous run from its record alone:

```bash
# Dry run — inspect what would be created
python run_manager.py relaunch --run-id re100_full_20260305_001 --dry-run

# Actual relaunch
python run_manager.py relaunch --run-id re100_full_20260305_001 \
    --override-notes "Re-run with fixed noise"

# Override the dataset path
python run_manager.py relaunch --run-id re100_full_20260305_001 \
    --override-dataset /new/path/to/dataset/train
```

This will:
1. Copy the clean `graph-physics` base (from `GRAPH_PHYSICS_REF`)
2. Apply the stored source diff to reconstruct the exact run folder
3. Write the stored `job.sh` and parameters JSON
4. Submit via `sbatch`
5. Register the new run with `parent_run_id` pointing to the original

---

## Run record fields

Each run record (`registry/<run_id>.yaml`) captures:

| Section | Fields |
|---------|--------|
| **Identity** | `run_id`, `parent_run_id`, `created_at`, `slurm_job_id` |
| **Source** | `base_commit`, `source_diff`, `run_dir` |
| **Dataset** | `name`, `re_case`, `variant`, `path`, `meta_path` |
| **Model** | `type`, `message_passing_num`, `hidden_size`, `node_input_size`, `output_size`, `edge_input_size`, `num_heads` |
| **Features** | `input_fields`, `no_edge_feature` |
| **Loss** | `type` (list), `weights` (list), `gradient_method` |
| **Noise** | `amplitudes`, `index_start`, `index_end` |
| **Training** | `num_epochs`, `init_lr`, `batch_size`, `warmup`, `seed`, `num_workers`, `prefetch_factor`, `project_name` |
| **Embedded files** | `parameters_json` (full JSON), `job_sh` (full shell script) |
| **SLURM** | `partition`, `gres`, `gpu_type`, `wall_clock_seconds`, `node` |
| **Checkpoint** | `path`, `save_name` |
| **wandb** | `run_id`, `url`, `project` |
| **Metrics** | `final/{val_rollout_rmse, val_1step_rmse, train_loss}`, `best/{...}` |
| **Notes** | Free text |

See [schema.yaml](schema.yaml) for the full annotated template.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPH_PHYSICS_REF` | `../graph-physics` (relative to this repo) | Path to clean graph-physics checkout for diffs |
| `WANDB_ENTITY` | *(none)* | wandb entity for API queries in `update` |
| `WANDB_MODE` | Set to `offline` in `job.sh` | wandb logging mode |

---

## Adding post-processing scripts

The `postprocess/` directory contains stub packages for future implementation:

- **`metrics/`** — Error computation (RMSE, MAE, drag/lift coefficients, …)
- **`visualization/`** — Matplotlib plots, ParaView XDMF export, video generation
- **`utils/`** — Shared I/O helpers (XDMF readers, mesh loading, …)

Each subfolder has a `README.md` describing its intended scope. To add a new script:

1. Create your `.py` file in the appropriate subfolder.
2. Import shared utilities from `postprocess.utils`.
3. Add a CLI entry point (use `click`) if the script should be callable standalone.
4. Document usage in the subfolder's `README.md`.

---

## Datasets

| Label | Re | Cases | Path pattern |
|-------|----|-------|--------------|
| `re100_full` | 100 (Re2) | 120 train | `dataset_onecyl_Re1e2_gmsh_redo/train` |
| `re1000_full` | 1 000 (Re3) | 120 train | `dataset_onecyl_Re1e3_gmsh_redo/train` |
| `re10000_full` | 10 000 (Re4) | 120 train | `dataset_onecyl_Re1e4_gmsh_redo/train` |

Variants with `_mini` suffix use 50 training cases.
