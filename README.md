# GNN CFD Postprocessing Repository

This repository is dedicated to postprocessing Graph Neural Network predictions for cylinder-flow CFD rollouts.

## Environment Setup

Use the `graph` conda environment:

```bash
conda activate graph
```

## Pipeline Architecture

The pipeline consists of five sequential stages, all consuming a unified JSON config:

1. **`postprocess_xdmf.py`** — Standardize raw prediction XDMF meshes (triangulation, field extraction)
2. **`extract_sensors.py`** — Extract point-cloud sensor signals (velocity, pressure) at specified locations
3. **`compute_errors.py`** — Compute RMSE (1-step, 50-step, cumulative) per case and aggregated statistics
4. **`compute_forces.py`** — Integrate surface forces (viscous and pressure) on cylinder
5. **`plot_results.py`** — Generate publication-ready figures (sensors, forces, RMSE, cumulative curves)

Optional:
- **`compare_models.py`** — Side-by-side comparison across multiple model outputs

## Configuration

See [postprocess/config.example.json](postprocess/config.example.json) for a fully-documented template with all parameters including:
- I/O paths (predictions, dataset, output directory)
- Feature/target field mappings
- Sensor placement and layout
- Force computation settings (viscosity, sign convention)
- Levelset source and rollout limits

---

## Typical Workflows (TL;DR)

| Task | Command |
|------|---------|
| **Postprocess one model** | `bash run_postprocess.sh postprocess/config.my_model.json` |
| **Replot without recompute** | `bash run_postprocess.sh --plots-only postprocess/config.my_model.json` |
| **Compare multiple models** | `bash run_postprocess.sh --compare postprocess/config.base.json --models ./res/mA ./res/mB --output_dir ./comparison` |
| **Process batch (local)** | `bash batch_postprocess.sh config.A.json config.B.json config.C.json` |
| **Process batch (SLURM)** | `bash postprocess/submit_postprocess.slurm.sh config.A.json config.B.json` |
| **Recompute single stage** | `python -m postprocess.compute_forces postprocess/config.my_model.json` |
| **Clear stale cache** | `rm results/model_name/forces/forces_*.csv` |

---

## Quick Start: Single Model Postprocessing

**Full pipeline** (XDMF standardization → sensors → errors → forces → plots):

```bash
bash run_postprocess.sh postprocess/config.my_model.json
```

**Verified smoke test** (realistic 600-step rollout on 15 cases, ~2 minutes):

```bash
bash run_postprocess.sh postprocess/config.re2trunc_allnoise_vpln_smoke.json
```

Output: `results_smoke/onecyl_Re2trunc_allnoise_VPLN_l/` with subdirectories `xdmf/`, `sensors/`, `errors/`, `forces/`, `figures/`

---

## Use Case: Replot Without Recompute

Skip all computation and regenerate figures from existing output (useful for tweaking plot settings):

```bash
bash run_postprocess.sh --plots-only postprocess/config.my_model.json
```

With custom output path:

```bash
bash run_postprocess.sh --plots-only postprocess/config.my_model.json \
  --output_dir ./results/my_custom_output
```

---

## Use Case: Multi-Model Comparison

Generate side-by-side comparison figures for multiple models:

```bash
bash run_postprocess.sh --compare postprocess/config.any_model.json \
  --models /path/to/results/model_A /path/to/results/model_B /path/to/results/model_C \
  --nicknames "Model A" "Model B" "Model C" \
  --output_dir ./results/comparison_ABC
```

**Example with locally-cached outputs:**

```bash
bash run_postprocess.sh --compare postprocess/config.re2trunc_allnoise_vpln_smoke.json \
  --models \
    ./results_smoke/onecyl_Re2trunc_allnoise_VPLN_l \
    ./results_full/onecyl_Re2trunc_allnoise_VPLN_l \
  --nicknames "Smoke" "Full" \
  --output_dir ./results/comparison_smoke_vs_full
```

**Requirements:**
- Each model folder must contain `sensors/`, `errors/`, `forces/` subdirectories (output from a full pipeline run)
- Config can be any model's original config (used only to load `configs_pool` for case metadata)
- Nicknames are optional (defaults to folder names)

---

## Advanced: Stage-by-Stage Execution

For debugging or custom workflows, run individual pipeline stages:

```bash
# Standardize prediction XDMFs (mesh extraction, field fixing)
python -m postprocess.postprocess_xdmf postprocess/config.my_model.json

# Extract sensor signals at all cases
python -m postprocess.extract_sensors postprocess/config.my_model.json

# Compute RMSE per case and cumulative metrics
python -m postprocess.compute_errors postprocess/config.my_model.json

# Integrate surface forces on cylinder
python -m postprocess.compute_forces postprocess/config.my_model.json

# Generate all figures
python -m postprocess.plot_results postprocess/config.my_model.json
```

---

## Advanced: Batch Processing (Local Machine)

Process multiple models sequentially on your local machine or login node:

```bash
bash batch_postprocess.sh \
  postprocess/config.model_A.json \
  postprocess/config.model_B.json \
  postprocess/config.model_C.json
```

The script validates all configs, then processes them sequentially with progress reporting and error handling.

---

## Advanced: Batch Processing (SLURM Cluster)

For HPC environments, submit batch processing as a single SLURM job:

```bash
bash postprocess/submit_postprocess.slurm.sh \
  postprocess/config.model_A.json \
  postprocess/config.model_B.json \
  postprocess/config.model_C.json
```

The script will automatically submit to SLURM and process all configs sequentially.

**Customize SLURM settings** by editing the `#SBATCH` directives in `postprocess/submit_postprocess.slurm.sh`:

```bash
#SBATCH --nodes=1              # Number of nodes (1 for serial postprocessing)
#SBATCH --ntasks=64            # Total tasks (increase for parallel I/O)
#SBATCH --time=12:00:00        # Adjust wall-clock limit (HH:MM:SS) based on data size
#SBATCH --partition=MAIN       # Partition name at your facility
#SBATCH --qos=calcul           # QoS tier (quality of service)
```

Monitor job progress:

```bash
tail -f slurm_postprocess_<jobid>.log
```

---

## Advanced: Recompute Only Errors or Forces

If you want to change force sign or RMSE metric without re-standardizing XDMFs:

```bash
# Recompute just errors (keep existing sensors/)
python -m postprocess.compute_errors postprocess/config.my_model.json
python -m postprocess.plot_results postprocess/config.my_model.json

# Recompute just forces (keep existing errors/)
python -m postprocess.compute_forces postprocess/config.my_model.json
python -m postprocess.plot_results postprocess/config.my_model.json
```

**Note:** Force cache checks only filename existence, not rollout_steps validity. If you change `rollout_steps`, manually delete the stale `forces/forces_*.csv` files:

```bash
rm results/model_name/forces/forces_*.csv
```

Then recompute forces.

---

## Quick Command Cookbook

## Config Templates

### Minimal Single-Model Template

Copy and adapt as `postprocess/config.my_run.json`:

```json
{
  "pred_dir": "/abs/path/to/prediction/cases",
  "dataset_dir": "/abs/path/to/dataset/predict",
  "configs_pool": "/abs/path/to/configs_pool.pkl",
  "model_name": "my_model_run",
  "model_shortname": "MyModel",
  "output_dir": "./results_full",
  "prediction_base_name": "graph_",
  "feature_map": {
    "x0": "velocity_x",
    "x1": "velocity_y",
    "x2": "pressure",
    "x3": "levelset",
    "x6": "nodetype"
  },
  "target_map": {
    "y0": "velocity_x",
    "y1": "velocity_y",
    "y2": "pressure"
  },
  "target_step_offset": 1,
  "rollout_steps": null,
  "sensor_mode": "auto",
  "sensor_csv": "",
  "sensor_drop_last_column": false,
  "levelset_source": "prediction",
  "force_mu": 0.001,
  "force_workers": 4,
  "force_sign_factor": -1.0
}
```

**Quick Notes:**
- `model_name`: Output folder name under `output_dir`
- `model_shortname`: Short label for legends (single-model and comparison plots)
- `rollout_steps`: null = use full length; integer = cap time-series extraction
- `sensor_drop_last_column`: true → 3×2 layout (6 panels) instead of 3×3 (9 panels)
- `force_sign_factor`: -1.0 to ensure Fx > 0 (left-to-right positive)
- Case ordering in plots: sorted by cylinder diameter from `configs_pool` (`radius_objects * 2` when available)

### Comparison-Only Template

Use this when model outputs are already postprocessed and you only want to generate comparison figures without rerunning the pipeline.

**Prerequisites:**
- Model output folders (each containing `sensors/`, `errors/`, `forces/` subdirectories)
- A config file with at least `configs_pool` path (you can reuse any model's config)
- Optional: custom nicknames for legend labels

**Command:**

```bash
bash run_postprocess.sh --compare postprocess/config.any_model.json \
  --models \
    ./results_full/model_A \
    ./results_full/model_B \
    ./results_full/model_C \
  --nicknames "Model A" "Model B" "Model C" \
  --output_dir ./results_full/comparison_model_A_B_C
```

**Direct Python equivalent:**

```bash
python -m postprocess.compare_models postprocess/config.any_model.json \
  --models ./results_full/model_A ./results_full/model_B ./results_full/model_C \
  --nicknames "Model A" "Model B" "Model C" \
  --output_dir ./results_full/comparison_model_A_B_C
```

---

## Output Layout

- `xdmf/`: standardized postprocessed XDMF files
- `sensors/`: extracted sensor signals + node mapping caches
- `errors/`: RMSE tables and cumulative curves
- `forces/`: per-case force CSV + summary
- `figures/`: publication figures (`png` and `pdf`)

## Assumptions To Verify Against Data

1. Prediction files expose `x0,x1,x2` for `(vx, vy, p)` and `y0,y1,y2` for target `(vx, vy, p)`.
2. `feature_map` includes the highest-index `x*` key as nodetype.
3. If `levelset_source="dataset"`, truth files in `dataset_dir` contain `LevelSetObject`.
4. Cases are matched by trailing integer case id in filenames.
5. Mesh topology is triangular 2D for force integration.
6. `configs_pool.pkl` contains per-case geometry and Reynolds columns (`Config`, `x_objects`, `y_objects`, `diameter`, `Re` or equivalents).

## Legacy Reference

Implementation and visual conventions were refactored from the legacy scripts in `../scripts/`:

- `error_gnn.py`
- `postprocess_gnn.py`
- `forces.py`
- `meshio_mesh.py`
- `trajectory.py`
- `nodetype.py`
