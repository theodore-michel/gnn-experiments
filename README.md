# GNN CFD Postprocessing Repository

This repository is dedicated to postprocessing Graph Neural Network predictions for cylinder-flow CFD rollouts.

## Environment

Use the `graph` conda environment:

```bash
conda activate graph
```

## Pipeline Scripts

All scripts consume the same unified JSON config (`postprocess/config.example.json`):

1. `postprocess/postprocess_xdmf.py`
2. `postprocess/extract_sensors.py`
3. `postprocess/compute_errors.py`
4. `postprocess/compute_forces.py`
5. `postprocess/plot_results.py`
6. `postprocess/compare_models.py`

## End-to-End Execution

From repository root:

```bash
bash run_postprocess.sh postprocess/config.example.json
```

This runs:

`postprocess_xdmf -> extract_sensors -> compute_errors -> compute_forces -> plot_results`

## Replot Without Recompute

```bash
bash run_postprocess.sh --plots-only postprocess/config.example.json
```

## Multi-Model Comparison

```bash
bash run_postprocess.sh --compare postprocess/config.example.json \
  --models /path/to/results/model_A /path/to/results/model_B /path/to/results/model_C \
  --nicknames "Model A" "Model B" "Model C" \
  --output_dir ./results/comparison
```

## Output Layout

Each model writes to `output_dir/model_name/`:

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
