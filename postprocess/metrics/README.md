# postprocess/metrics

Compute error statistics and custom metrics from GNN predictions.

## Intended scope

- Per-timestep RMSE, MAE, relative L2 error for velocity and pressure fields.
- Spatial error maps (per-node error at selected timesteps).
- Aggregated statistics across trajectories (mean, std, percentiles).
- Drag/lift coefficient computation from predicted fields.
- Comparison tables: predicted vs. ground-truth integral quantities.

## Conventions

- All scripts should accept a prediction directory (containing `.xdmf` files
  produced by `graphphysics.predict`) and a ground-truth directory.
- Output CSV tables and/or numpy `.npz` files to a `results/` directory.
