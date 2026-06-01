# Error Metrics

This page documents the metrics implemented by the active root-level pipeline and the older compatibility modules.

## Active RMSE Pipeline

Implemented in [postprocess/compute_errors.py](../postprocess/compute_errors.py).

| Metric | Definition | Code behavior | Output |
|---|---|---|---|
| Per-timestep RMSE | $$\mathrm{RMSE}_t = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(p_{t,i}-q_{t,i})^2}$$ | `_rmse(pred, targ)` flattens the spatial dimensions and computes the mean over axis 1 | `per_case_rmse.csv`, `cumulative_rmse_<case>.csv` |
| Velocity RMSE | Same definition applied to flattened `(vx, vy)` arrays | `rmse_vel = _rmse(vpred.reshape(T, -1), vtarg.reshape(T, -1))` | Saved in per-case and cumulative CSVs |
| Pressure RMSE | Same definition applied to scalar pressure arrays | `rmse_pres = _rmse(ppred, ptarg)` | Saved in per-case and cumulative CSVs |
| Total RMSE | Same definition applied to concatenated `(vx, vy, p)` arrays | `rmse_total = _rmse(total_pred.reshape(T, -1), total_targ.reshape(T, -1))` | Saved in per-case and cumulative CSVs |
| Cumulative RMSE | $$\mathrm{CumRMSE}_t = \sum_{\tau=0}^{t} \mathrm{RMSE}_\tau$$ | `_cumulative_sum(values)` uses `np.cumsum` | `cumulative_rmse_<case>.csv`, `cumulative_rmse_mean.csv` |
| Rollout RMSE | $$\mathrm{RolloutRMSE}_t = \frac{\mathrm{CumRMSE}_t}{\max(t,1)}$$ | `_cum_rmse_horizon` and `_rollout_from_cumulative` use a denominator clipped at 1 | `cumulative_rmse_<case>.csv`, `cumulative_rmse_mean.csv` |

## Summary Statistics Written by the Active Pipeline

`compute_errors.py` writes these aggregated columns:

| Column | Meaning |
|---|---|
| `rmse_step1` | Mean total RMSE over the first timestep only |
| `rmse_50step` | Mean total RMSE over the first 50 timesteps, or the full rollout if shorter |
| `rmse_all` | Mean total RMSE over the full rollout |
| `rmse_total_mean` | Mean of per-timestep total RMSE across the rollout |
| `rmse_total_std` | Standard deviation of per-timestep total RMSE across the rollout |
| `rmse_vel_mean` | Mean of per-timestep velocity RMSE |
| `rmse_pres_mean` | Mean of per-timestep pressure RMSE |

`summary_statistics.csv` stores case-count, mean, and standard deviation of the headline metrics across cases.

## Sensor Error Metrics in the Legacy Pipeline

The older [postprocess/metrics/compute_errors.py](../postprocess/metrics/compute_errors.py) supports sensor-level cumulative error.

| Metric | Definition | Notes |
|---|---|---|
| `AE` | $$|q - p|$$ | Absolute error, accumulated over time with `np.cumsum` |
| `SE` | $$(q - p)^2$$ | Squared error, accumulated over time with `np.cumsum` |

The legacy code computes per-sensor cumulative curves, then aggregates mean and standard deviation across sensors and cases. That package also writes `sensor_error_summary.csv`, `sensor_errors/*.json`, and `*_cum_error.npz` files.

## Force Metrics

The active force pipeline does not normalize to coefficients. It integrates raw traction on the obstacle boundary and writes `fx_pred`, `fy_pred`, `fx_targ`, and `fy_targ`.

The legacy [postprocess/metrics/compute_forces.py](../postprocess/metrics/compute_forces.py) does normalize to coefficients:

$$
C_D = \frac{F_D}{\tfrac{1}{2}\rho U_\infty^2 D}, \qquad
C_L = \frac{F_L}{\tfrac{1}{2}\rho U_\infty^2 D}
$$

It uses `mu`, `rho`, `U_inf`, and `D` from the `physical_parameters` section.

## Caveats

1. The active RMSE code measures error over nodes, not over mesh elements.
2. The denominator for rollout RMSE is clipped at 1, so the first point is not divided by zero.
3. `rmse_total_std` in `per_case_rmse.csv` is the standard deviation across timesteps, not across nodes.
4. The force metrics depend on the exact `nodetype` obstacle mask and levelset orientation.

---
Prev: [Configuration](Configuration.md) | Next: [XDMF-HDF5 Postprocessing](XDMF-HDF5-Postprocessing.md)
