# Configuration

The active pipeline reads a compact JSON schema. The same parameter names appear in the `configs/*.json` files and in `postprocess/config.example.json`.

## Active Schema

| Key | Type | Default / Typical Value | Used by | Effect / Failure Mode |
|---|---|---|---|---|
| `pred_dir` | string | required | `postprocess_xdmf.py`, `extract_sensors.py`, `compute_errors.py`, `compute_forces.py` | Folder containing raw prediction XDMFs. If wrong, case discovery returns nothing. |
| `dataset_dir` | string | required | `postprocess_xdmf.py` | Truth dataset folder used when `levelset_source` is `dataset` or when the prediction lacks a levelset field. |
| `configs_pool` | string | required | `extract_sensors.py`, `plot_results.py`, `compare_models.py` | Pickle with per-case geometry and Reynolds metadata. If wrong, sensor placement and Reynolds sorting fail. |
| `model_name` | string | required | all active stages | Output subdirectory name under `output_dir`. |
| `model_shortname` | string | optional | `plot_results.py` | Legend label for figures; falls back to `nickname` or `model_name`. |
| `output_dir` | string | required | all active stages | Root results directory. Each stage writes under `output_dir/model_name`. |
| `prediction_base_name` | string | usually `graph_` | `postprocess_xdmf.py` | File prefix used to discover raw prediction XDMFs. Missing prefix means no cases are found. |
| `feature_map` | object | varies by model family | `postprocess_xdmf.py`, `compute_forces.py` | Maps `x*` names to semantics such as `velocity_x`, `velocity_y`, `pressure`, `levelset`, `nodetype`. If no `x` entry exists, the active pipeline cannot align fields. |
| `target_map` | object | varies by model family | documentation only in active pipeline | Maps `y*` names to semantics. The active root-level stages do not use it directly, but the configs keep it for clarity and compatibility. |
| `target_step_offset` | integer | `1` for rollouts, `0` for 1-step | `postprocess_xdmf.py` | Aligns prediction timestep `t` with target timestep `t - target_step_offset`. If too large, `postprocess_xdmf.py` writes zero cases and raises. |
| `rollout_steps` | integer or `null` | `600` or `null` | `postprocess_xdmf.py` | Crops the aligned rollout length before writing output XDMFs. |
| `sensor_mode` | string | `auto` | `extract_sensors.py` | `auto` uses `configs_pool` geometry; `csv` uses `sensor_csv`. Any other value falls back to `auto` behavior. |
| `sensor_csv` | string | empty string or a path | `extract_sensors.py` | CSV file with `sensor_id`, `x`, and `y` columns. Required when `sensor_mode` is `csv`. |
| `sensor_drop_last_column` | boolean | `false` | `plot_results.py`, `compare_models.py` | Switches the sensor grid from 3x3 to 3x2. The plotting code also changes the x-axis range to start at 300. |
| `levelset_source` | string | `prediction` or `dataset` | `postprocess_xdmf.py` | Chooses whether `levelset` is copied from the prediction or injected from truth data. If `prediction` is requested but missing, the code falls back to dataset levelset. |
| `force_mu` | float | `0.001`, `0.01`, or `0.0001` | `compute_forces.py` | Dynamic viscosity used in viscous traction. Wrong values scale forces incorrectly. |
| `force_workers` | integer | `4` | `compute_forces.py` | Thread count for force time-series computation. |
| `force_sign_factor` | float | `-1.0` | `compute_forces.py` | Multiplies all force components before writing CSVs. The active code uses this to match a body-force sign convention. |
| `force_summary_start` | integer | `100` | `compute_forces.py` | First timestep included in the summary statistics. If missing, the code defaults to 100. |
| `nickname` | string | optional | `plot_results.py`, `compare_models.py` | Optional display label; if absent the code uses `model_shortname` or `model_name`. |

## Common Model Families

| Family | Files | Notes |
|---|---|---|
| Re2 1-step | `configs/config.re2_1step_VPLN.json`, `configs/config.re2_1step_VPLN_ld.json`, `configs/config.re2_1step_VPN.json` | `target_step_offset = 0`, `rollout_steps = 1`, `output_dir = ./results_ReX_1step` |
| Re3 1-step | `configs/config.re3_1step_VPLN.json`, `configs/config.re3_1step_VPLN_ld.json`, `configs/config.re3_1step_VPN.json` | Same layout as Re2 1-step, different paths and `force_mu = 0.001` |
| Re4 1-step | `configs/config.re4_1step_VPLN.json`, `configs/config.re4_1step_VPLN_ld.json`, `configs/config.re4_1step_VPN.json` | Same layout as Re2 1-step, different paths and `force_mu = 0.0001` |
| Re2 trunc all-noise | `configs/config.re2trunc_allnoise_vpln_l.json`, `configs/config.re2trunc_allnoise_vpln_ld.json`, `configs/config.re2trunc_allnoise_vpn_l.json` | `target_step_offset = 1`, `rollout_steps = 600`, `output_dir = ./results_ReX_allnoise` |
| Re3 trunc all-noise | `configs/config.re3trunc_allnoise_vpln_l.json`, `configs/config.re3trunc_allnoise_vpln_ld.json`, `configs/config.re3trunc_allnoise_vpn_l.json` | Same layout as Re2 trunc all-noise, different paths and `force_mu = 0.001` |
| Re4 trunc all-noise | `configs/config.re4trunc_allnoise_vpln_l.json`, `configs/config.re4trunc_allnoise_vpln_ld.json`, `configs/config.re4trunc_allnoise_vpn_l.json` | Same layout as Re2 trunc all-noise, different paths and `force_mu = 0.0001` |
| Re2/Re3/Re4 init-iter all-noise | `configs/config.re2_allnoise_vpln_ld_init-iter.json`, `configs/config.re3_allnoise_vpln_ld_init-iter.json`, `configs/config.re4_allnoise_vpln_ld_init-iter.json` | Output goes to `./results_ReX_allnoise_full` and these configs are used for trunc-vs-init comparisons |

## Concrete Example

The file `configs/config.re2trunc_allnoise_vpln_l.json` uses:

| Key | Value |
|---|---|
| `model_name` | `onecyl_Re2trunc_allnoise_VPLN_l` |
| `model_shortname` | `Re2 VPLN-l` |
| `output_dir` | `./results_ReX_allnoise` |
| `prediction_base_name` | `graph_` |
| `target_step_offset` | `1` |
| `rollout_steps` | `600` |
| `sensor_mode` | `auto` |
| `levelset_source` | `prediction` |
| `force_mu` | `0.01` |

If you change `feature_map` or `levelset_source` incorrectly, downstream force computation and sensor extraction are the first stages to fail.

## Legacy Nested Schema

The file `postprocess/Re2_onecyl_VPLN_l.json` uses a different structure:

| Section | Keys | Effect |
|---|---|---|
| `dataset_parameters` | `dt`, `trajectory_length`, `prediction_folder`, `path_to_configs_pool` | Legacy dataset and discovery settings |
| `model_parameters` | `name`, `final_base_name`, `prediction_start_index` | Legacy model selection and alignment |
| `plot_parameters` | `truth_prediction_pairs`, `points_choice` | Legacy truth/pred pairing and sensor selection |
| `feature_map` | `velocity_x`, `velocity_y`, `pressure`, `levelset`, `nodetype` | Legacy semantic mapping |
| `physical_parameters` | `mu`, `rho`, `U_inf`, `D` | Legacy force normalization inputs |

That schema is documented here because the files are still present, but the active root-level scripts do not consume it.

---
Prev: [Repository Structure](Repository-Structure.md) | Next: [Error Metrics](Error-Metrics.md)
