"""
compute_errors.py — RMSE and cumulative-error computation for GNN predictions.

Computes per-timestep RMSE (over all mesh nodes) for each field and case,
cumulative RMSE, and optional sensor-level error metrics.  Outputs CSV files
suitable for downstream plotting by ``visualization/plot_results.py``.

Usage (standalone)
------------------
::

    python -m postprocess.metrics.compute_errors \\
        -p config.json -d ./results --sensor-errors

Subcommands
~~~~~~~~~~~
* **compute** — run error computation from XDMFs.
* **plot**    — (delegated to plot_results.py; kept here for backward compat.)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Local imports — use relative when invoked as package, absolute otherwise
try:
    from ..utils.xdmf_io import (
        create_auto_sensor_location,
        extract_point_values_multi,
        gather_cases,
        load_configs_pool,
        load_json,
        save_sensor_data,
        xdmf_to_meshes,
    )
except ImportError:
    from postprocess.utils.xdmf_io import (
        create_auto_sensor_location,
        extract_point_values_multi,
        gather_cases,
        load_configs_pool,
        load_json,
        save_sensor_data,
        xdmf_to_meshes,
    )


# ============================================================================
# Core RMSE helpers
# ============================================================================


def _resolve_model_folder(pred_folder: str, model_name: str) -> str:
    """Resolve model folder for both layouts:
    1) <prediction_folder>/<model_name>/... and 2) <prediction_folder>/... (already model-specific).
    """
    nested = os.path.join(pred_folder, model_name)
    return nested if os.path.isdir(nested) else pred_folder


def _discover_cases(
    case_folder: str,
    preferred_base_name: str,
    fallback_base_name: Optional[str] = None,
) -> Tuple[Dict[str, str], str]:
    """Discover XDMF cases with robust base-name fallback.

    Tries preferred base name first, then fallback base name, then any ``*.xdmf``.
    Returns ``(cases, used_base_name)``.
    """
    cases = gather_cases(case_folder, preferred_base_name)
    if cases:
        return cases, preferred_base_name

    if fallback_base_name and fallback_base_name != preferred_base_name:
        cases = gather_cases(case_folder, fallback_base_name)
        if cases:
            return cases, fallback_base_name

    any_xdmf = sorted(Path(case_folder).glob("*.xdmf"))
    if any_xdmf:
        # Recover case ids from filename stem directly
        recovered = {p.stem: str(p) for p in any_xdmf}
        return recovered, "<auto:*>"

    return {}, preferred_base_name


def _to_physical_time(timesteps: np.ndarray, dt: float) -> np.ndarray:
    """Convert XDMF time values to physical time when they are index-like.

    If timesteps look like integer indices (0,1,2,...) we scale by ``dt``.
    If timesteps are missing / invalid (non-increasing), we rebuild ``0..T-1`` and scale by ``dt``.
    """
    ts = np.asarray(timesteps, dtype=float)
    if ts.size == 0:
        return ts

    diffs = np.diff(ts)
    if diffs.size and np.any(diffs <= 0):
        return np.arange(ts.size, dtype=float) * float(dt)

    is_integer_like = np.allclose(ts, np.round(ts), atol=1e-10)
    if is_integer_like and float(dt) != 1.0:
        return ts * float(dt)
    return ts


def _default_shift_for_pair(truth_field: str, pred_field: str) -> int:
    """Infer default truth shift for legacy x/y naming.

    In rollout dumps: x* at step t aligns with y* at step t-1.
    """
    if truth_field.startswith("y") and pred_field.startswith("x"):
        return 1
    return 0


def _align_pred_truth_arrays(
    pred_arr: np.ndarray,
    truth_arr: np.ndarray,
    shift_steps: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Align prediction and truth arrays under a truth lead of ``shift_steps``.

    Returns ``(pred_aligned, truth_aligned, offset_steps)`` where offset_steps
    is the number of leading prediction steps dropped.
    """
    if shift_steps <= 0:
        n = min(len(pred_arr), len(truth_arr))
        return pred_arr[:n], truth_arr[:n], 0

    if len(pred_arr) <= shift_steps or len(truth_arr) <= shift_steps:
        return pred_arr[:0], truth_arr[:0], shift_steps

    pred_aligned = pred_arr[shift_steps:]
    truth_aligned = truth_arr[:-shift_steps]
    n = min(len(pred_aligned), len(truth_aligned))
    return pred_aligned[:n], truth_aligned[:n], shift_steps


def rmse_per_timestep(
    pred: np.ndarray,
    targ: np.ndarray,
) -> np.ndarray:
    """Compute RMSE between *pred* and *targ* at each timestep.

    Parameters
    ----------
    pred, targ : np.ndarray
        Shape ``(T, N)`` — *T* timesteps, *N* mesh nodes.

    Returns
    -------
    np.ndarray
        Shape ``(T,)`` with ``RMSE[t] = sqrt(mean((pred[t] - targ[t])^2))``.
    """
    return np.sqrt(np.mean((pred - targ) ** 2, axis=1))


def cumulative_rmse(rmse_arr: np.ndarray) -> np.ndarray:
    """Cumulative sum of per-timestep RMSE."""
    return np.cumsum(rmse_arr)


# ============================================================================
# Full-mesh error computation
# ============================================================================


def compute_case_rmse(
    meshes: list,
    truth_field: str,
    pred_field: str,
    start_index: int = 0,
    shift_steps: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-timestep and cumulative RMSE for a single case.

    The *truth* and *pred* fields are read from ``mesh.point_data`` across
    the timestep range ``[start_index, …]``.

    Returns
    -------
    (rmse_per_step, cum_rmse) — each of shape ``(T - start_index,)``.
    """
    pred_vals = []
    targ_vals = []
    for mesh in meshes[start_index:]:
        if truth_field not in mesh.point_data:
            raise KeyError(f"Field '{truth_field}' missing from mesh point_data")
        if pred_field not in mesh.point_data:
            raise KeyError(f"Field '{pred_field}' missing from mesh point_data")
        targ_vals.append(mesh.point_data[truth_field])
        pred_vals.append(mesh.point_data[pred_field])

    pred_arr = np.array(pred_vals, dtype=np.float64)
    targ_arr = np.array(targ_vals, dtype=np.float64)
    pred_arr, targ_arr, _ = _align_pred_truth_arrays(pred_arr, targ_arr, shift_steps)
    if len(pred_arr) == 0:
        return np.array([]), np.array([])

    rmse = rmse_per_timestep(pred_arr, targ_arr)
    return rmse, cumulative_rmse(rmse)


def compute_cumulative_rmse_over_cases(
    model_folder: str,
    case_base_name: str,
    truth_field: str,
    pred_field: str,
    start_index: int = 0,
    shift_steps: int = 0,
    dt: float = 1.0,
) -> pd.DataFrame:
    """Compute per-timestep and cumulative RMSE for every case in *model_folder*.

    Returns a DataFrame with columns:
    ``case, timestep, time, rmse, cum_rmse``.
    """
    cases = gather_cases(model_folder, case_base_name)
    rows = []
    for case_id, xdmf_path in tqdm(cases.items(), desc="RMSE over cases"):
        meshes, timesteps = xdmf_to_meshes(xdmf_path)
        timesteps = _to_physical_time(timesteps, dt=dt)
        rmse_arr, cum_arr = compute_case_rmse(
            meshes,
            truth_field,
            pred_field,
            start_index=start_index,
            shift_steps=shift_steps,
        )
        ts = timesteps[start_index:]
        if shift_steps > 0 and len(ts) > shift_steps:
            ts = ts[shift_steps:]
        ts = ts[: len(rmse_arr)]
        for k, (r, c, t) in enumerate(zip(rmse_arr, cum_arr, ts)):
            rows.append(
                {
                    "case": case_id,
                    "rollout_step": k,
                    "timestep": k,
                    "time": float(t),
                    "rmse": float(r),
                    "cum_rmse": float(c),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["case", "rollout_step", "timestep", "time", "rmse", "cum_rmse"]
        )
    return pd.DataFrame(rows)


def summarize_over_models(
    model_dfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute mean / std RMSE across cases for each model at every timestep.

    Parameters
    ----------
    model_dfs : dict
        ``{model_name: DataFrame}`` as returned by
        :func:`compute_cumulative_rmse_over_cases`.

    Returns
    -------
    DataFrame
        Columns: ``model, timestep, time, rmse_mean, rmse_std,
        cum_rmse_mean, cum_rmse_std``.
    """
    rows = []
    for model_name, df in model_dfs.items():
        if df.empty:
            continue
        grouped = (
            df.groupby("rollout_step")
            .agg(
                time=("time", "first"),
                rmse_mean=("rmse", "mean"),
                rmse_std=("rmse", "std"),
                cum_rmse_mean=("cum_rmse", "mean"),
                cum_rmse_std=("cum_rmse", "std"),
            )
            .reset_index()
        )
        grouped["timestep"] = grouped["rollout_step"]
        grouped["model"] = model_name
        rows.append(grouped)
    if not rows:
        raise ValueError(
            "No RMSE rows were produced. Check prediction folder path and file base name "
            "(e.g. graph_ vs pred_) in the config."
        )
    return pd.concat(rows, ignore_index=True)


# ============================================================================
# Sensor-level error computation
# ============================================================================


def compute_sensor_cumulated_error(
    sensor_data: Dict[str, Any],
    model_name: str,
    truth_field: str,
    pred_field: str,
    metric: str = "AE",
    shift_steps: int = 0,
) -> Dict[str, np.ndarray]:
    """Compute cumulated pointwise error at each sensor for one model.

    Parameters
    ----------
    sensor_data : dict
        ``{sensor_name: {model_name: {field_name: [values]}}}``.
    model_name : str
    truth_field, pred_field : str
    metric : str
        ``"AE"`` for absolute error, ``"SE"`` for squared error.

    Returns
    -------
    dict
        ``{sensor_name: cum_error_array}``.
    """
    result: Dict[str, np.ndarray] = {}
    for sensor_name, model_dict in sensor_data.items():
        truth = np.asarray(model_dict[model_name][truth_field], dtype=np.float64)
        pred = np.asarray(model_dict[model_name][pred_field], dtype=np.float64)
        pred, truth, _ = _align_pred_truth_arrays(pred, truth, shift_steps)
        if len(pred) == 0:
            continue
        if metric == "AE":
            err = np.abs(truth - pred)
        elif metric == "SE":
            err = (truth - pred) ** 2
        else:
            raise ValueError(f"metric must be 'AE' or 'SE', got '{metric}'")
        result[sensor_name] = np.cumsum(err)
    return result


def case_cumulated_error_stats(
    sensor_cum_errors: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and std of cumulated error across sensors for one case.

    Returns (mean_array, std_array).
    """
    stacked = np.array(list(sensor_cum_errors.values()))
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def model_cumulated_error_stats(
    case_means: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and std of cumulated error across cases for one model.

    Returns (mean_array, std_array).
    """
    stacked = np.array(list(case_means.values()))
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)


# ============================================================================
# End-to-end computation from config
# ============================================================================


def run_error_computation(config: Dict[str, Any], output_dir: str) -> None:
    """Run full error computation pipeline driven by a JSON config.

    Produces:
    * ``rmse_per_case/<model>.csv``   — per-case per-timestep RMSE
    * ``rmse_summary.csv``            — mean/std across cases for all models
    * ``sensor_errors/<model>_<case>.json`` — (optional) sensor-level cum. error
    """
    dataset_params = config["dataset_parameters"]
    model_params = config["model_parameters"]
    model_names = (
        model_params["name"]
        if isinstance(model_params["name"], list)
        else [model_params["name"]]
    )
    base_name = model_params["final_base_name"]
    fallback_base = dataset_params.get("prediction_base_name")
    pred_folder = dataset_params["prediction_folder"]
    dt = dataset_params.get("dt", 1.0)
    start_idx = model_params.get("prediction_start_index", 0)

    # Truth / pred field pairs
    plot_params = config.get("plot_parameters", {})
    tp_pairs = plot_params.get(
        "truth_prediction_pairs",
        config.get("truth_prediction_pairs", {"V_vect_targ": "V_vect_pred"}),
    )

    os.makedirs(output_dir, exist_ok=True)
    per_case_dir = os.path.join(output_dir, "rmse_per_case")
    os.makedirs(per_case_dir, exist_ok=True)

    model_dfs: Dict[str, pd.DataFrame] = {}

    for model_name in model_names:
        model_folder = _resolve_model_folder(pred_folder, model_name)
        cases_probe, used_base = _discover_cases(model_folder, base_name, fallback_base)
        if not cases_probe:
            raise ValueError(
                f"No .xdmf files found in '{model_folder}'. "
                f"Tried base names: '{base_name}'"
                + (f" and '{fallback_base}'" if fallback_base else "")
                + "."
            )

        all_case_rows: list = []

        for truth_field, pred_field in tp_pairs.items():
            shift_steps = model_params.get(
                "truth_shift_steps",
                _default_shift_for_pair(truth_field, pred_field),
            )
            df = compute_cumulative_rmse_over_cases(
                model_folder=model_folder,
                case_base_name=(base_name if used_base == "<auto:*>" else used_base),
                truth_field=truth_field,
                pred_field=pred_field,
                start_index=start_idx,
                shift_steps=shift_steps,
                dt=dt,
            )
            if used_base == "<auto:*>" and df.empty:
                # Auto-discovery path: compute over explicit cases (base-name free)
                rows = []
                for case_id, xdmf_path in tqdm(
                    cases_probe.items(), desc="RMSE over cases"
                ):
                    meshes, timesteps = xdmf_to_meshes(xdmf_path)
                    timesteps = _to_physical_time(timesteps, dt=dt)
                    rmse_arr, cum_arr = compute_case_rmse(
                        meshes,
                        truth_field,
                        pred_field,
                        start_index=start_idx,
                        shift_steps=shift_steps,
                    )
                    ts = timesteps[start_idx:]
                    if shift_steps > 0 and len(ts) > shift_steps:
                        ts = ts[shift_steps:]
                    ts = ts[: len(rmse_arr)]
                    for k, (r, c, t) in enumerate(zip(rmse_arr, cum_arr, ts)):
                        rows.append(
                            {
                                "case": case_id,
                                "rollout_step": k,
                                "timestep": k,
                                "time": float(t),
                                "rmse": float(r),
                                "cum_rmse": float(c),
                            }
                        )
                df = pd.DataFrame(rows)
            df["truth_field"] = truth_field
            df["pred_field"] = pred_field
            all_case_rows.append(df)

        model_df = pd.concat(all_case_rows, ignore_index=True)
        model_df.to_csv(os.path.join(per_case_dir, f"{model_name}.csv"), index=False)
        model_dfs[model_name] = model_df

    # Summary across models
    summary = summarize_over_models(model_dfs)
    summary.to_csv(os.path.join(output_dir, "rmse_summary.csv"), index=False)
    print(f"[compute_errors] Wrote rmse_summary.csv → {output_dir}")


def run_sensor_error_computation(
    config: Dict[str, Any],
    output_dir: str,
    metric: str = "AE",
    load_data: bool = False,
) -> None:
    """Run sensor-level cumulated error computation.

    Produces per-case JSON of sensor cumulated errors and a summary CSV.
    """
    dataset_params = config["dataset_parameters"]
    model_params = config["model_parameters"]
    model_names = (
        model_params["name"]
        if isinstance(model_params["name"], list)
        else [model_params["name"]]
    )
    base_name = model_params["final_base_name"]
    fallback_base = dataset_params.get("prediction_base_name")
    pred_folder = dataset_params["prediction_folder"]
    dt = dataset_params.get("dt", 1.0)

    plot_params = config.get("plot_parameters", {})
    tp_pairs = plot_params.get(
        "truth_prediction_pairs",
        config.get("truth_prediction_pairs", {"V_vect_targ": "V_vect_pred"}),
    )

    os.makedirs(output_dir, exist_ok=True)
    sensor_dir = os.path.join(output_dir, "sensor_errors")
    os.makedirs(sensor_dir, exist_ok=True)
    signal_dir = os.path.join(output_dir, "sensor_signals")
    os.makedirs(signal_dir, exist_ok=True)

    # Gather cases per model
    models_cases: Dict[str, Dict[str, str]] = {}
    for mn in model_names:
        model_folder = _resolve_model_folder(pred_folder, mn)
        cases, used_base = _discover_cases(model_folder, base_name, fallback_base)
        if not cases:
            raise ValueError(
                f"No .xdmf files found for model '{mn}' in '{model_folder}'. "
                f"Tried '{base_name}'"
                + (f" and '{fallback_base}'" if fallback_base else "")
                + "."
            )
        models_cases[mn] = cases

    case_names = sorted(next(iter(models_cases.values())).keys())
    for mn, model_case_dict in models_cases.items():
        if set(model_case_dict.keys()) != set(case_names):
            raise ValueError(
                f"Model '{mn}' does not have the same case ids as other models. "
                "Ensure all compared models contain the same XDMF cases."
            )

    # Determine sensor locations
    points_choice = plot_params.get("points_choice", "auto")
    if points_choice == "auto" and dataset_params.get("path_to_configs_pool"):
        configs_pool = load_configs_pool(dataset_params["path_to_configs_pool"])
    else:
        configs_pool = None

    # Accumulators
    all_model_last_errors: Dict[str, List[float]] = {mn: [] for mn in model_names}
    all_model_case_means: Dict[str, Dict[str, np.ndarray]] = {
        mn: {} for mn in model_names
    }

    for case_name in tqdm(case_names, desc="Sensor errors"):
        # Fields to extract
        fields = list(tp_pairs.keys()) + list(tp_pairs.values())

        # Load or extract sensor data
        cache_path = os.path.join(sensor_dir, f"sensor_data_points_{case_name}.json")
        if load_data and os.path.exists(cache_path):
            with open(cache_path) as fh:
                sensor_data = json.load(fh)
        else:
            model_meshes: Dict[str, list] = {}
            for mn in model_names:
                meshes, _ = xdmf_to_meshes(models_cases[mn][case_name])
                model_meshes[mn] = meshes

            # Determine sensors only when extraction is needed
            if points_choice == "auto":
                if configs_pool is not None:
                    config_df = configs_pool[configs_pool["Config"] == case_name]
                    try:
                        obj_center = [
                            config_df["x_objects"].values[0][0],
                            config_df["y_objects"].values[0][0],
                            0.0,
                        ]
                    except (KeyError, IndexError):
                        obj_center = [0, 1.5, 0]
                else:
                    pts = model_meshes[model_names[0]][0].points
                    cx = float(0.5 * (np.min(pts[:, 0]) + np.max(pts[:, 0])))
                    cy = float(0.5 * (np.min(pts[:, 1]) + np.max(pts[:, 1])))
                    obj_center = [cx, cy, 0.0]

                domain_dims = dataset_params.get("domain_dimensions")
                if domain_dims is None:
                    pts = model_meshes[model_names[0]][0].points
                    x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
                    y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
                    domain_dims = {
                        "x_min": x_min,
                        "y_min": y_min,
                        "dx": x_max - x_min,
                        "dy": y_max - y_min,
                    }

                sensors = create_auto_sensor_location(
                    domain_dim_dict=domain_dims,
                    num_sensors=9,
                    object_center=obj_center,
                )
            else:
                sensors = plot_params.get("user_point_choices", {})

            sensor_data = extract_point_values_multi(model_meshes, sensors, fields)
            save_sensor_data(sensor_data, sensor_dir, suffix=f"points_{case_name}")

        # Compute per-model cumulated error
        for mn in model_names:
            truth_field = next(iter(tp_pairs.keys()))
            pred_field = tp_pairs[truth_field]
            shift_steps = model_params.get(
                "truth_shift_steps",
                _default_shift_for_pair(truth_field, pred_field),
            )
            sensor_cum = compute_sensor_cumulated_error(
                sensor_data,
                mn,
                truth_field,
                pred_field,
                metric=metric,
                shift_steps=shift_steps,
            )
            if not sensor_cum:
                continue
            mean_arr, std_arr = case_cumulated_error_stats(sensor_cum)
            all_model_case_means[mn][case_name] = mean_arr
            all_model_last_errors[mn].append(float(mean_arr[-1]))

            # Save aligned truth-vs-pred sensor signals for plotting
            aligned_payload: Dict[str, Any] = {
                "model": mn,
                "case": case_name,
                "rollout_step": list(range(len(mean_arr))),
                "pairs": {},
            }
            for tf, pf in tp_pairs.items():
                aligned_payload["pairs"][f"{tf}|{pf}"] = {}
                for sensor_name, model_dict in sensor_data.items():
                    truth = np.asarray(model_dict[mn][tf], dtype=np.float64)
                    pred = np.asarray(model_dict[mn][pf], dtype=np.float64)
                    pred_al, truth_al, _ = _align_pred_truth_arrays(
                        pred, truth, shift_steps
                    )
                    n = min(len(pred_al), len(truth_al))
                    aligned_payload["pairs"][f"{tf}|{pf}"][sensor_name] = {
                        "truth": truth_al[:n].tolist(),
                        "pred": pred_al[:n].tolist(),
                    }

            # Additional velocity norm comparison (legacy-style expected output)
            if all(
                k in [p.split("|")[0] for p in aligned_payload["pairs"].keys()]
                for k in ["y0", "y1"]
            ):
                if all(
                    k in [p.split("|")[1] for p in aligned_payload["pairs"].keys()]
                    for k in ["x0", "x1"]
                ):
                    aligned_payload["pairs"]["Vnorm_targ|Vnorm_pred"] = {}
                    for sensor_name, model_dict in sensor_data.items():
                        truth_x = np.asarray(model_dict[mn]["y0"], dtype=np.float64)
                        truth_y = np.asarray(model_dict[mn]["y1"], dtype=np.float64)
                        pred_x = np.asarray(model_dict[mn]["x0"], dtype=np.float64)
                        pred_y = np.asarray(model_dict[mn]["x1"], dtype=np.float64)
                        pred_norm = np.sqrt(pred_x**2 + pred_y**2)
                        truth_norm = np.sqrt(truth_x**2 + truth_y**2)
                        pred_al, truth_al, _ = _align_pred_truth_arrays(
                            pred_norm, truth_norm, shift_steps
                        )
                        n = min(len(pred_al), len(truth_al))
                        aligned_payload["pairs"]["Vnorm_targ|Vnorm_pred"][
                            sensor_name
                        ] = {
                            "truth": truth_al[:n].tolist(),
                            "pred": pred_al[:n].tolist(),
                        }

            model_signal_dir = os.path.join(signal_dir, mn)
            os.makedirs(model_signal_dir, exist_ok=True)
            with open(os.path.join(model_signal_dir, f"{case_name}.json"), "w") as fh:
                json.dump(aligned_payload, fh, indent=2)

    # Summary CSV: final cumulated error per model per case
    summary_rows = []
    for mn in model_names:
        for cn, last_err in zip(case_names, all_model_last_errors[mn]):
            summary_rows.append({"model": mn, "case": cn, "final_cum_error": last_err})
        # Overall stats
        if not all_model_case_means[mn]:
            continue
        model_mean, model_std = model_cumulated_error_stats(all_model_case_means[mn])
        # Save model-level time-series
        np.savez(
            os.path.join(sensor_dir, f"{mn}_cum_error.npz"),
            mean=model_mean,
            std=model_std,
            dt=dt,
        )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, "sensor_error_summary.csv"), index=False
    )
    print(f"[compute_errors] Sensor error summary → {output_dir}")


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute RMSE / cumulated error for GNN predictions."
    )
    p.add_argument(
        "-p",
        "--parameters",
        required=True,
        help="JSON config file path.",
    )
    p.add_argument(
        "-d",
        "--directory",
        default="./error_results",
        help="Output directory for CSV / NPZ results.",
    )
    p.add_argument(
        "--sensor-errors",
        action="store_true",
        help="Also compute sensor-level cumulated errors.",
    )
    p.add_argument(
        "--metric",
        choices=["AE", "SE"],
        default="AE",
        help="Error metric for sensor computation (AE=absolute, SE=squared).",
    )
    p.add_argument(
        "--load-data",
        action="store_true",
        help="Re-use cached sensor JSON instead of re-extracting.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_json(args.parameters)

    run_error_computation(config, args.directory)

    if args.sensor_errors:
        run_sensor_error_computation(
            config, args.directory, metric=args.metric, load_data=args.load_data
        )


if __name__ == "__main__":
    main()
