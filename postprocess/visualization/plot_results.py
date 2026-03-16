"""
plot_results.py — Publication-quality plotting for GNN prediction analysis.

Provides figure types:

1. **Sensor time-series** — ground truth vs prediction at sensor locations.
2. **Line profiles** — field values along x/y-lines at selected timesteps.
3. **Cumulative RMSE** — mean ± std across cases for each model.
4. **Per-case RMSE bars** — bar chart of final RMSE per case per model.
5. **Drag / lift curves** — predicted vs truth force coefficients over time.
6. **Sensor cumulative error** — mean ± std across cases.

All functions accept a ``PLOT_CONFIG`` dict for consistent styling (font,
palette, line widths, etc.) and an ``article_style`` flag that strips
excessive annotations for camera-ready figures.

Can be used as a library or driven from the CLI:

::

    python -m postprocess.visualization.plot_results \\
        -d ./results --compare --article-style

"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend


# ============================================================================
# Default plot configuration
# ============================================================================

PLOT_CONFIG: Dict[str, Any] = {
    "font_family": "serif",
    "font_size": 16,
    "title_size": 18,
    "tick_size": 14,
    "legend_size": 13,
    "figsize": (13, 7),
    "dpi": 300,
    "line_width": 2.4,
    "truth_style": {
        "color": "black",
        "linestyle": "--",
        "linewidth": 2.0,
        "alpha": 0.9,
    },
    "pred_style": {
        "color": "#1f77b4",
        "linestyle": "-",
        "linewidth": 2.4,
        "alpha": 0.9,
    },
    "error_band_alpha": 0.2,
    "error_margin_alpha": 0.3,
    "palette": [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ],
    "save_formats": ["png"],
}


def _apply_style(cfg: Dict[str, Any]) -> None:
    """Set matplotlib rcParams from config."""
    plt.rcParams.update(
        {
            "font.family": cfg.get("font_family", "serif"),
            "font.size": cfg.get("font_size", 14),
            "axes.titlesize": cfg.get("title_size", 16),
            "axes.labelsize": cfg.get("font_size", 14),
            "xtick.labelsize": cfg.get("tick_size", 12),
            "ytick.labelsize": cfg.get("tick_size", 12),
            "legend.fontsize": cfg.get("legend_size", 12),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save_fig(
    fig: plt.Figure,
    output_dir: str,
    name: str,
    cfg: Dict[str, Any],
    article_style: bool = False,
) -> None:
    """Save figure in all configured formats."""
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_article" if article_style else ""
    for fmt in cfg.get("save_formats", ["png"]):
        path = os.path.join(output_dir, f"{name}{suffix}.{fmt}")
        fig.savefig(path, dpi=cfg.get("dpi", 300), bbox_inches="tight")
    plt.close(fig)


def _format_name(name: str) -> str:
    """Convert snake_case / dash-separated model name to display form."""
    return name.replace("_", " ").replace("-", " ").title()


def _normalize_case_id(case_id: str) -> str:
    """Normalize case ids to their trailing numeric token when present."""
    match = re.search(r"(\d+)$", str(case_id))
    return match.group(1) if match else str(case_id)


# ============================================================================
# 1. Sensor time-series plots
# ============================================================================


def plot_sensors(
    data_dict: Dict[str, Dict[str, List[float]]],
    truth_pred_pairs: Dict[str, str],
    output_dir: str,
    times: Optional[Union[np.ndarray, List[float]]] = None,
    model_name: str = "Model",
    cfg: Optional[Dict[str, Any]] = None,
    article_style: bool = False,
) -> None:
    """Plot ground truth vs prediction at sensor locations (single model).

    Parameters
    ----------
    data_dict : dict
        ``{sensor_name: {field: [values_over_time]}}``.
    truth_pred_pairs : dict
        ``{truth_field: pred_field}``.
    times : array-like | None
        Time stamps; defaults to integer indices.
    """
    cfg = cfg or PLOT_CONFIG
    _apply_style(cfg)

    num_sensors = len(data_dict)
    ncols = min(3, num_sensors)
    nrows = int(np.ceil(num_sensors / ncols))

    for truth_field, pred_field in truth_pred_pairs.items():
        fig, axs = plt.subplots(nrows, ncols, figsize=cfg["figsize"], squeeze=False)
        if not article_style:
            fig.suptitle(
                f"{_format_name(model_name)}: {truth_field} vs {pred_field}",
                fontsize=cfg["title_size"],
            )
        axs_flat = axs.flatten()

        for i, (sensor_name, sensor_data) in enumerate(data_dict.items()):
            if i >= len(axs_flat):
                break
            ax = axs_flat[i]
            truth_vals = np.asarray(sensor_data[truth_field])
            pred_vals = np.asarray(sensor_data[pred_field])
            t = np.asarray(times) if times is not None else np.arange(len(truth_vals))

            ax.plot(t, truth_vals, label="Ground Truth", **cfg["truth_style"])
            ax.plot(t, pred_vals, label="Prediction", **cfg["pred_style"])
            ax.set_title(
                f"${sensor_name[0]}_{{{sensor_name[1:]}}}$"
                if len(sensor_name) > 1
                else sensor_name
            )
            ax.set_xlabel("Rollout step")

            is_velocity = "V" in truth_field or "V" in pred_field
            ax.set_ylabel(r"Velocity (m s$^{-1}$)" if is_velocity else "Pressure (Pa)")
            if i == 0 and not article_style:
                ax.legend(loc="best")

        # Hide unused axes
        for j in range(i + 1, len(axs_flat)):
            axs_flat[j].set_visible(False)

        fig.tight_layout()
        _save_fig(
            fig, output_dir, f"sensors_{truth_field}_{pred_field}", cfg, article_style
        )


def plot_sensors_multi(
    data_dict: Dict[str, Dict[str, Dict[str, List[float]]]],
    truth_pred_pairs: Dict[str, str],
    output_dir: str,
    times: Optional[Dict[str, np.ndarray]] = None,
    comparison_criterion: str = "models",
    cfg: Optional[Dict[str, Any]] = None,
    article_style: bool = False,
) -> None:
    """Plot sensor time-series for multiple models overlaid.

    Parameters
    ----------
    data_dict : dict
        ``{sensor_name: {model_name: {field: [values]}}}``.
    times : dict | None
        ``{model_name: array}`` — per-model time axes.
    """
    cfg = cfg or PLOT_CONFIG
    _apply_style(cfg)
    palette = cfg.get("palette", PLOT_CONFIG["palette"])

    num_sensors = len(data_dict)
    ncols = min(3, num_sensors)
    nrows = int(np.ceil(num_sensors / ncols))

    for truth_field, pred_field in truth_pred_pairs.items():
        fig, axs = plt.subplots(nrows, ncols, figsize=cfg["figsize"], squeeze=False)
        if not article_style:
            fig.suptitle(
                f"Comparison ({comparison_criterion}): {truth_field} vs {pred_field}",
                fontsize=cfg["title_size"],
            )
        axs_flat = axs.flatten()

        model_names = None
        for i, (sensor_name, model_dict) in enumerate(data_dict.items()):
            if i >= len(axs_flat):
                break
            ax = axs_flat[i]
            if model_names is None:
                model_names = list(model_dict.keys())

            for j, mn in enumerate(model_names):
                vals = np.asarray(model_dict[mn][pred_field])
                t = (
                    np.asarray(times[mn])
                    if times and mn in times
                    else np.arange(len(vals))
                )
                ax.plot(
                    t,
                    vals,
                    color=palette[j % len(palette)],
                    linewidth=cfg["line_width"],
                    alpha=0.8,
                    label=_format_name(mn),
                )

            # Truth (from first model)
            first_model = model_names[0]
            truth_vals = np.asarray(model_dict[first_model][truth_field])
            t0 = (
                np.asarray(times[first_model])
                if times and first_model in times
                else np.arange(len(truth_vals))
            )
            ax.plot(t0, truth_vals, label="Ground Truth", **cfg["truth_style"])

            ax.set_title(
                f"${sensor_name[0]}_{{{sensor_name[1:]}}}$"
                if len(sensor_name) > 1
                else sensor_name
            )
            ax.set_xlabel("Rollout step")
            is_velocity = "V" in truth_field or "V" in pred_field
            ax.set_ylabel(r"Velocity (m s$^{-1}$)" if is_velocity else "Pressure (Pa)")
            if i == 0 and not article_style:
                ax.legend(loc="best", fontsize="small", ncol=2)

        for j in range(i + 1, len(axs_flat)):
            axs_flat[j].set_visible(False)

        fig.tight_layout()
        _save_fig(
            fig,
            output_dir,
            f"sensors_multi_{truth_field}_{pred_field}",
            cfg,
            article_style,
        )


# ============================================================================
# 2. Line profile plots
# ============================================================================


def plot_lines(
    data_dict: Dict[str, Dict[str, Any]],
    line_type: str,
    line_points_axis: np.ndarray,
    truth_pred_pairs: Dict[str, str],
    output_dir: str,
    model_name: str = "Model",
    case_name: str = "",
    auto_y_limits: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
    article_style: bool = False,
) -> None:
    """Plot field profiles along a line at selected timesteps (single model).

    Parameters
    ----------
    data_dict : dict
        ``{timestep_key: {field_name: [values_along_line]}}``.
    line_type : str
        ``"x"`` or ``"y"``.
    line_points_axis : array
        The varying axis coordinate values.
    """
    cfg = cfg or PLOT_CONFIG
    _apply_style(cfg)

    num_steps = len(data_dict)
    ncols = int(np.ceil(np.sqrt(num_steps)))
    nrows = int(np.ceil(num_steps / ncols))

    for truth_field, pred_field in truth_pred_pairs.items():
        fig, axs = plt.subplots(nrows, ncols, figsize=cfg["figsize"], squeeze=False)
        if not article_style:
            fig.suptitle(
                f"{_format_name(model_name)} case {case_name}: "
                f"{truth_field} vs {pred_field} along {line_type}-line",
                fontsize=cfg["title_size"],
            )
        axs_flat = axs.flatten()

        # Auto y-limits
        if auto_y_limits:
            all_truth = [
                v for td in data_dict.values() for v in td.get(truth_field, [0])
            ]
            y_min, y_max = min(all_truth), max(all_truth)
            margin = 0.1 * max(abs(y_min), abs(y_max), 1e-6)
            y_lim = (y_min - margin, y_max + margin)
        else:
            y_lim = None

        for i, (ts_key, ts_data) in enumerate(data_dict.items()):
            if i >= len(axs_flat):
                break
            ax = axs_flat[i]
            ax.plot(
                line_points_axis,
                ts_data[truth_field],
                label="Ground Truth",
                **cfg["truth_style"],
            )
            # 5 % error margin band
            truth_arr = np.asarray(ts_data[truth_field])
            margin_arr = 0.05 * np.abs(truth_arr)
            ax.fill_between(
                line_points_axis,
                truth_arr - margin_arr,
                truth_arr + margin_arr,
                color="gray",
                alpha=cfg["error_margin_alpha"],
                label=r"$\pm$5% margin",
            )
            ax.plot(
                line_points_axis,
                ts_data[pred_field],
                label="Prediction",
                **cfg["pred_style"],
            )
            ax.set_title(f"Step {ts_key}")
            ax.set_xlabel(rf"${line_type}$-axis")
            is_velocity = "V" in truth_field
            ax.set_ylabel(r"Velocity (m s$^{-1}$)" if is_velocity else "Pressure (Pa)")
            if y_lim:
                ax.set_ylim(y_lim)
            ax.set_xlim(line_points_axis[0], line_points_axis[-1])
            if i == 0 and not article_style:
                ax.legend(loc="best")

        for j in range(i + 1, len(axs_flat)):
            axs_flat[j].set_visible(False)

        fig.tight_layout()
        _save_fig(
            fig,
            output_dir,
            f"{truth_field}_{pred_field}_{line_type}line",
            cfg,
            article_style,
        )


def plot_lines_multi(
    data_dict: Dict[str, Dict[str, Dict[str, Any]]],
    line_type: str,
    line_points_axis: np.ndarray,
    truth_pred_pairs: Dict[str, str],
    output_dir: str,
    comparison_criterion: str = "models",
    auto_y_limits: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
    article_style: bool = False,
) -> None:
    """Plot line profiles for multiple models overlaid."""
    cfg = cfg or PLOT_CONFIG
    _apply_style(cfg)
    palette = cfg.get("palette", PLOT_CONFIG["palette"])

    num_steps = len(data_dict)
    ncols = int(np.ceil(np.sqrt(num_steps)))
    nrows = int(np.ceil(num_steps / ncols))

    for truth_field, pred_field in truth_pred_pairs.items():
        fig, axs = plt.subplots(nrows, ncols, figsize=cfg["figsize"], squeeze=False)
        if not article_style:
            fig.suptitle(
                f"Comparison ({comparison_criterion}): {truth_field} vs {pred_field} along {line_type}-line",
                fontsize=cfg["title_size"],
            )
        axs_flat = axs.flatten()

        for i, (ts_key, ts_models) in enumerate(data_dict.items()):
            if i >= len(axs_flat):
                break
            ax = axs_flat[i]
            model_names = list(ts_models.keys())

            # Truth from first model
            first_data = ts_models[model_names[0]]
            ax.plot(
                line_points_axis,
                first_data[truth_field],
                label="Ground Truth",
                **cfg["truth_style"],
            )
            truth_arr = np.asarray(first_data[truth_field])
            margin_arr = 0.05 * np.abs(truth_arr)
            ax.fill_between(
                line_points_axis,
                truth_arr - margin_arr,
                truth_arr + margin_arr,
                color="gray",
                alpha=cfg["error_margin_alpha"],
                label=r"$\pm$5% margin",
            )

            # Predictions per model
            for j, mn in enumerate(model_names):
                ax.plot(
                    line_points_axis,
                    ts_models[mn][pred_field],
                    color=palette[j % len(palette)],
                    linewidth=cfg["line_width"],
                    alpha=0.7,
                    label=_format_name(mn),
                )

            if article_style:
                ax.text(
                    0.5,
                    0.92,
                    f"Step {ts_key}",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=cfg["title_size"],
                )
            else:
                ax.set_title(f"Step {ts_key}")
            ax.set_xlabel(rf"${line_type}$-axis")
            is_velocity = "V" in truth_field
            ax.set_ylabel(r"Velocity (m s$^{-1}$)" if is_velocity else "Pressure (Pa)")
            ax.set_xlim(line_points_axis[0], line_points_axis[-1])
            if i == 0 and not article_style:
                ax.legend(loc="best", fontsize="small", ncol=2)

        for j in range(i + 1, len(axs_flat)):
            axs_flat[j].set_visible(False)

        fig.tight_layout()
        _save_fig(
            fig,
            output_dir,
            f"comparison_{comparison_criterion}_{truth_field}_{pred_field}_{line_type}line",
            cfg,
            article_style,
        )


# ============================================================================
# 3. Cumulative RMSE plots
# ============================================================================


def plot_cumulated_rmse(
    summary_csv: str,
    output_dir: str,
    cfg: Optional[Dict[str, Any]] = None,
    article_style: bool = False,
) -> None:
    """Plot cumulative RMSE (mean ± std) from the summary CSV.

    Expects columns: ``model, timestep, time, cum_rmse_mean, cum_rmse_std``.
    """
    cfg = cfg or PLOT_CONFIG
    _apply_style(cfg)
    palette = cfg.get("palette", PLOT_CONFIG["palette"])

    df = pd.read_csv(summary_csv)
    models = df["model"].unique()

    fig, ax = plt.subplots(figsize=cfg["figsize"])
    for i, model in enumerate(models):
        sub = df[df["model"] == model].sort_values("timestep")
        if "rollout_step" in sub.columns:
            t = sub["rollout_step"].values
        elif "timestep" in sub.columns:
            t = sub["timestep"].values
        else:
            t = np.arange(len(sub))
        mean = sub["cum_rmse_mean"].values
        std = sub["cum_rmse_std"].values
        c = palette[i % len(palette)]
        ax.plot(
            t, mean, color=c, linewidth=cfg["line_width"], label=_format_name(model)
        )
        ax.fill_between(
            t, mean - std, mean + std, color=c, alpha=cfg["error_band_alpha"]
        )

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Cumulative RMSE")
    if not article_style:
        ax.set_title("Cumulative RMSE across cases")
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_fig(fig, output_dir, "cumulated_rmse", cfg, article_style)


def plot_rmse_per_timestep(
    summary_csv: str,
    output_dir: str,
    cfg: Optional[Dict[str, Any]] = None,
    article_style: bool = False,
) -> None:
    """Plot per-timestep RMSE (mean ± std) from the summary CSV."""
    cfg = cfg or PLOT_CONFIG
    _apply_style(cfg)
    palette = cfg.get("palette", PLOT_CONFIG["palette"])

    df = pd.read_csv(summary_csv)
    models = df["model"].unique()

    fig, ax = plt.subplots(figsize=cfg["figsize"])
    for i, model in enumerate(models):
        sub = df[df["model"] == model].sort_values("timestep")
        if "rollout_step" in sub.columns:
            t = sub["rollout_step"].values
        elif "timestep" in sub.columns:
            t = sub["timestep"].values
        else:
            t = np.arange(len(sub))
        mean = sub["rmse_mean"].values
        std = sub["rmse_std"].values
        c = palette[i % len(palette)]
        ax.plot(
            t, mean, color=c, linewidth=cfg["line_width"], label=_format_name(model)
        )
        ax.fill_between(
            t, mean - std, mean + std, color=c, alpha=cfg["error_band_alpha"]
        )

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("RMSE")
    if not article_style:
        ax.set_title("Per-timestep RMSE across cases")
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_fig(fig, output_dir, "rmse_per_timestep", cfg, article_style)


# ============================================================================
# 4. Per-case RMSE bar chart
# ============================================================================


def plot_case_rmse_bars(
    per_case_dir: str,
    output_dir: str,
    cfg: Optional[Dict[str, Any]] = None,
    article_style: bool = False,
) -> None:
    """Bar chart of per-case final RMSE in legacy grouped-model style.

    Mimics scripts/error_gnn.py: each model gets one contiguous group of case bars,
    with a dotted horizontal average line over that group.
    """
    cfg = cfg or PLOT_CONFIG
    _apply_style(cfg)
    palette = cfg.get("palette", PLOT_CONFIG["palette"])

    csvs = sorted(glob.glob(os.path.join(per_case_dir, "*.csv")))
    if not csvs:
        print(f"[plot] No CSVs found in {per_case_dir}")
        return

    model_data: Dict[str, Dict[str, float]] = {}
    for csv_path in csvs:
        model_name = Path(csv_path).stem
        df = pd.read_csv(csv_path)
        final = df.groupby("case")["cum_rmse"].last()
        model_data[model_name] = final.to_dict()

    all_cases = sorted({c for d in model_data.values() for c in d})
    model_names = list(model_data.keys())
    n_cases = len(all_cases)
    bar_width = 1.0

    fig, ax = plt.subplots(figsize=(max(cfg["figsize"][0], 16), cfg["figsize"][1]))
    group_centers = []
    group_labels = []

    for i, mn in enumerate(model_names):
        vals = np.array([model_data[mn].get(c, np.nan) for c in all_cases], dtype=float)
        x = np.arange(n_cases) + i * (n_cases + 1)
        c = palette[i % len(palette)]
        ax.bar(
            x,
            vals,
            width=bar_width,
            color=c,
            edgecolor="black",
            alpha=0.75,
            label=_format_name(mn),
        )
        avg_val = np.nanmean(vals)
        ax.hlines(
            avg_val,
            x[0] - bar_width,
            x[-1] + bar_width,
            color=c,
            linestyle=(0, (3, 1)),
            linewidth=2.5,
        )
        group_centers.append(float(np.mean(x)))
        group_labels.append(_format_name(mn))

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, rotation=0, ha="center")
    ax.set_ylabel("Rollout RMSE")
    if not article_style:
        ax.set_title("Per-case Rollout RMSE for each model")
        ax.legend()
    fig.tight_layout()
    _save_fig(fig, output_dir, "case_rmse_bars", cfg, article_style)


# ============================================================================
# 5. Drag / Lift curves
# ============================================================================


def plot_forces(
    forces_dir: str,
    output_dir: str,
    truth_dir: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    article_style: bool = False,
) -> None:
    """Plot drag and lift time-series for each model, optionally with truth.

    Reads per-model directories under ``forces_dir/<model>/``.
    """
    cfg = cfg or PLOT_CONFIG
    _apply_style(cfg)
    palette = cfg.get("palette", PLOT_CONFIG["palette"])

    model_dirs = sorted(
        [
            d
            for d in os.listdir(forces_dir)
            if os.path.isdir(os.path.join(forces_dir, d)) and d != "truth"
        ]
    )
    if not model_dirs:
        print(f"[plot] No model subdirs in {forces_dir}")
        return

    # Collect case names from first model
    first_dir = os.path.join(forces_dir, model_dirs[0])
    case_csvs = sorted(glob.glob(os.path.join(first_dir, "*.csv")))
    case_names = [Path(c).stem for c in case_csvs]

    for case_name in case_names:
        for qty in ("drag", "lift"):
            fig, ax = plt.subplots(figsize=cfg["figsize"])

            # Truth
            if truth_dir:
                truth_csv = os.path.join(truth_dir, f"{case_name}.csv")
                if not os.path.exists(truth_csv):
                    case_norm = _normalize_case_id(case_name)
                    candidates = sorted(glob.glob(os.path.join(truth_dir, "*.csv")))
                    matched = [
                        p
                        for p in candidates
                        if _normalize_case_id(Path(p).stem) == case_norm
                    ]
                    if len(matched) == 1:
                        truth_csv = matched[0]
                if os.path.exists(truth_csv):
                    df_truth = pd.read_csv(truth_csv)
                    x_truth = (
                        df_truth["rollout_step"]
                        if "rollout_step" in df_truth.columns
                        else np.arange(len(df_truth))
                    )
                    ax.plot(
                        x_truth,
                        df_truth[qty],
                        label="Ground Truth",
                        **cfg["truth_style"],
                    )

            # Models
            for j, mn in enumerate(model_dirs):
                csv_path = os.path.join(forces_dir, mn, f"{case_name}.csv")
                if not os.path.exists(csv_path):
                    continue
                df = pd.read_csv(csv_path)
                x_pred = (
                    df["rollout_step"]
                    if "rollout_step" in df.columns
                    else np.arange(len(df))
                )
                ax.plot(
                    x_pred,
                    df[qty],
                    color=palette[j % len(palette)],
                    linewidth=cfg["line_width"],
                    alpha=0.85,
                    label=_format_name(mn),
                )

            ax.set_xlabel("Rollout step")
            ax.set_ylabel(f"$C_{{{qty[0].upper()}}}$")
            if not article_style:
                ax.set_title(f"{qty.capitalize()} — case {case_name}")
            ax.legend()
            fig.tight_layout()
            _save_fig(fig, output_dir, f"forces_{qty}_{case_name}", cfg, article_style)


# ============================================================================
# 6. Sensor truth-vs-pred signal plots
# ============================================================================


def plot_sensor_signals_from_cache(
    signal_root: str,
    output_dir: str,
    cfg: Optional[Dict[str, Any]] = None,
    article_style: bool = False,
    compact: bool = False,
) -> None:
    """Plot truth vs prediction signals at sensors from cached JSON files.

    Expects files under ``sensor_signals/<model>/<case>.json``.
    """
    cfg = cfg or PLOT_CONFIG
    _apply_style(cfg)

    model_dirs = sorted(
        [
            d
            for d in os.listdir(signal_root)
            if os.path.isdir(os.path.join(signal_root, d))
        ]
    )
    for model in model_dirs:
        case_files = sorted(glob.glob(os.path.join(signal_root, model, "*.json")))
        for case_file in case_files:
            with open(case_file) as fh:
                payload = json.load(fh)

            case_name = payload.get("case", Path(case_file).stem)
            pairs = payload.get("pairs", {})

            # Keep only requested pairs: velocity norm and pressure
            selected_pairs = []
            if "Vnorm_targ|Vnorm_pred" in pairs:
                selected_pairs.append(("Vnorm_targ", "Vnorm_pred"))
            for pressure_key in ("y2|x2", "P_targ|P_pred"):
                if pressure_key in pairs:
                    tf, pf = pressure_key.split("|")
                    selected_pairs.append((tf, pf))
                    break

            for truth_field, pred_field in selected_pairs:
                pair_key = f"{truth_field}|{pred_field}"
                sensor_dict = pairs[pair_key]

                # 3x3 legacy layout, optional compact mode drops last column (3x2)
                nrows, ncols = 3, (2 if compact else 3)
                fig, axs = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(20, 10),
                    dpi=cfg.get("dpi", 300),
                    squeeze=False,
                )
                axs_flat = axs.flatten()

                # Legacy-like ordering across the 9 sensor panels
                sensor_names = sorted(sensor_dict.keys())
                max_panels = nrows * ncols
                ordered_names = [
                    sensor_names[i] for i in range(min(max_panels, len(sensor_names)))
                ]

                for i, sensor_name in enumerate(ordered_names):
                    ax = axs_flat[i]
                    sig = sensor_dict[sensor_name]
                    truth = np.asarray(sig.get("truth", []), dtype=float)
                    pred = np.asarray(sig.get("pred", []), dtype=float)
                    n = min(len(truth), len(pred))
                    x = np.arange(n)

                    ax.plot(
                        x,
                        truth[:n],
                        color="black",
                        linestyle=(0, (4, 1)),
                        linewidth=1.75,
                        alpha=0.8,
                        label="Ground Truth",
                    )
                    ax.plot(
                        x,
                        pred[:n],
                        color="darkblue",
                        linestyle="-",
                        linewidth=2.25,
                        alpha=0.8,
                        label="Prediction",
                    )

                    ax.set_xlabel("Rollout step", fontsize=16)
                    ax.set_ylabel("Field", fontsize=16)
                    if len(x) > 0:
                        ax.set_xlim([float(np.min(x)), float(np.max(x))])

                    # Match article_style text format from legacy script
                    if article_style:
                        ax.text(
                            0.5,
                            0.97,
                            f"${sensor_name[0]}_{{{sensor_name[1:]}}}$",
                            transform=ax.transAxes,
                            fontsize=16,
                            ha="center",
                            va="top",
                            bbox=dict(
                                facecolor="white",
                                alpha=0.0,
                                edgecolor="none",
                                boxstyle="round,pad=0.2",
                            ),
                        )
                        ax.tick_params(axis="x", labelsize=14)
                        ax.tick_params(axis="y", labelsize=14)
                    else:
                        ax.set_title(
                            f"${sensor_name[0]}_{{{sensor_name[1:]}}}$", fontsize=16
                        )
                        if i == 0:
                            ax.legend(loc="upper right", fontsize=14)

                    ax.label_outer()

                for j in range(len(ordered_names), len(axs_flat)):
                    axs_flat[j].set_visible(False)

                fig.tight_layout()
                _save_fig(
                    fig,
                    output_dir,
                    f"sensor_signals_{model}_{case_name}_{truth_field}_{pred_field}",
                    cfg,
                    article_style,
                )


# ============================================================================
# Sensor & line location map
# ============================================================================


def plot_sensor_and_line_locations(
    domain_dim_dict: Dict[str, float],
    sensors: Dict[str, List[float]],
    lines: Dict[str, List[float]],
    object_origin: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Plot sensor points and line locations in the computational domain."""
    cfg = cfg or PLOT_CONFIG
    _apply_style(cfg)

    x_min = domain_dim_dict["x_min"]
    dx = domain_dim_dict["dx"]
    y_min = domain_dim_dict["y_min"]
    dy = domain_dim_dict["dy"]

    fig, ax = plt.subplots(figsize=(dx / 5, dy / 5))
    ax.set_xlim(x_min, x_min + dx)
    ax.set_ylim(y_min, y_min + dy)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Sensor & line locations")

    for name, coords in sensors.items():
        ax.plot(coords[0], coords[1], "o", color="none", markeredgecolor="black")
        label_txt = f"${name[0]}_{{{name[1:]}}}$" if len(name) > 1 else name
        ax.annotate(
            label_txt,
            (coords[0] + 0.2, coords[1]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="center",
        )

    if object_origin is not None:
        ax.plot(object_origin[0], object_origin[1], "x", color="red", markersize=10)
        ax.annotate(
            "Object",
            (object_origin[0], object_origin[1]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="center",
            color="red",
        )

    for line_name, line_coords in lines.items():
        if line_name.startswith("x"):
            ax.axhline(y=line_coords[1], color="blue", linestyle="--", label="x-Line")
        elif line_name.startswith("y"):
            ax.axvline(x=line_coords[0], color="green", linestyle="--", label="y-Line")

    ax.legend(loc="upper right")
    fig.tight_layout()
    if output_dir:
        _save_fig(fig, output_dir, "sensor_line_locations", cfg)
    else:
        plt.close(fig)


# ============================================================================
# Cumulated error per case (sensor-based)
# ============================================================================


def plot_sensor_cum_error(
    sensor_dir: str,
    output_dir: str,
    dt: float = 1.0,
    cfg: Optional[Dict[str, Any]] = None,
    article_style: bool = False,
) -> None:
    """Plot per-model cumulated sensor error (mean ± std over cases).

    Reads ``<model>_cum_error.npz`` files from ``sensor_dir``.
    """
    cfg = cfg or PLOT_CONFIG
    _apply_style(cfg)
    palette = cfg.get("palette", PLOT_CONFIG["palette"])

    npz_files = sorted(glob.glob(os.path.join(sensor_dir, "*_cum_error.npz")))
    if not npz_files:
        print(f"[plot] No *_cum_error.npz files in {sensor_dir}")
        return

    fig, ax = plt.subplots(figsize=cfg["figsize"])
    for i, npz_path in enumerate(npz_files):
        model_name = Path(npz_path).stem.replace("_cum_error", "")
        data = np.load(npz_path)
        mean = data["mean"]
        std = data["std"]
        t = np.arange(len(mean))
        c = palette[i % len(palette)]
        ax.plot(
            t,
            mean,
            color=c,
            linewidth=cfg["line_width"],
            label=_format_name(model_name),
        )
        ax.fill_between(
            t, mean - std, mean + std, color=c, alpha=cfg["error_band_alpha"]
        )

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Cumulated Error")
    if not article_style:
        ax.set_title("Cumulated sensor error (mean ± std over cases)")
    ax.legend(ncol=2)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save_fig(fig, output_dir, "sensor_cum_error", cfg, article_style)


# ============================================================================
# CLI — run all available plots from pre-computed results
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate publication-quality plots from postprocessing results."
    )
    p.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Root results directory (output of compute_errors / compute_forces).",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Plot output directory (defaults to <directory>/plots).",
    )
    p.add_argument(
        "--article-style",
        action="store_true",
        help="Use minimal annotation style for camera-ready figures.",
    )
    p.add_argument(
        "--forces-dir",
        default=None,
        help="forces/ directory if different from <directory>/forces.",
    )
    p.add_argument(
        "--truth-dir",
        default=None,
        help="Ground-truth forces directory for comparison.",
    )
    p.add_argument(
        "--format",
        nargs="+",
        default=["png"],
        help="Output image format(s), e.g., png pdf svg.",
    )
    p.add_argument(
        "--compact-sensors",
        action="store_true",
        help="Use compact sensor layout (drop last subplot column: 3x2 instead of 3x3).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    results_dir = args.directory
    out_dir = args.output or os.path.join(results_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    cfg = dict(PLOT_CONFIG)
    cfg["save_formats"] = args.format

    art = args.article_style

    rmse_out = os.path.join(out_dir, "rmse")
    sensor_out = os.path.join(out_dir, "sensors")
    force_out = os.path.join(out_dir, "forces")

    # --- RMSE plots ---
    summary_csv = os.path.join(results_dir, "rmse_summary.csv")
    if os.path.exists(summary_csv):
        plot_cumulated_rmse(summary_csv, rmse_out, cfg=cfg, article_style=art)
        plot_rmse_per_timestep(summary_csv, rmse_out, cfg=cfg, article_style=art)
        print(f"[plot] RMSE curves → {rmse_out}")

    per_case_dir = os.path.join(results_dir, "rmse_per_case")
    if os.path.isdir(per_case_dir):
        plot_case_rmse_bars(per_case_dir, rmse_out, cfg=cfg, article_style=art)
        print(f"[plot] Case RMSE bars → {rmse_out}")

    # --- Sensor error ---
    sensor_dir = os.path.join(results_dir, "sensor_errors")
    if os.path.isdir(sensor_dir):
        plot_sensor_cum_error(sensor_dir, sensor_out, cfg=cfg, article_style=art)
    signal_root = os.path.join(results_dir, "sensor_signals")
    if os.path.isdir(signal_root):
        plot_sensor_signals_from_cache(
            signal_root,
            sensor_out,
            cfg=cfg,
            article_style=art,
            compact=args.compact_sensors,
        )
    if os.path.isdir(sensor_dir) or os.path.isdir(signal_root):
        print(f"[plot] Sensor plots → {sensor_out}")

    # --- Forces ---
    forces_dir = args.forces_dir or os.path.join(results_dir, "forces")
    if os.path.isdir(forces_dir):
        plot_forces(
            forces_dir, force_out, truth_dir=args.truth_dir, cfg=cfg, article_style=art
        )
        print(f"[plot] Force curves → {force_out}")

    print(f"[plot] All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
