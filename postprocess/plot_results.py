from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from tqdm import tqdm

from postprocess.utils.xdmf_io import (
    case_sort_key_from_configs,
    load_configs_pool,
    load_json,
)

matplotlib.use("Agg")


PLOT_CONFIG = {
    "font_family": "serif",
    "fig_sensor": (15, 7.5),
    "fig_forces": (10.5, 6),
    "fig_rmse": (9.75, 5.25),
    "dpi": 300,
    "gt_color": "black",
    "gt_linestyle": (0, (4, 1)),
    "gt_linewidth": 1.75,
    "pred_color": "darkblue",
    "pred_linestyle": "-",
    "pred_linewidth": 2.25,
    "pred_alpha": 0.8,
    "rmse_linewidth": 1.8,
    "palette": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    "std_alpha": 0.3,
    "bar_truth_color": "#7f7f7f",
    "legend_face_alpha": 0.4,
    "legend_edge_color": "black",
    "fontsize_title": 19,
    "fontsize_axis": 19,
    "fontsize_ticks": 17,
    "fontsize_legend": 16,
    "save_formats": ["png"],
    "sensor_ylim_velocity": (0.0, 1.6),
    "sensor_yticks_velocity": [0.0, 0.5, 1.0, 1.5],
    "sensor_ylim_pressure": (-0.6, 0.6),
    "sensor_yticks_pressure": [-0.5, -0.25, 0.0, 0.25, 0.5],
    "rollout_start": 0,
    "rollout_end": 600,
    "rollout_tick_step": 100,
    "sensor_drop_last_column": False,
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": PLOT_CONFIG["font_family"],
            "axes.titlesize": PLOT_CONFIG["fontsize_title"],
            "axes.labelsize": PLOT_CONFIG["fontsize_axis"],
            "xtick.labelsize": PLOT_CONFIG["fontsize_ticks"],
            "ytick.labelsize": PLOT_CONFIG["fontsize_ticks"],
            "legend.fontsize": PLOT_CONFIG["fontsize_legend"],
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def _box_spines(ax: plt.Axes) -> None:
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)


def _rollout_xticks() -> List[int]:
    ticks = list(
        range(
            int(PLOT_CONFIG["rollout_tick_step"]),
            int(PLOT_CONFIG["rollout_end"]),
            int(PLOT_CONFIG["rollout_tick_step"]),
        )
    )
    return [int(PLOT_CONFIG["rollout_start"]), *ticks, int(PLOT_CONFIG["rollout_end"])]


def _apply_rollout_axis(ax: plt.Axes) -> None:
    ax.set_xlim(int(PLOT_CONFIG["rollout_start"]), int(PLOT_CONFIG["rollout_end"]))
    ax.set_xticks(_rollout_xticks())


def _apply_sensor_rollout_axis(ax: plt.Axes) -> None:
    if PLOT_CONFIG["sensor_drop_last_column"]:
        ax.set_xlim(300, int(PLOT_CONFIG["rollout_end"]))
        ax.set_xticks([300, 400, 500, int(PLOT_CONFIG["rollout_end"])])
    else:
        _apply_rollout_axis(ax)


def _lighten_color(color: str, blend: float = 0.75) -> tuple:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1.0 - blend) * rgb + blend * np.ones(3))


def _ordered_sensor_ids(sensor_ids: List[str], cropped: bool) -> List[str]:
    # Reading order layout:
    # top row    -> p1 p2 p3
    # middle row -> p4 p5 p6
    # bottom row -> p7 p8 p9
    layout = [
        ["p1", "p2", "p3"],
        ["p4", "p5", "p6"],
        ["p7", "p8", "p9"],
    ]
    if cropped:
        layout = [row[:2] for row in layout]

    available = {sid.lower(): sid for sid in sensor_ids}
    ordered = []
    for row in layout:
        for sid in row:
            if sid in available:
                ordered.append(available[sid])

    remaining = [sid for sid in sensor_ids if sid not in ordered]
    return ordered + remaining


def _style_legend(legend: Optional[matplotlib.legend.Legend]) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor((1.0, 1.0, 1.0, PLOT_CONFIG["legend_face_alpha"]))
    frame.set_edgecolor(mcolors.to_rgba(PLOT_CONFIG["legend_edge_color"], alpha=1.0))
    frame.set_linewidth(1.0)


def _hide_top_right_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _apply_rollout_rmse_style(ax: plt.Axes) -> None:
    ax.set_ylabel("RMSE rollout")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v * 1e3:g}"))
    ax.text(
        0.0,
        1.01,
        "1e-3",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PLOT_CONFIG["fontsize_ticks"],
    )
    _hide_top_right_spines(ax)


def _style_rollout_legend(legend: Optional[matplotlib.legend.Legend]) -> None:
    if legend is None:
        return
    legend.set_frame_on(False)
    for handle in legend.legend_handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(max(3.0, PLOT_CONFIG["rmse_linewidth"] * 1.8))


def _force_compact_legend(ax: plt.Axes, labels: List[str], colors: List[str]) -> None:
    handles: List[object] = [
        *[Patch(facecolor=c, edgecolor="black", linewidth=0.8) for c in colors],
        Patch(facecolor="white", edgecolor="black", linewidth=0.8, label="mean"),
        Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            hatch="//",
            label="std",
        ),
    ]
    legend_labels: List[str] = [*labels, "mean", "std"]
    legend = ax.legend(handles, legend_labels, loc="upper left", ncol=2)
    _style_legend(legend)


def _rollout_from_cumulative(cum_vals: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    denom = np.maximum(timesteps.astype(float), 1.0)
    return cum_vals / denom


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in PLOT_CONFIG["save_formats"]:
        fig.savefig(
            out_dir / f"{stem}.{ext}", dpi=PLOT_CONFIG["dpi"], bbox_inches="tight"
        )
    plt.close(fig)


def _sorted_cases_by_re(case_ids: List[str], configs_df: pd.DataFrame) -> List[str]:
    tuples = [(case_sort_key_from_configs(configs_df, cid), cid) for cid in case_ids]
    tuples.sort(
        key=lambda x: (x[0][0], int(x[1]) if str(x[1]).isdigit() else str(x[1]))
    )
    return [x[1] for x in tuples]


def _plot_sensors(
    model_root: Path,
    figures_dir: Path,
    model_label: str,
    configs_df: pd.DataFrame,
) -> None:
    df = pd.read_csv(model_root / "sensors" / "sensor_data.csv")
    cases = _sorted_cases_by_re(df["case_id"].astype(str).unique().tolist(), configs_df)
    for case_id in tqdm(cases, desc="Plot sensor cases"):
        dcase = df[df["case_id"].astype(str) == case_id]
        sensor_ids = _ordered_sensor_ids(
            sorted(dcase["sensor_id"].astype(str).unique()),
            cropped=PLOT_CONFIG["sensor_drop_last_column"],
        )
        for truth_col, pred_col, field_label in [
            ("v_targ", "v_pred", "velocity"),
            ("p_targ", "p_pred", "pressure"),
        ]:
            max_panels = 6 if PLOT_CONFIG["sensor_drop_last_column"] else 9
            ncols = 2 if max_panels == 6 else 3
            sensor_suffix = "_cropped" if PLOT_CONFIG["sensor_drop_last_column"] else ""
            fig, axs = plt.subplots(
                3, ncols, figsize=PLOT_CONFIG["fig_sensor"], squeeze=False
            )
            for i, sid in enumerate(sensor_ids[:max_panels]):
                ax = axs.flatten()[i]
                ds = dcase[dcase["sensor_id"].astype(str) == sid].sort_values(
                    "timestep"
                )
                x = ds["timestep"].to_numpy()
                ax.plot(
                    x,
                    ds[truth_col].to_numpy(),
                    color=PLOT_CONFIG["gt_color"],
                    linestyle=PLOT_CONFIG["gt_linestyle"],
                    linewidth=PLOT_CONFIG["gt_linewidth"],
                    alpha=0.8,
                    label="Ground Truth",
                )
                ax.plot(
                    x,
                    ds[pred_col].to_numpy(),
                    color=PLOT_CONFIG["pred_color"],
                    linestyle=PLOT_CONFIG["pred_linestyle"],
                    linewidth=PLOT_CONFIG["pred_linewidth"],
                    alpha=PLOT_CONFIG["pred_alpha"],
                    label=model_label,
                )
                ax.text(
                    0.5,
                    0.96,
                    f"${sid[0]}_{{{sid[1:]}}}$",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=PLOT_CONFIG["fontsize_title"],
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": "black",
                        "alpha": 0.85,
                    },
                )

                # Shared y-range across all sensors/cases/models
                if field_label == "velocity":
                    ax.set_ylim(*PLOT_CONFIG["sensor_ylim_velocity"])
                    ax.set_yticks(PLOT_CONFIG["sensor_yticks_velocity"])
                else:
                    ax.set_ylim(*PLOT_CONFIG["sensor_ylim_pressure"])
                    ax.set_yticks(PLOT_CONFIG["sensor_yticks_pressure"])

                _apply_sensor_rollout_axis(ax)

                # Only one y-label (middle-left) and one x-label (bottom-middle)
                row_i = i // ncols
                col_i = i % ncols
                if row_i == 1 and col_i == 0:
                    if field_label == "velocity":
                        ax.set_ylabel(r"Velocity [$m.s^{-1}$]")
                    else:
                        ax.set_ylabel(r"Pressure [$Pa$]")
                else:
                    ax.set_ylabel("")
                if row_i == 2 and col_i == ncols // 2:
                    ax.set_xlabel("Rollout step")
                else:
                    ax.set_xlabel("")

                if col_i > 0:
                    ax.tick_params(axis="y", labelleft=False)
                if row_i < 2:
                    ax.tick_params(axis="x", labelbottom=False)

                _box_spines(ax)

            for j in range(len(sensor_ids[:max_panels]), 3 * ncols):
                axs.flatten()[j].set_visible(False)

            # Place one global legend as a single centered line above subplots.
            handles, labels = axs.flatten()[0].get_legend_handles_labels()
            legend = fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                frameon=True,
                ncol=len(labels),
            )
            _style_legend(legend)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.6)
            _save(fig, figures_dir, f"sensor_{field_label}_{case_id}{sensor_suffix}")


def _plot_forces(model_root: Path, figures_dir: Path, model_label: str) -> None:
    forces_dir = model_root / "forces"
    force_csvs = sorted(forces_dir.glob("forces_*.csv"))
    for csv_path in tqdm(force_csvs, desc="Plot force cases"):
        if csv_path.name == "forces_summary.csv":
            continue
        df = pd.read_csv(csv_path).sort_values("timestep")
        case_id = str(df["case_id"].iloc[0])
        fig, axs = plt.subplots(2, 1, figsize=PLOT_CONFIG["fig_forces"], sharex=True)
        x = df["timestep"].to_numpy()
        axs[0].plot(
            x,
            df["fx_targ"],
            color=PLOT_CONFIG["gt_color"],
            linestyle=PLOT_CONFIG["gt_linestyle"],
            linewidth=PLOT_CONFIG["gt_linewidth"],
            label="Ground Truth",
        )
        axs[0].plot(
            x,
            df["fx_pred"],
            color=PLOT_CONFIG["pred_color"],
            linestyle=PLOT_CONFIG["pred_linestyle"],
            linewidth=PLOT_CONFIG["pred_linewidth"],
            label=model_label,
        )
        axs[0].set_ylabel(r"$F_x$ [$N$]")
        legend = axs[0].legend(loc="upper left")
        _style_legend(legend)
        axs[0].axhline(
            0.0, color="#505050", linewidth=0.8, linestyle="--", alpha=0.35, zorder=1
        )
        _apply_rollout_axis(axs[0])
        _box_spines(axs[0])

        axs[1].plot(
            x,
            df["fy_targ"],
            color=PLOT_CONFIG["gt_color"],
            linestyle=PLOT_CONFIG["gt_linestyle"],
            linewidth=PLOT_CONFIG["gt_linewidth"],
        )
        axs[1].plot(
            x,
            df["fy_pred"],
            color=PLOT_CONFIG["pred_color"],
            linestyle=PLOT_CONFIG["pred_linestyle"],
            linewidth=PLOT_CONFIG["pred_linewidth"],
        )
        axs[1].set_ylabel(r"$F_y$ [$N$]")
        axs[1].set_xlabel("Rollout step")
        axs[1].axhline(
            0.0, color="#505050", linewidth=0.8, linestyle="--", alpha=0.35, zorder=1
        )
        _apply_rollout_axis(axs[1])
        _box_spines(axs[1])
        fig.tight_layout()
        _save(fig, figures_dir, f"forces_{case_id}")


def _plot_rmse(
    model_root: Path, figures_dir: Path, cfg: Dict[str, object], model_label: str
) -> None:
    err_dir = model_root / "errors"
    per_case = pd.read_csv(err_dir / "per_case_rmse.csv")
    cum_mean = pd.read_csv(err_dir / "cumulative_rmse_mean.csv")
    configs_df = load_configs_pool(cfg["configs_pool"])

    rmse_curves = sorted(err_dir.glob("cumulative_rmse_*.csv"))
    for case_csv in tqdm(rmse_curves, desc="Plot rollout error cases"):
        if case_csv.name == "cumulative_rmse_mean.csv":
            continue
        d = pd.read_csv(case_csv)
        cid = case_csv.stem.replace("cumulative_rmse_", "")
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["fig_rmse"])
        x = d["timestep"].to_numpy()
        rollout_total = _rollout_from_cumulative(d["cum_rmse_total"].to_numpy(), x)
        ax.plot(
            x,
            rollout_total,
            label="total",
            linewidth=PLOT_CONFIG["rmse_linewidth"],
        )
        ax.set_xlabel("Rollout step")
        _apply_rollout_axis(ax)
        _apply_rollout_rmse_style(ax)
        ax.set_ylim(bottom=0.0)
        legend = ax.legend(loc="upper left", ncol=1)
        _style_rollout_legend(legend)
        fig.tight_layout()
        _save(fig, figures_dir, f"rollout_error_{cid}")

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["fig_rmse"])
    x = cum_mean["timestep"].to_numpy()
    if "rollout_rmse_total_mean" in cum_mean.columns:
        m = cum_mean["rollout_rmse_total_mean"].to_numpy()
        s = cum_mean["rollout_rmse_total_std"].to_numpy()
    else:
        m = _rollout_from_cumulative(cum_mean["cum_rmse_total_mean"].to_numpy(), x)
        s = _rollout_from_cumulative(cum_mean["cum_rmse_total_std"].to_numpy(), x)
    ax.plot(x, m, label="total", linewidth=PLOT_CONFIG["rmse_linewidth"])
    ax.fill_between(x, m - s, m + s, alpha=PLOT_CONFIG["std_alpha"])
    ax.set_xlabel("Rollout step")
    _apply_rollout_axis(ax)
    _apply_rollout_rmse_style(ax)
    ax.set_ylim(bottom=0.0)
    legend = ax.legend(loc="upper left", ncol=1)
    _style_rollout_legend(legend)
    fig.tight_layout()
    _save(fig, figures_dir, "rollout_error_mean")

    per_case["case_id"] = per_case["case_id"].astype(str)
    ordered = _sorted_cases_by_re(per_case["case_id"].tolist(), configs_df)
    per_case = per_case.set_index("case_id").loc[ordered].reset_index()

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["fig_rmse"])
    x = np.arange(len(per_case), dtype=float)
    y = per_case["rmse_total_mean"].to_numpy()
    ax.bar(
        x,
        y,
        width=1.0,
        align="edge",
        color=PLOT_CONFIG["pred_color"],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
        label=model_label,
        zorder=2,
    )
    overall_mean = float(np.mean(y)) if len(y) else 0.0
    ax.axhline(
        overall_mean,
        color=PLOT_CONFIG["pred_color"],
        linestyle="--",
        linewidth=3.0,
        zorder=3,
    )
    ax.set_xlabel(r"Case ID (ascending $Re$)")
    ax.set_ylabel("RMSE total mean")
    ax.set_xlim(0.0, float(len(per_case)))
    ax.set_xticks(x + 0.5)
    ax.set_xticklabels(per_case["case_id"].tolist())
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    legend = ax.legend(loc="upper right")
    _style_legend(legend)
    _box_spines(ax)
    fig.tight_layout()
    _save(fig, figures_dir, "per_case_rmse_bar")


def _plot_force_bars(
    model_root: Path, figures_dir: Path, cfg: Dict[str, object], model_label: str
) -> None:
    summary = pd.read_csv(model_root / "forces" / "forces_summary.csv")
    summary["case_id"] = summary["case_id"].astype(str)
    configs_df = load_configs_pool(cfg["configs_pool"])
    ordered = _sorted_cases_by_re(summary["case_id"].tolist(), configs_df)
    summary = summary.set_index("case_id").loc[ordered].reset_index()

    x = np.arange(len(summary), dtype=float)

    for comp in ["fx", "fy"]:
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["fig_rmse"])
        # Two paired groups per case: mean and std, each comparing truth vs model.
        block = 0.8
        barw = block / 4.0
        if comp == "fx":
            cols = {
                "mean_targ": "fx_targ_mean",
                "mean_pred": "fx_pred_mean",
                "std_targ": "fx_targ_std",
                "std_pred": "fx_pred_std",
            }
            ylabel = r"$F_x$ [$N$]"
        else:
            cols = {
                "mean_targ": "fy_targ_mean",
                "mean_pred": "fy_pred_mean",
                "std_targ": "fy_targ_std",
                "std_pred": "fy_pred_std",
            }
            ylabel = r"$F_y$ [$N$]"

        x_mean_targ = x - block / 2 + 0.5 * barw
        x_mean_pred = x - block / 2 + 1.5 * barw
        x_std_targ = x - block / 2 + 2.5 * barw
        x_std_pred = x - block / 2 + 3.5 * barw

        ax.bar(
            x_mean_targ,
            summary[cols["mean_targ"]],
            width=barw,
            color=PLOT_CONFIG["bar_truth_color"],
            edgecolor="black",
            linewidth=0.8,
            label="Ground Truth mean",
        )
        ax.bar(
            x_mean_pred,
            summary[cols["mean_pred"]],
            width=barw,
            color=PLOT_CONFIG["pred_color"],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            label=f"{model_label} mean",
        )
        ax.bar(
            x_std_targ,
            summary[cols["std_targ"]],
            width=barw,
            color=_lighten_color(PLOT_CONFIG["bar_truth_color"]),
            edgecolor="black",
            linewidth=0.8,
            hatch="//",
            label="Ground Truth std",
        )
        ax.bar(
            x_std_pred,
            summary[cols["std_pred"]],
            width=barw,
            color=_lighten_color(PLOT_CONFIG["pred_color"]),
            edgecolor="black",
            linewidth=0.8,
            hatch="//",
            alpha=0.9,
            label=f"{model_label} std",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(summary["case_id"].tolist())
        ax.set_xlabel(r"Case ID (ascending $Re$)")
        ax.set_ylabel(ylabel)
        ax.axhline(
            0.0, color="#505050", linewidth=0.8, linestyle="--", alpha=0.35, zorder=1
        )
        _force_compact_legend(
            ax,
            labels=["Ground Truth", model_label],
            colors=[PLOT_CONFIG["bar_truth_color"], PLOT_CONFIG["pred_color"]],
        )
        ymin, ymax = ax.get_ylim()
        if ymax > 0:
            ax.set_ylim(ymin, ymax * 1.2)
        ax.tick_params(axis="x", rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        _box_spines(ax)
        fig.tight_layout()
        _save(fig, figures_dir, f"force_bar_{comp}")


def run(
    config_path: str,
    output_override: Optional[str],
    only: str = "all",
    sensor_drop_last_column_override: bool = False,
) -> None:
    cfg = load_json(config_path)
    PLOT_CONFIG["sensor_drop_last_column"] = bool(
        sensor_drop_last_column_override
        or cfg.get("sensor_drop_last_column", PLOT_CONFIG["sensor_drop_last_column"])
    )
    _setup_style()

    model_name = cfg["model_name"]
    model_label = str(cfg.get("model_shortname") or cfg.get("nickname") or model_name)
    configs_df = load_configs_pool(cfg["configs_pool"])
    model_root = Path(cfg["output_dir"]) / model_name
    figures_dir = Path(output_override) if output_override else model_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if only in {"all", "sensors"}:
        _plot_sensors(model_root, figures_dir, model_label, configs_df)
    if only in {"all", "forces"}:
        _plot_forces(model_root, figures_dir, model_label)
        _plot_force_bars(model_root, figures_dir, cfg, model_label)
    if only in {"all", "rmse"}:
        _plot_rmse(model_root, figures_dir, cfg, model_label)
    print(f"[plot_results] Figures written to {figures_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication figures for one model"
    )
    parser.add_argument("config", help="Unified JSON config file")
    parser.add_argument(
        "--output_dir", default=None, help="Optional custom output figure directory"
    )
    parser.add_argument(
        "--only",
        choices=["all", "sensors", "rmse", "forces"],
        default="all",
        help="Restrict plotting to one figure family",
    )
    parser.add_argument(
        "--sensor-drop-last-column",
        action="store_true",
        help="Override config and use 3x2 cropped sensor layout",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config, args.output_dir, args.only, args.sensor_drop_last_column)


if __name__ == "__main__":
    main()
