from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib


import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from tqdm import tqdm

from postprocess.plot_results import (
    PLOT_CONFIG,
    _apply_sensor_rollout_axis,
    _ordered_sensor_ids,
    _style_legend,
)
from postprocess.utils.xdmf_io import (
    case_sort_key_from_configs,
    load_configs_pool,
    load_json,
)

matplotlib.use("Agg")


def _model_label(model_dir: Path, nickname: Optional[str]) -> str:
    return nickname if nickname else model_dir.name


def _load_model_outputs(model_dir: Path) -> dict:
    return {
        "sensor": pd.read_csv(model_dir / "sensors" / "sensor_data.csv"),
        "rmse": pd.read_csv(model_dir / "errors" / "per_case_rmse.csv"),
        "cum": pd.read_csv(model_dir / "errors" / "cumulative_rmse_mean.csv"),
        "forces_summary": pd.read_csv(model_dir / "forces" / "forces_summary.csv"),
        "forces": {
            p.stem.replace("forces_", ""): pd.read_csv(p)
            for p in sorted((model_dir / "forces").glob("forces_*.csv"))
        },
    }


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ["png"]:
        fig.savefig(out / f"{stem}.{ext}", dpi=PLOT_CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)


def _box_spines(ax: plt.Axes) -> None:
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)


def _hide_top_right_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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


def _lighten_color(color: str, blend: float = 0.75) -> tuple:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1.0 - blend) * rgb + blend * np.ones(3))


def _rollout_from_cumulative(cum_vals: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
    denom = np.maximum(timesteps.astype(float), 1.0)
    return cum_vals / denom


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


def _style_rollout_legend(legend):
    if legend is None:
        return
    legend.set_frame_on(False)
    for handle in legend.legend_handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(max(3.0, PLOT_CONFIG["rmse_linewidth"] * 1.8))


def _force_compact_legend(ax: plt.Axes, labels: List[str], colors: List[str]) -> None:
    handles: List[object] = [
        *[Patch(facecolor=c, edgecolor="black", linewidth=0.8) for c in colors],
        Patch(facecolor="white", edgecolor="black", linewidth=0.8),
        Patch(facecolor="white", edgecolor="black", linewidth=0.8, hatch="//"),
    ]
    legend_labels: List[str] = [*labels, "mean", "std"]
    legend = ax.legend(handles, legend_labels, loc="upper left", ncol=2)
    _style_legend(legend)


def _write_rmse_summary(
    models_data: List[dict],
    labels: List[str],
    model_paths: List[Path],
    common_cases: List[str],
    out: Path,
) -> None:
    rows = []
    for label, model_path, mdata in zip(labels, model_paths, models_data):
        rmse = mdata["rmse"].copy()
        rmse["case_id"] = rmse["case_id"].astype(str)
        rmse = rmse.set_index("case_id").loc[common_cases].reset_index()
        rows.append(
            {
                "model": label,
                "model_dir": str(model_path),
                "rmse_1step_mean": float(np.mean(rmse["rmse_step1"].to_numpy())),
                "rmse_1step_std": float(np.std(rmse["rmse_step1"].to_numpy(), ddof=0)),
                "rmse_50step_mean": float(np.mean(rmse["rmse_50step"].to_numpy())),
                "rmse_50step_std": float(
                    np.std(rmse["rmse_50step"].to_numpy(), ddof=0)
                ),
                "rmse_all_mean": float(np.mean(rmse["rmse_all"].to_numpy())),
                "rmse_all_std": float(np.std(rmse["rmse_all"].to_numpy(), ddof=0)),
            }
        )
    pd.DataFrame(rows).to_csv(out / "compare_rmse_summary.csv", index=False)


def _assert_same_cases(models_data: List[dict], labels: List[str]) -> List[str]:
    case_sets = [set(d["rmse"]["case_id"].astype(str).tolist()) for d in models_data]
    ref = case_sets[0]
    for label, cset in zip(labels[1:], case_sets[1:]):
        if cset != ref:
            raise ValueError(
                f"Case ID mismatch for model {label}: expected {sorted(ref)}, got {sorted(cset)}"
            )
    return sorted(ref, key=lambda x: int(x) if x.isdigit() else x)


def run(
    config_path: str,
    model_dirs: List[str],
    nicknames: Optional[List[str]],
    output_dir: str,
    only: str = "all",
    sensor_drop_last_column_override: bool = False,
) -> None:
    cfg = load_json(config_path)
    PLOT_CONFIG["sensor_drop_last_column"] = bool(
        sensor_drop_last_column_override
        or cfg.get("sensor_drop_last_column", PLOT_CONFIG["sensor_drop_last_column"])
    )
    plt.rcParams.update(
        {
            "font.family": PLOT_CONFIG["font_family"],
            "axes.titlesize": PLOT_CONFIG["fontsize_title"],
            "axes.labelsize": PLOT_CONFIG["fontsize_axis"],
            "xtick.labelsize": PLOT_CONFIG["fontsize_ticks"],
            "ytick.labelsize": PLOT_CONFIG["fontsize_ticks"],
            "legend.fontsize": PLOT_CONFIG["fontsize_legend"],
        }
    )

    model_paths = [Path(p) for p in model_dirs]
    labels = [
        _model_label(p, nicknames[i] if nicknames else None)
        for i, p in enumerate(model_paths)
    ]
    colors = PLOT_CONFIG["palette"]

    models_data = [
        _load_model_outputs(p) for p in tqdm(model_paths, desc="Load model outputs")
    ]
    common_cases = _assert_same_cases(models_data, labels)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_pool = load_configs_pool(cfg["configs_pool"])
    common_cases = sorted(
        common_cases,
        key=lambda cid: (
            case_sort_key_from_configs(cfg_pool, cid)[0],
            int(cid) if str(cid).isdigit() else str(cid),
        ),
    )
    _write_rmse_summary(models_data, labels, model_paths, common_cases, out)

    if only in {"all", "sensors"}:
        for case_id in tqdm(common_cases, desc="Compare sensor cases"):
            for truth_col, pred_col, field_label in [
                ("v_targ", "v_pred", "velocity"),
                ("p_targ", "p_pred", "pressure"),
            ]:
                max_panels = 6 if PLOT_CONFIG["sensor_drop_last_column"] else 9
                ncols = 2 if max_panels == 6 else 3
                sensor_suffix = (
                    "_cropped" if PLOT_CONFIG["sensor_drop_last_column"] else ""
                )
                fig, axs = plt.subplots(
                    3, ncols, figsize=PLOT_CONFIG["fig_sensor"], squeeze=False
                )
                base = models_data[0]["sensor"]
                dcase_base = base[base["case_id"].astype(str) == case_id]
                sensors = _ordered_sensor_ids(
                    sorted(dcase_base["sensor_id"].astype(str).unique()),
                    cropped=PLOT_CONFIG["sensor_drop_last_column"],
                )[:max_panels]
                for i, sid in enumerate(sensors):
                    ax = axs.flatten()[i]
                    gt = dcase_base[
                        dcase_base["sensor_id"].astype(str) == sid
                    ].sort_values("timestep")
                    x_gt = gt["timestep"].to_numpy()
                    ax.plot(
                        x_gt,
                        gt[truth_col],
                        color="black",
                        linestyle=(0, (4, 1)),
                        linewidth=2.0,
                        alpha=0.9,
                        label="Ground Truth",
                    )
                    for midx, mdata in enumerate(models_data):
                        ds = mdata["sensor"]
                        s = ds[
                            (ds["case_id"].astype(str) == case_id)
                            & (ds["sensor_id"].astype(str) == sid)
                        ].sort_values("timestep")
                        x = s["timestep"].to_numpy()
                        ax.plot(
                            x,
                            s[pred_col],
                            color=colors[midx % len(colors)],
                            linewidth=2.4,
                            alpha=0.72,
                            label=labels[midx],
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
                    if field_label == "velocity":
                        ax.set_ylim(*PLOT_CONFIG["sensor_ylim_velocity"])
                        ax.set_yticks(PLOT_CONFIG["sensor_yticks_velocity"])
                    else:
                        ax.set_ylim(*PLOT_CONFIG["sensor_ylim_pressure"])
                        ax.set_yticks(PLOT_CONFIG["sensor_yticks_pressure"])
                    _apply_sensor_rollout_axis(ax)

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

                for j in range(len(sensors), 3 * ncols):
                    axs.flatten()[j].set_visible(False)

                handles, legend_labels = axs.flatten()[0].get_legend_handles_labels()
                legend = fig.legend(
                    handles,
                    legend_labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.02),
                    frameon=True,
                    ncol=len(legend_labels),
                )
                _style_legend(legend)
                fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.6)
                _save(
                    fig, out, f"compare_sensor_{field_label}_{case_id}{sensor_suffix}"
                )

    if only in {"all", "rmse"}:
        fig, ax = plt.subplots(figsize=PLOT_CONFIG["fig_rmse"])
        for midx, mdata in enumerate(models_data):
            d = mdata["cum"]
            x = d["timestep"].to_numpy()
            if "rollout_rmse_total_mean" in d.columns:
                m = d["rollout_rmse_total_mean"].to_numpy()
                s = d["rollout_rmse_total_std"].to_numpy()
            else:
                m = _rollout_from_cumulative(d["cum_rmse_total_mean"].to_numpy(), x)
                s = _rollout_from_cumulative(d["cum_rmse_total_std"].to_numpy(), x)
            ax.plot(
                x,
                m,
                color=colors[midx % len(colors)],
                linewidth=2.2,
                label=labels[midx],
            )
            ax.fill_between(
                x, m - s, m + s, color=colors[midx % len(colors)], alpha=0.25
            )
        ax.set_xlabel("Rollout step")
        _apply_rollout_axis(ax)
        _apply_rollout_rmse_style(ax)
        ax.set_ylim(bottom=0.0)
        legend = ax.legend(loc="upper left", ncol=1)
        _style_rollout_legend(legend)
        fig.tight_layout()
        _save(fig, out, "compare_rollout_error_total")

        fig, ax = plt.subplots(figsize=PLOT_CONFIG["fig_forces"])
        num_cases = len(common_cases)
        block_gap = 1.0
        xticks = []
        xticklabels = []
        for midx, mdata in enumerate(models_data):
            rmse = mdata["rmse"].copy()
            rmse["case_id"] = rmse["case_id"].astype(str)
            rmse = rmse.set_index("case_id").loc[common_cases].reset_index()
            start = midx * (num_cases + block_gap)
            x = start + np.arange(num_cases, dtype=float)
            y = rmse["rmse_total_mean"].to_numpy()
            ax.bar(
                x,
                y,
                width=1.0,
                align="edge",
                color=colors[midx % len(colors)],
                edgecolor="black",
                linewidth=0.8,
                label=labels[midx],
                zorder=2,
            )
            mean_y = float(np.mean(y)) if len(y) else 0.0
            ax.hlines(
                y=mean_y,
                xmin=start,
                xmax=start + num_cases,
                colors=colors[midx % len(colors)],
                linestyles="--",
                linewidth=3.0,
                zorder=3,
            )
            xticks.append(start + num_cases / 2.0)
            xticklabels.append(labels[midx])
        if len(models_data) > 0:
            ax.set_xlim(0.0, len(models_data) * (num_cases + block_gap) - block_gap)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE total mean")
        legend = ax.legend(loc="upper right", ncol=2)
        _style_legend(legend)
        _box_spines(ax)
        fig.tight_layout()
        _save(fig, out, "compare_rmse_bars")

    if only in {"all", "forces"}:
        for comp in tqdm(["fx", "fy"], desc="Compare force bar groups"):
            fig, ax = plt.subplots(figsize=PLOT_CONFIG["fig_forces"])
            base_summary = models_data[0]["forces_summary"].copy()
            base_summary["case_id"] = base_summary["case_id"].astype(str)
            base_summary = (
                base_summary.set_index("case_id").loc[common_cases].reset_index()
            )

            group_width = 0.9
            series_per_case = 2 * (len(models_data) + 1)
            barw = group_width / series_per_case
            for i, case_id in enumerate(common_cases):
                if comp == "fx":
                    t_mean_col, t_std_col = "fx_targ_mean", "fx_targ_std"
                    p_mean_col, p_std_col = "fx_pred_mean", "fx_pred_std"
                else:
                    t_mean_col, t_std_col = "fy_targ_mean", "fy_targ_std"
                    p_mean_col, p_std_col = "fy_pred_mean", "fy_pred_std"

                # Group by statistic per case: all means first, then all stds.
                x_start = i - group_width / 2.0
                truth_mean = base_summary.loc[
                    base_summary["case_id"] == case_id, t_mean_col
                ].iloc[0]
                truth_std = base_summary.loc[
                    base_summary["case_id"] == case_id, t_std_col
                ].iloc[0]
                ax.bar(
                    x_start + 0.5 * barw,
                    truth_mean,
                    width=barw,
                    color="#7f7f7f",
                    edgecolor="black",
                    linewidth=0.8,
                    label="Ground Truth mean" if i == 0 else None,
                )
                ax.bar(
                    x_start + (len(models_data) + 1 + 0.5) * barw,
                    truth_std,
                    width=barw,
                    color=_lighten_color("#7f7f7f"),
                    edgecolor="black",
                    linewidth=0.8,
                    hatch="//",
                    label="Ground Truth std" if i == 0 else None,
                )
                for midx, mdata in enumerate(models_data):
                    summ = mdata["forces_summary"].copy()
                    summ["case_id"] = summ["case_id"].astype(str)
                    summ = summ.set_index("case_id").loc[common_cases].reset_index()
                    pred_mean = summ.loc[summ["case_id"] == case_id, p_mean_col].iloc[0]
                    pred_std = summ.loc[summ["case_id"] == case_id, p_std_col].iloc[0]
                    mean_idx = 1 + midx
                    std_idx = (len(models_data) + 1) + 1 + midx
                    ax.bar(
                        x_start + (mean_idx + 0.5) * barw,
                        pred_mean,
                        width=barw,
                        color=colors[midx % len(colors)],
                        edgecolor="black",
                        linewidth=0.8,
                        label=f"{labels[midx]} mean" if i == 0 else None,
                    )
                    ax.bar(
                        x_start + (std_idx + 0.5) * barw,
                        pred_std,
                        width=barw,
                        color=_lighten_color(colors[midx % len(colors)]),
                        edgecolor="black",
                        linewidth=0.8,
                        hatch="//",
                        label=f"{labels[midx]} std" if i == 0 else None,
                    )
            ax.set_xticks(np.arange(len(common_cases)))
            ax.set_xticklabels(common_cases)
            ax.set_xlabel(r"Case ID (ascending $Re$)")
            if comp == "fx":
                ax.set_ylabel(r"$F_x$ [$N$]")
            else:
                ax.set_ylabel(r"$F_y$ [$N$]")
            ax.axhline(
                0.0,
                color="#505050",
                linewidth=0.8,
                linestyle="--",
                alpha=0.35,
                zorder=1,
            )
            _force_compact_legend(
                ax,
                labels=["Ground Truth", *labels],
                colors=[
                    "#7f7f7f",
                    *[colors[i % len(colors)] for i in range(len(labels))],
                ],
            )
            ymin, ymax = ax.get_ylim()
            if ymax > 0:
                ax.set_ylim(ymin, ymax * 1.2)
            ax.tick_params(axis="x", rotation=45)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
            _box_spines(ax)
            fig.tight_layout()
            _save(fig, out, f"compare_force_bar_{comp}")

    print(f"[compare_models] Comparison figures written to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare multiple postprocessed model outputs"
    )
    parser.add_argument("config", help="Unified JSON config file")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model output directories (output_dir/model_name)",
    )
    parser.add_argument(
        "--nicknames",
        nargs="+",
        default=None,
        help="Optional display names, one per model",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Comparison output directory"
    )
    parser.add_argument(
        "--only",
        choices=["all", "sensors", "rmse", "forces"],
        default="all",
        help="Restrict comparison plotting to one figure family",
    )
    parser.add_argument(
        "--sensor-drop-last-column",
        action="store_true",
        help="Override config and use 3x2 cropped sensor layout",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.nicknames and len(args.nicknames) != len(args.models):
        raise ValueError("--nicknames must map 1-to-1 with --models")
    run(
        args.config,
        args.models,
        args.nicknames,
        args.output_dir,
        args.only,
        args.sensor_drop_last_column,
    )


if __name__ == "__main__":
    main()
