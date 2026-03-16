from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from postprocess.plot_results import PLOT_CONFIG
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
) -> None:
    cfg = load_json(config_path)
    plt.rcParams.update({"font.family": PLOT_CONFIG["font_family"]})

    model_paths = [Path(p) for p in model_dirs]
    labels = [
        _model_label(p, nicknames[i] if nicknames else None)
        for i, p in enumerate(model_paths)
    ]
    colors = PLOT_CONFIG["palette"]

    models_data = [_load_model_outputs(p) for p in model_paths]
    common_cases = _assert_same_cases(models_data, labels)

    out = Path(output_dir)
    cfg_pool = load_configs_pool(cfg["configs_pool"])
    common_cases = sorted(
        common_cases,
        key=lambda cid: (
            case_sort_key_from_configs(cfg_pool, cid)[0],
            int(cid) if str(cid).isdigit() else str(cid),
        ),
    )

    for case_id in common_cases:
        fig, axs = plt.subplots(3, 3, figsize=(20, 10), squeeze=False)
        base = models_data[0]["sensor"]
        dcase_base = base[base["case_id"].astype(str) == case_id]
        sensors = sorted(dcase_base["sensor_id"].astype(str).unique())[:9]
        for i, sid in enumerate(sensors):
            ax = axs.flatten()[i]
            gt = dcase_base[dcase_base["sensor_id"].astype(str) == sid].sort_values(
                "timestep"
            )
            ax.plot(
                gt["timestep"],
                gt["v_targ"],
                color="black",
                linestyle=(0, (4, 1)),
                linewidth=1.75,
                label="Ground Truth",
            )
            for midx, mdata in enumerate(models_data):
                ds = mdata["sensor"]
                s = ds[
                    (ds["case_id"].astype(str) == case_id)
                    & (ds["sensor_id"].astype(str) == sid)
                ].sort_values("timestep")
                ax.plot(
                    s["timestep"],
                    s["v_pred"],
                    color=colors[midx % len(colors)],
                    linewidth=2.0,
                    alpha=0.8,
                    label=labels[midx],
                )
            ax.set_title(sid)
            ax.set_ylim(*PLOT_CONFIG["sensor_ylim_velocity"])
            row_i = i // 3
            col_i = i % 3
            if row_i == 1 and col_i == 0:
                ax.set_ylabel("velocity")
            else:
                ax.set_ylabel("")
            if row_i == 2 and col_i == 1:
                ax.set_xlabel("Rollout step")
            else:
                ax.set_xlabel("")
            if col_i > 0:
                ax.tick_params(axis="y", labelleft=False)
            if row_i < 2:
                ax.tick_params(axis="x", labelbottom=False)
            _box_spines(ax)
            if i == 0:
                ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        _save(fig, out, f"compare_sensor_velocity_{case_id}")

    fig, ax = plt.subplots(figsize=(13, 7))
    for midx, mdata in enumerate(models_data):
        d = mdata["cum"]
        x = d["timestep"].to_numpy()
        m = d["cum_rmse_total_mean"].to_numpy()
        s = d["cum_rmse_total_std"].to_numpy()
        ax.plot(
            x, m, color=colors[midx % len(colors)], linewidth=2.2, label=labels[midx]
        )
        ax.fill_between(x, m - s, m + s, color=colors[midx % len(colors)], alpha=0.25)
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Cumulative RMSE total")
    ax.legend(loc="upper left")
    _box_spines(ax)
    fig.tight_layout()
    _save(fig, out, "compare_cumulative_rmse_total")

    for case_id in common_cases:
        fig, ax = plt.subplots(figsize=(13, 7))
        for midx, model_path in enumerate(model_paths):
            cfile = model_path / "errors" / f"cumulative_rmse_{case_id}.csv"
            d = pd.read_csv(cfile)
            ax.plot(
                d["timestep"],
                d["cum_rmse_total"],
                color=colors[midx % len(colors)],
                linewidth=2.0,
                label=labels[midx],
            )
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("Cumulative RMSE total")
        ax.legend(loc="upper left")
        _box_spines(ax)
        fig.tight_layout()
        _save(fig, out, f"compare_cumulative_rmse_case_{case_id}")

    x = np.arange(len(common_cases), dtype=float)
    width = 0.8 / max(1, len(models_data))
    fig, ax = plt.subplots(figsize=(14, 7))
    for midx, mdata in enumerate(models_data):
        rmse = mdata["rmse"].copy()
        rmse["case_id"] = rmse["case_id"].astype(str)
        rmse = rmse.set_index("case_id").loc[common_cases].reset_index()
        ax.bar(
            x - 0.4 + (midx + 0.5) * width,
            rmse["rmse_total_mean"],
            width=width,
            color=colors[midx % len(colors)],
            label=labels[midx],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(common_cases)
    ax.set_xlabel("Case ID")
    ax.set_ylabel("RMSE total mean")
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    _box_spines(ax)
    fig.tight_layout()
    _save(fig, out, "compare_rmse_bars")

    for comp in ["fx", "fy"]:
        fig, ax = plt.subplots(figsize=(14, 7))
        base_summary = models_data[0]["forces_summary"].copy()
        base_summary["case_id"] = base_summary["case_id"].astype(str)
        base_summary = base_summary.set_index("case_id").loc[common_cases].reset_index()

        group_width = 0.9
        barw = group_width / (len(models_data) + 1)
        for i, case_id in enumerate(common_cases):
            truth_mean = base_summary.loc[
                base_summary["case_id"] == case_id, f"{comp}_targ_mean"
            ].iloc[0]
            truth_std = base_summary.loc[
                base_summary["case_id"] == case_id, f"{comp}_targ_std"
            ].iloc[0]
            ax.bar(
                i - group_width / 2 + barw * 0.5,
                truth_mean,
                yerr=truth_std,
                width=barw,
                color="#7f7f7f",
                label="Ground Truth" if i == 0 else None,
            )
            for midx, mdata in enumerate(models_data):
                summ = mdata["forces_summary"].copy()
                summ["case_id"] = summ["case_id"].astype(str)
                summ = summ.set_index("case_id").loc[common_cases].reset_index()
                pm = summ.loc[summ["case_id"] == case_id, f"{comp}_pred_mean"].iloc[0]
                ps = summ.loc[summ["case_id"] == case_id, f"{comp}_pred_std"].iloc[0]
                xpos = i - group_width / 2 + barw * (midx + 1 + 0.5)
                ax.bar(
                    xpos,
                    pm,
                    yerr=ps,
                    width=barw,
                    color=colors[midx % len(colors)],
                    label=labels[midx] if i == 0 else None,
                )
        ax.set_xticks(np.arange(len(common_cases)))
        ax.set_xticklabels(common_cases)
        ax.set_xlabel("Case ID")
        ax.set_ylabel(comp)
        ax.legend(loc="upper right")
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.nicknames and len(args.nicknames) != len(args.models):
        raise ValueError("--nicknames must map 1-to-1 with --models")
    run(args.config, args.models, args.nicknames, args.output_dir)


if __name__ == "__main__":
    main()
