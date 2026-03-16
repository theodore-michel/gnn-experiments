from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from postprocess.utils.xdmf_io import (
    discover_xdmf_cases,
    ensure_dir,
    load_json,
    read_xdmf_series,
)


def _rmse(pred: np.ndarray, targ: np.ndarray) -> np.ndarray:
    diff = pred - targ
    return np.sqrt(np.mean(diff * diff, axis=1))


def _cumulative_sum(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    return np.cumsum(values)


def run(config_path: str) -> None:
    cfg = load_json(config_path)
    model_name = cfg["model_name"]
    model_root = Path(cfg["output_dir"]) / model_name
    xdmf_dir = model_root / "xdmf"
    errors_dir = Path(ensure_dir(model_root / "errors"))

    cases = discover_xdmf_cases(xdmf_dir)
    if not cases:
        raise FileNotFoundError(f"No postprocessed XDMFs found in {xdmf_dir}")

    per_case_rows: List[Dict[str, float]] = []
    all_cumulative_rows: List[pd.DataFrame] = []

    for case_id, xdmf_path in tqdm(cases.items(), total=len(cases), desc="RMSE cases"):
        meshes, _ = read_xdmf_series(xdmf_path)
        if not meshes:
            continue

        vpred = np.stack(
            [np.asarray(m.point_data["v_pred"])[:, :2] for m in meshes], axis=0
        )
        vtarg = np.stack(
            [np.asarray(m.point_data["v_targ"])[:, :2] for m in meshes], axis=0
        )
        ppred = np.stack(
            [np.asarray(m.point_data["p"]).reshape(-1) for m in meshes], axis=0
        )
        ptarg = np.stack(
            [np.asarray(m.point_data["p_targ"]).reshape(-1) for m in meshes], axis=0
        )

        rmse_vel = _rmse(vpred.reshape(len(meshes), -1), vtarg.reshape(len(meshes), -1))
        rmse_pres = _rmse(ppred, ptarg)

        total_pred = np.concatenate([vpred, ppred[:, :, None]], axis=2)
        total_targ = np.concatenate([vtarg, ptarg[:, :, None]], axis=2)
        rmse_total = _rmse(
            total_pred.reshape(len(meshes), -1), total_targ.reshape(len(meshes), -1)
        )

        cum_vel = _cumulative_sum(rmse_vel)
        cum_pres = _cumulative_sum(rmse_pres)
        cum_total = _cumulative_sum(rmse_total)

        case_curve = pd.DataFrame(
            {
                "timestep": np.arange(len(meshes), dtype=int),
                "cum_rmse_total": cum_total,
                "cum_rmse_vel": cum_vel,
                "cum_rmse_pres": cum_pres,
            }
        )
        case_curve.to_csv(errors_dir / f"cumulative_rmse_{case_id}.csv", index=False)
        case_curve.insert(0, "case_id", case_id)
        all_cumulative_rows.append(case_curve)

        per_case_rows.append(
            {
                "model_name": model_name,
                "case_id": case_id,
                "rmse_total_mean": float(np.mean(rmse_total)),
                "rmse_vel_mean": float(np.mean(rmse_vel)),
                "rmse_pres_mean": float(np.mean(rmse_pres)),
                "rmse_total_std": float(np.std(rmse_total)),
                "rmse_step1": float(rmse_total[0]),
                "rmse_50step": float(np.mean(rmse_total[: min(50, len(rmse_total))])),
            }
        )

    per_case_df = pd.DataFrame(per_case_rows).sort_values("case_id")
    per_case_df.to_csv(errors_dir / "per_case_rmse.csv", index=False)

    all_cum_df = pd.concat(all_cumulative_rows, ignore_index=True)
    mean_df = (
        all_cum_df.groupby("timestep")
        .agg(
            cum_rmse_total_mean=("cum_rmse_total", "mean"),
            cum_rmse_total_std=("cum_rmse_total", "std"),
            cum_rmse_vel_mean=("cum_rmse_vel", "mean"),
            cum_rmse_vel_std=("cum_rmse_vel", "std"),
            cum_rmse_pres_mean=("cum_rmse_pres", "mean"),
            cum_rmse_pres_std=("cum_rmse_pres", "std"),
        )
        .reset_index()
    )
    mean_df = mean_df.fillna(0.0)
    mean_df.to_csv(errors_dir / "cumulative_rmse_mean.csv", index=False)

    stats = pd.DataFrame(
        [
            {"Statistic": "Number of Cases", "Value": len(per_case_df)},
            {
                "Statistic": "RMSE Total Mean",
                "Value": per_case_df["rmse_total_mean"].mean(),
            },
            {
                "Statistic": "RMSE Total Std",
                "Value": per_case_df["rmse_total_mean"].std(),
            },
            {
                "Statistic": "RMSE Velocity Mean",
                "Value": per_case_df["rmse_vel_mean"].mean(),
            },
            {
                "Statistic": "RMSE Pressure Mean",
                "Value": per_case_df["rmse_pres_mean"].mean(),
            },
            {
                "Statistic": "RMSE Step-1 Mean",
                "Value": per_case_df["rmse_step1"].mean(),
            },
            {
                "Statistic": "RMSE First-50 Mean",
                "Value": per_case_df["rmse_50step"].mean(),
            },
        ]
    )
    stats.to_csv(errors_dir / "summary_statistics.csv", index=False)

    print(f"[compute_errors] Results written to {errors_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute RMSE metrics from postprocessed XDMFs"
    )
    parser.add_argument("config", help="Unified JSON config file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
