from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from postprocess.utils.xdmf_io import (
    auto_sensor_coordinates,
    case_cylinder_geometry,
    case_row_from_configs_pool,
    discover_xdmf_cases,
    ensure_dir,
    load_configs_pool,
    load_json,
    nearest_node_indices,
    read_xdmf_series,
)

# Sensor offsets in units of cylinder diameter D, relative to (cx, cy)
# p1 (-3D,0), p2 (-1.5D,1.5D), p3 (-1.5D,-1.5D), p4 (1.5D,1.5D),
# p5 (1.5D,-1.5D), p6 (3D,0), p7 (0,3D), p8 (0,-3D), p9 (2D,0)


def _sensor_coords_from_csv(path: str) -> Dict[str, List[float]]:
    df = pd.read_csv(path)
    required = {"sensor_id", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"sensor_csv must contain columns {required}")
    out: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        out[str(row["sensor_id"])] = [float(row["x"]), float(row["y"]), 0.0]
    return out


def run(config_path: str) -> None:
    cfg = load_json(config_path)
    model_name = cfg["model_name"]
    output_dir = Path(cfg["output_dir"]) / model_name
    xdmf_dir = output_dir / "xdmf"
    sensors_dir = Path(ensure_dir(output_dir / "sensors"))

    mode = cfg.get("sensor_mode", "auto")
    sensor_csv = cfg.get("sensor_csv")

    configs_df = load_configs_pool(cfg["configs_pool"])
    cases = discover_xdmf_cases(xdmf_dir)
    rows = []

    for case_id, xdmf_path in tqdm(
        cases.items(), total=len(cases), desc="Sensor cases"
    ):
        meshes, _ = read_xdmf_series(xdmf_path)
        if not meshes:
            continue

        cx = 0.0
        cy = 0.0
        diam = 1.0

        if mode == "csv":
            if not sensor_csv:
                raise ValueError("sensor_mode=csv requires sensor_csv in config")
            sensors = _sensor_coords_from_csv(sensor_csv)
        else:
            row = case_row_from_configs_pool(configs_df, case_id)
            if row is None:
                raise KeyError(f"Case {case_id} not found in configs_pool")
            cx, cy, diam = case_cylinder_geometry(row)
            sensors = auto_sensor_coordinates(cx, cy, diam, points=meshes[0].points)

        mapping_file = sensors_dir / f"sensor_points_{case_id}.json"
        if mapping_file.exists():
            with open(mapping_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            sensors_meta = payload.get("sensors", {})
            node_map = {k: int(v["node_id"]) for k, v in sensors_meta.items()}
        else:
            node_map = nearest_node_indices(meshes[0].points, sensors)
            sensors_meta = {
                sid: {
                    "node_id": int(node_idx),
                    "x": float(sensors[sid][0]),
                    "y": float(sensors[sid][1]),
                    "z": float(sensors[sid][2]),
                }
                for sid, node_idx in node_map.items()
            }
            payload = {
                "case_id": str(case_id),
                "cylinder_center": {"x": float(cx), "y": float(cy)},
                "cylinder_diameter": float(diam),
                "sensors": sensors_meta,
            }
            with open(mapping_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)

            meta_rows = []
            for sid, sinfo in sensors_meta.items():
                meta_rows.append(
                    {
                        "case_id": str(case_id),
                        "sensor_id": sid,
                        "node_id": int(sinfo["node_id"]),
                        "x": float(sinfo["x"]),
                        "y": float(sinfo["y"]),
                        "cylinder_center_x": float(cx),
                        "cylinder_center_y": float(cy),
                        "cylinder_diameter": float(diam),
                    }
                )
            pd.DataFrame(meta_rows).to_csv(
                sensors_dir / f"sensor_points_{case_id}.csv", index=False
            )

        for t, mesh in tqdm(
            enumerate(meshes),
            total=len(meshes),
            desc=f"Timesteps {case_id}",
            leave=False,
        ):
            vpred = np.asarray(mesh.point_data["v_pred"])[:, :2]
            vtarg = np.asarray(mesh.point_data["v_targ"])[:, :2]
            ppred = np.asarray(mesh.point_data["p"]).reshape(-1)
            ptarg = np.asarray(mesh.point_data["p_targ"]).reshape(-1)

            for sid, node_idx in node_map.items():
                rows.append(
                    {
                        "model_name": model_name,
                        "case_id": case_id,
                        "sensor_id": sid,
                        "timestep": int(t),
                        "v_targ": float(np.linalg.norm(vtarg[node_idx])),
                        "v_pred": float(np.linalg.norm(vpred[node_idx])),
                        "p_targ": float(ptarg[node_idx]),
                        "p_pred": float(ppred[node_idx]),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(sensors_dir / "sensor_data.csv", index=False)
    print(
        f"[extract_sensors] Saved {len(df)} rows to {sensors_dir / 'sensor_data.csv'}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract sensor signals from postprocessed XDMF"
    )
    parser.add_argument("config", help="Unified JSON config file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
