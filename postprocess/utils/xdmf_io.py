from __future__ import annotations

import json
import os
import pickle
import re
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import meshio
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


class NodeType(IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: str | Path) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)


def normalize_case_id(name: str) -> str:
    match = re.search(r"(\d+)$", str(name))
    return match.group(1) if match else str(name)


def discover_xdmf_cases(
    folder: str | Path, prefix: Optional[str] = None
) -> Dict[str, str]:
    folder = Path(folder)
    if not folder.exists():
        return {}
    if prefix:
        files = sorted(folder.glob(f"{prefix}*.xdmf"))
    else:
        files = sorted(folder.glob("*.xdmf"))
    cases: Dict[str, str] = {}
    for fpath in files:
        stem = fpath.stem
        if prefix and stem.startswith(prefix):
            raw_case = stem[len(prefix) :]
        else:
            raw_case = stem
        cases[normalize_case_id(raw_case)] = str(fpath)
    return dict(
        sorted(cases.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])
    )


def read_xdmf_series(path: str | Path) -> Tuple[List[meshio.Mesh], np.ndarray]:
    meshes: List[meshio.Mesh] = []
    timesteps: List[float] = []
    with meshio.xdmf.TimeSeriesReader(str(path)) as reader:
        points, cells = reader.read_points_cells()
        for idx in range(reader.num_steps):
            t, point_data, _ = reader.read_data(idx)
            timesteps.append(float(t))
            meshes.append(
                meshio.Mesh(points=points, cells=cells, point_data=point_data)
            )
    return meshes, np.asarray(timesteps, dtype=float)


def write_xdmf_series(
    path: str | Path, meshes: List[meshio.Mesh], timesteps: np.ndarray
) -> None:
    if not meshes:
        raise ValueError("Cannot write an empty mesh series")
    path = Path(path).resolve()
    out_dir = path.parent
    out_name = path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    prev = Path.cwd()
    try:
        os.chdir(out_dir)
        with meshio.xdmf.TimeSeriesWriter(out_name) as writer:
            writer.write_points_cells(meshes[0].points, meshes[0].cells)
            for mesh, t in zip(meshes, timesteps):
                writer.write_data(float(t), point_data=mesh.point_data)
    finally:
        os.chdir(prev)


def feature_key_for_semantic(
    feature_map: Dict[str, str], semantic: str
) -> Optional[str]:
    for key, value in feature_map.items():
        if value == semantic:
            return key
    return None


def latest_x_feature_key(feature_map: Dict[str, str]) -> str:
    x_keys = [k for k in feature_map.keys() if k.startswith("x") and k[1:].isdigit()]
    if not x_keys:
        raise ValueError("feature_map must contain x* entries")
    return sorted(x_keys, key=lambda k: int(k[1:]))[-1]


def stacked_vector(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(vx)
    return np.column_stack([vx, vy, zeros])


def load_configs_pool(path: str | Path) -> pd.DataFrame:
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, list):
        return pd.DataFrame(data)
    raise TypeError(f"Unsupported configs_pool type: {type(data)}")


def _row_case_id(row: pd.Series) -> Optional[str]:
    for key in ["Config", "config", "case_id", "case", "id"]:
        if key in row and pd.notna(row[key]):
            return normalize_case_id(str(row[key]))
    return None


def case_row_from_configs_pool(
    configs_df: pd.DataFrame, case_id: str
) -> Optional[pd.Series]:
    target = normalize_case_id(case_id)
    for _, row in configs_df.iterrows():
        rid = _row_case_id(row)
        if rid == target:
            return row
    return None


def case_reynolds(row: pd.Series) -> float:
    if row is None:
        return float("nan")
    for key in ["Re", "re", "Reynolds", "reynolds"]:
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return float("nan")


def _scalar_from_row_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        return float(value[0])
    return float(value)


def case_diameter_proxy(row: pd.Series) -> float:
    """Return a diameter-like scalar for Reynolds sorting.

    Uses diameter directly when available, otherwise converts radius/radii to
    diameter. This follows the user's convention that Re is proportional to D.
    """
    if row is None:
        return float("nan")

    # Prefer explicit object radius from configs_pool when available.
    for key in ["radius_objects", "object_radius", "obj_radius"]:
        if key in row and pd.notna(row[key]):
            val = _scalar_from_row_value(row[key])
            if val is not None:
                return float(2.0 * val)

    for key in ["diameter", "Diameter", "D", "d", "obj_diameter"]:
        if key in row and pd.notna(row[key]):
            val = _scalar_from_row_value(row[key])
            if val is not None:
                return float(val)

    for key in ["radius", "Radius", "r", "radii", "Radii"]:
        if key in row and pd.notna(row[key]):
            val = _scalar_from_row_value(row[key])
            if val is not None:
                return float(2.0 * val)

    return float("nan")


def case_sort_key_from_configs(
    configs_df: pd.DataFrame, case_id: str
) -> Tuple[float, str]:
    row = case_row_from_configs_pool(configs_df, case_id)
    d_proxy = case_diameter_proxy(row)
    if pd.isna(d_proxy):
        d_proxy = float("inf")
    cid = normalize_case_id(case_id)
    return d_proxy, cid


def case_cylinder_geometry(row: pd.Series) -> Tuple[float, float, float]:
    cx_candidates = ["x_objects", "cx", "center_x", "x_center"]
    cy_candidates = ["y_objects", "cy", "center_y", "y_center"]
    d_candidates = ["diameter", "Diameter", "D", "d", "obj_diameter"]
    r_candidates = [
        "radius_objects",
        "object_radius",
        "obj_radius",
        "radius",
        "Radius",
        "r",
    ]

    def _pick(cands: List[str], default: float) -> float:
        for key in cands:
            if key in row and pd.notna(row[key]):
                val = row[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    if len(val) > 0:
                        return float(val[0])
                return float(val)
        return default

    cx = _pick(cx_candidates, 0.0)
    cy = _pick(cy_candidates, 0.0)
    diam = float("nan")
    for key in r_candidates:
        if key in row and pd.notna(row[key]):
            val = row[key]
            if isinstance(val, (list, tuple, np.ndarray)):
                if len(val) > 0:
                    diam = float(2.0 * val[0])
                    break
            else:
                diam = float(2.0 * val)
                break
    if pd.isna(diam):
        diam = _pick(d_candidates, 1.0)
    return cx, cy, diam


def default_sensor_offsets() -> Dict[str, Tuple[float, float]]:
    return {
        "p1": (-3.0, 0.0),
        "p2": (-1.5, 1.5),
        "p3": (-1.5, -1.5),
        "p4": (1.5, 1.5),
        "p5": (1.5, -1.5),
        "p6": (3.0, 0.0),
        "p7": (0.0, 3.0),
        "p8": (0.0, -3.0),
        "p9": (2.0, 0.0),
    }


def auto_sensor_coordinates(
    cx: float,
    cy: float,
    diameter: float,
    points: Optional[np.ndarray] = None,
) -> Dict[str, List[float]]:
    """Create 9 sensors in the wake, inspired by legacy scripts.

    Layout: 3x3 downstream grid beginning near x = cx + 1.5D and spanning
    approximately to x = cx + 3.5D, with y in [cy-1.5D, cy+1.5D].
    """
    d = max(float(diameter), 1e-6)
    x_start = cx + 1.5 * d
    x_end = cx + 3.5 * d
    y_lo = cy - 1.5 * d
    y_hi = cy + 1.5 * d

    if points is not None and len(points) > 0:
        pts2d = points[:, :2]
        x_min, y_min = np.min(pts2d, axis=0)
        x_max, y_max = np.max(pts2d, axis=0)
        eps = 0.01 * d
        x_start = float(np.clip(x_start, x_min + eps, x_max - eps))
        x_end = float(np.clip(x_end, x_min + eps, x_max - eps))
        y_lo = float(np.clip(y_lo, y_min + eps, y_max - eps))
        y_hi = float(np.clip(y_hi, y_min + eps, y_max - eps))

    xs = np.linspace(x_start, x_end, 3)
    ys = np.linspace(y_hi, y_lo, 3)

    sensors: Dict[str, List[float]] = {}
    k = 1
    for y in ys:
        for x in xs:
            sensors[f"p{k}"] = [float(x), float(y), 0.0]
            k += 1
    return sensors


def nearest_node_indices(
    points: np.ndarray, coords: Dict[str, List[float]]
) -> Dict[str, int]:
    pts2d = points[:, :2] if points.shape[1] >= 2 else points
    tree = cKDTree(pts2d)
    out: Dict[str, int] = {}
    for sid, xyz in coords.items():
        _, idx = tree.query(np.asarray(xyz[:2], dtype=float))
        out[sid] = int(idx)
    return out


def read_case_levelset_from_dataset(
    dataset_dir: str | Path, case_id: str
) -> np.ndarray:
    target = normalize_case_id(case_id)
    for xdmf in sorted(Path(dataset_dir).glob("*.xdmf")):
        if normalize_case_id(xdmf.stem) != target:
            continue
        meshes, _ = read_xdmf_series(xdmf)
        if not meshes:
            break
        pdata = meshes[0].point_data
        if "LevelSetObject" in pdata:
            return np.asarray(pdata["LevelSetObject"]).reshape(-1)
        if "levelset" in pdata:
            return np.asarray(pdata["levelset"]).reshape(-1)
    raise FileNotFoundError(
        f"Could not find levelset for case {case_id} under dataset_dir={dataset_dir}"
    )


def crop_rollout(
    meshes: List[meshio.Mesh], timesteps: np.ndarray, rollout_steps: Optional[int]
) -> Tuple[List[meshio.Mesh], np.ndarray]:
    if rollout_steps is None:
        return meshes, timesteps
    n = max(0, min(int(rollout_steps), len(meshes)))
    return meshes[:n], timesteps[:n]


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def iter_force_csvs(forces_dir: str | Path) -> Iterable[Path]:
    for csv_path in sorted(Path(forces_dir).glob("forces_*.csv")):
        yield csv_path
