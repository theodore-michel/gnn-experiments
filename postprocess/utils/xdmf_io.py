"""
xdmf_io.py — XDMF / mesh I/O utilities for GNN prediction postprocessing.

Provides helpers for:
  * Reading XDMF time-series into a list of meshio.Mesh objects.
  * Writing a list of meshio.Mesh objects back to XDMF.
  * Gathering prediction cases from a directory tree.
  * Finding nearest mesh nodes for sensor / line coordinates.
  * Extracting point-data time-series at sensor locations.
  * Creating auto-sensor grids and line discretisations.
  * NodeType enum matching the GNN training convention.

All heavy lifting is done through *meshio*; the module has no dependency on
PyTorch or PyG so it can be used in lightweight post-processing environments.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import meshio
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# NodeType enum — mirrors graphphysics.dataset.nodetype
# ---------------------------------------------------------------------------
class NodeType(IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6


# ---------------------------------------------------------------------------
# XDMF read / write
# ---------------------------------------------------------------------------


def xdmf_to_meshes(
    xdmf_file_path: str | Path,
    verbose: bool = False,
) -> Tuple[List[meshio.Mesh], np.ndarray]:
    """Read an XDMF time-series file and return (meshes, timesteps).

    Each element of *meshes* is a ``meshio.Mesh`` sharing the same topology
    but carrying the point-data for a single timestep.

    Parameters
    ----------
    xdmf_file_path : str | Path
        Path to the ``.xdmf`` file.
    verbose : bool
        Print progress information.

    Returns
    -------
    meshes : list[meshio.Mesh]
        One mesh per timestep.
    timesteps : np.ndarray
        1-D array of time stamps read from the file.
    """
    xdmf_file_path = str(xdmf_file_path)
    meshes: List[meshio.Mesh] = []
    timesteps: List[float] = []

    with meshio.xdmf.TimeSeriesReader(xdmf_file_path) as reader:
        points, cells = reader.read_points_cells()
        for k in range(reader.num_steps):
            t, point_data, _ = reader.read_data(k)
            timesteps.append(t)
            meshes.append(
                meshio.Mesh(points=points, cells=cells, point_data=point_data)
            )

    timesteps_arr = np.array(timesteps)
    if verbose:
        print(
            f"[xdmf_io] Read {len(meshes)} timesteps from {xdmf_file_path} "
            f"(t={timesteps_arr[0]:.4g} … {timesteps_arr[-1]:.4g})"
        )
    return meshes, timesteps_arr


def meshes_to_xdmf(
    filename: str | Path,
    meshes: List[meshio.Mesh],
    timesteps: np.ndarray | List[float],
    drop_first: bool = False,
    verbose: bool = False,
) -> None:
    """Write a list of meshes to an XDMF time-series file.

    Parameters
    ----------
    filename : str | Path
        Output ``.xdmf`` path (companion ``.h5`` written automatically).
    meshes : list[meshio.Mesh]
        One mesh per timestep, all sharing the same topology.
    timesteps : array-like
        Corresponding time stamp for each mesh.
    drop_first : bool
        If *True*, skip the first mesh/timestep (useful when it is an
        initial-condition duplicate).
    verbose : bool
        Print a summary line.
    """
    filename = os.path.abspath(str(filename))
    out_dir = os.path.dirname(filename) or "."
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.basename(filename)

    start = 1 if drop_first else 0
    prev_cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        with meshio.xdmf.TimeSeriesWriter(out_name) as writer:
            writer.write_points_cells(meshes[0].points, meshes[0].cells)
            for k in range(start, len(meshes)):
                writer.write_data(float(timesteps[k]), point_data=meshes[k].point_data)
    finally:
        os.chdir(prev_cwd)
    if verbose:
        print(f"[xdmf_io] Wrote {len(meshes) - start} timesteps to {filename}")


# ---------------------------------------------------------------------------
# Case gathering
# ---------------------------------------------------------------------------


def gather_cases(
    case_folder: str | Path,
    case_base_name: str,
    extension: str = ".xdmf",
) -> Dict[str, str]:
    """Scan *case_folder* for XDMF files matching ``<case_base_name><id>.xdmf``.

    Returns a dict ``{case_id: full_path}``, sorted by case id.

    Example
    -------
    If ``case_folder`` contains ``rollout_config_001.xdmf``, ``rollout_config_002.xdmf``
    and ``case_base_name="rollout_"`` then the returned dict is
    ``{"config_001": "/…/rollout_config_001.xdmf", "config_002": "…"}``.
    """
    case_folder = str(case_folder)
    pattern = os.path.join(case_folder, f"{case_base_name}*{extension}")
    files = sorted(glob.glob(pattern))
    cases: Dict[str, str] = {}
    for f in files:
        basename = os.path.basename(f).replace(extension, "")
        case_id = basename[len(case_base_name) :]
        cases[case_id] = f
    return cases


def load_configs_pool(path: str | Path) -> pd.DataFrame:
    """Load the pickled configs pool (geometry metadata per case)."""
    with open(str(path), "rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# Nearest-node sensor lookup
# ---------------------------------------------------------------------------


def build_kdtree(points: np.ndarray) -> cKDTree:
    """Build a *scipy* cKDTree from mesh node coordinates."""
    return cKDTree(points[:, :2] if points.shape[1] == 3 else points)


def nearest_node_indices(
    tree: cKDTree,
    coords: Dict[str, List[float]],
) -> Dict[str, int]:
    """For each named coordinate, find the index of the nearest mesh node.

    Parameters
    ----------
    tree : cKDTree
        Built from mesh node positions.
    coords : dict
        ``{name: [x, y]}`` or ``{name: [x, y, z]}``.

    Returns
    -------
    dict
        ``{name: node_index}``.
    """
    indices: Dict[str, int] = {}
    for name, xy in coords.items():
        _, idx = tree.query(np.array(xy[:2]))
        indices[name] = int(idx)
    return indices


# ---------------------------------------------------------------------------
# Point-data extraction (single model)
# ---------------------------------------------------------------------------


def extract_point_values(
    meshes: List[meshio.Mesh],
    coords_dict: Dict[str, List[float]],
    fields: List[str],
) -> Dict[str, Dict[str, List[float]]]:
    """Extract field time-series at specified coordinates.

    Parameters
    ----------
    meshes : list[meshio.Mesh]
        One mesh per timestep (same topology).
    coords_dict : dict
        ``{sensor_name: [x, y(, z)]}``.
    fields : list[str]
        Field names present in ``mesh.point_data``.

    Returns
    -------
    dict
        ``{sensor_name: {field_name: [value_t0, value_t1, …]}}``.
    """
    tree = build_kdtree(meshes[0].points)
    node_indices = nearest_node_indices(tree, coords_dict)

    result: Dict[str, Dict[str, List[float]]] = {}
    for sensor_name, idx in node_indices.items():
        result[sensor_name] = {}
        for field in fields:
            values = []
            for mesh in meshes:
                if field not in mesh.point_data:
                    raise KeyError(
                        f"Field '{field}' not in mesh point_data at sensor '{sensor_name}'"
                    )
                val = mesh.point_data[field][idx]
                values.append(
                    float(val) if np.ndim(val) == 0 else float(np.linalg.norm(val))
                )
            result[sensor_name][field] = values
    return result


def extract_point_values_multi(
    model_meshes: Dict[str, List[meshio.Mesh]],
    coords_dict: Dict[str, List[float]],
    fields: List[str],
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """Extract field time-series at specified coordinates for multiple models.

    Parameters
    ----------
    model_meshes : dict
        ``{model_name: list_of_meshes}``.
    coords_dict : dict
        ``{sensor_name: [x, y(, z)]}``.
    fields : list[str]
        Field names present in ``mesh.point_data``.

    Returns
    -------
    dict
        ``{sensor_name: {model_name: {field_name: [values…]}}}``.
    """
    result: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for model_name, meshes in model_meshes.items():
        single = extract_point_values(meshes, coords_dict, fields)
        for sensor_name, field_dict in single.items():
            result.setdefault(sensor_name, {})[model_name] = field_dict
    return result


# ---------------------------------------------------------------------------
# Auto-sensor / line placement
# ---------------------------------------------------------------------------


def create_auto_sensor_location(
    domain_dim_dict: Dict[str, float],
    num_sensors: int = 9,
    object_center: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    """Create a grid of sensor locations in the wake region downstream of the object.

    Sensors are placed in a num_rows × num_cols grid that spans from the
    object centre to roughly half-way downstream, and ±1 diameter vertically.

    Parameters
    ----------
    domain_dim_dict : dict
        Must contain ``x_min``, ``dx``, ``y_min``, ``dy``.
    num_sensors : int
        Approximate target count (rounded to a square grid).
    object_center : list | None
        ``[x, y, z]`` of the object centre.  Defaults to domain centre.

    Returns
    -------
    dict
        ``{name: [x, y, 0]}`` where *name* encodes row/column (e.g. ``"S01"``).
    """
    x_min = domain_dim_dict["x_min"]
    dx = domain_dim_dict["dx"]
    y_min = domain_dim_dict["y_min"]
    dy = domain_dim_dict["dy"]

    if object_center is None:
        cx, cy = x_min + dx / 2, y_min + dy / 2
    else:
        cx, cy = object_center[0], object_center[1]

    # wake region: from object to 60 % downstream, ±30 % of dy centred on object
    x_start = cx + 0.5
    x_end = min(cx + 0.5 * dx, x_min + dx - 0.5)
    y_lo = max(cy - 0.3 * dy, y_min + 0.1)
    y_hi = min(cy + 0.3 * dy, y_min + dy - 0.1)

    ncols = max(int(np.ceil(np.sqrt(num_sensors))), 2)
    nrows = max(int(np.ceil(num_sensors / ncols)), 2)

    xs = np.linspace(x_start, x_end, ncols)
    ys = np.linspace(y_lo, y_hi, nrows)

    sensors: Dict[str, List[float]] = {}
    k = 0
    for yy in ys:
        for xx in xs:
            sensors[f"S{k:02d}"] = [float(xx), float(yy), 0.0]
            k += 1
    return sensors


def create_auto_line_location(
    object_center: List[float],
    wake_margin: float = 1.0,
) -> Dict[str, List[float]]:
    """Return canonical x-line and y-line origins for profile extraction.

    Parameters
    ----------
    object_center : list
        ``[x, y, z]`` of the object centre.
    wake_margin : float
        Downstream offset for the y-line from the object centre.

    Returns
    -------
    dict
        ``{"x_line": [x0, y0, 0], "y_line": [x0, y0, 0]}``.
    """
    cx, cy = object_center[0], object_center[1]
    return {
        "x_line": [cx, cy, 0.0],
        "y_line": [cx + wake_margin, cy, 0.0],
    }


def create_line_points_dict(
    line_axis: str,
    line_origins: List[float],
    domain_dim_dict: Dict[str, float],
    num_points: int = 200,
) -> Tuple[Dict[str, List[float]], np.ndarray]:
    """Discretise a line (x or y) across the domain for profile extraction.

    Parameters
    ----------
    line_axis : str
        ``"x"`` or ``"y"``.
    line_origins : list
        ``[x0, y0, z0]`` — the line passes through this point.
    domain_dim_dict : dict
        Must contain ``x_min``, ``dx``, ``y_min``, ``dy``.
    num_points : int
        Number of sample points along the line.

    Returns
    -------
    points_dict : dict
        ``{point_name: [x, y, z]}``.
    axis_values : np.ndarray
        The varying coordinate values (for plotting).
    """
    x_min, dx = domain_dim_dict["x_min"], domain_dim_dict["dx"]
    y_min, dy = domain_dim_dict["y_min"], domain_dim_dict["dy"]

    if line_axis == "x":
        axis_values = np.linspace(x_min, x_min + dx, num_points)
        points = {
            f"L{i:04d}": [float(v), line_origins[1], 0.0]
            for i, v in enumerate(axis_values)
        }
    elif line_axis == "y":
        axis_values = np.linspace(y_min, y_min + dy, num_points)
        points = {
            f"L{i:04d}": [line_origins[0], float(v), 0.0]
            for i, v in enumerate(axis_values)
        }
    else:
        raise ValueError(f"line_axis must be 'x' or 'y', got '{line_axis}'")
    return points, axis_values


# ---------------------------------------------------------------------------
# Field renaming / vector-field creation helpers
# ---------------------------------------------------------------------------


def rename_fields(
    meshes: List[meshio.Mesh],
    field_map: Dict[str, str],
) -> List[meshio.Mesh]:
    """Rename point-data fields in-place according to *field_map* (old→new)."""
    for mesh in meshes:
        for old, new in field_map.items():
            if old in mesh.point_data:
                mesh.point_data[new] = mesh.point_data.pop(old)
    return meshes


def create_vector_field(
    mesh: meshio.Mesh,
    field_name: str,
    field_scalars: List[str],
    fill: bool = True,
) -> meshio.Mesh:
    """Combine scalar components into a vector point-data field.

    Parameters
    ----------
    mesh : meshio.Mesh
    field_name : str
        Name for the resulting vector field (e.g. ``"V_vect_pred"``).
    field_scalars : list[str]
        Ordered component field names, e.g. ``["V_x_pred", "V_y_pred"]``.
    fill : bool
        If *True* and fewer than 3 components are given, pad with zeros
        to produce a 3-component vector (for ParaView compatibility).
    """
    components = [mesh.point_data[s] for s in field_scalars]
    n = len(components[0])
    if fill and len(components) < 3:
        components.append(np.zeros(n))
    mesh.point_data[field_name] = np.column_stack(components)
    return mesh


def create_norm_field(
    mesh: meshio.Mesh,
    field_name: str,
    components: List[str],
) -> meshio.Mesh:
    """Create a scalar norm field from vector components."""
    arrays = [mesh.point_data[c] for c in components]
    mesh.point_data[field_name] = np.linalg.norm(np.column_stack(arrays), axis=1)
    return mesh


# ---------------------------------------------------------------------------
# Serialisation helpers (numpy → JSON-safe)
# ---------------------------------------------------------------------------


def convert_np(obj: Any) -> Any:
    """Recursively convert numpy types to Python builtins for JSON serialisation."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_np(v) for v in obj]
    return obj


def save_sensor_data(
    data_dict: Dict[str, Any],
    output_location: str,
    suffix: Optional[str] = None,
    fmt: str = "json",
    verbose: bool = False,
) -> str:
    """Persist sensor time-series data to JSON or CSV.

    Returns the written file path.
    """
    os.makedirs(output_location, exist_ok=True)
    tag = f"_{suffix}" if suffix else ""
    out_path = os.path.join(output_location, f"sensor_data{tag}.{fmt}")
    if fmt == "json":
        with open(out_path, "w") as f:
            json.dump(convert_np(data_dict), f, indent=2)
    elif fmt == "csv":
        df = pd.DataFrame.from_dict(data_dict, orient="index")
        df.to_csv(out_path, index_label="Sensor")
    else:
        raise ValueError(f"Unsupported format '{fmt}'")
    if verbose:
        print(f"[xdmf_io] Saved sensor data → {out_path}")
    return out_path


def load_sensor_data(
    location: str,
    suffix: Optional[str] = None,
    fmt: str = "json",
) -> Dict[str, Any]:
    """Load sensor time-series data from JSON or CSV."""
    tag = f"_{suffix}" if suffix else ""
    path = os.path.join(location, f"sensor_data{tag}.{fmt}")
    if fmt == "json":
        with open(path) as f:
            return json.load(f)
    elif fmt == "csv":
        df = pd.read_csv(path, index_col="Sensor")
        return df.to_dict(orient="index")
    raise ValueError(f"Unsupported format '{fmt}'")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON configuration file."""
    with open(str(path)) as f:
        return json.load(f)
