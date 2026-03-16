"""
compute_forces.py — Drag / lift computation from GNN prediction XDMFs.

Extracts boundary (obstacle) edges, computes pressure and viscous tractions,
and integrates to give drag and lift coefficients (or raw forces) per timestep.

The algorithm mirrors the reference implementation in ``scripts/forces.py``:

1. Identify obstacle boundary edges via ``nodetype == OBSTACLE (1)``.
2. Compute triangle (P1) gradients of velocity on every element.
3. Build a nodal gradient via area-weighted averaging of element gradients.
4. At each boundary edge, assemble pressure + viscous contributions using the
   outward normal (oriented via the levelset gradient).
5. Integrate (sum) over all boundary edges.

Outputs per-case CSV files (columns: ``time, drag, lift``), plus a comparison
summary across models.

Usage
-----
::

    python -m postprocess.metrics.compute_forces \\
        -p config.json -d ./force_results

"""

from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import meshio
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ..utils.xdmf_io import (
        NodeType,
        gather_cases,
        load_json,
        xdmf_to_meshes,
    )
except ImportError:
    from postprocess.utils.xdmf_io import (
        NodeType,
        gather_cases,
        load_json,
        xdmf_to_meshes,
    )


# ============================================================================
# Geometry helpers
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
) -> Dict[str, str]:
    """Discover XDMF cases with preferred/fallback base names and auto fallback."""
    cases = gather_cases(case_folder, preferred_base_name)
    if cases:
        return cases

    if fallback_base_name and fallback_base_name != preferred_base_name:
        cases = gather_cases(case_folder, fallback_base_name)
        if cases:
            return cases

    any_xdmf = sorted(Path(case_folder).glob("*.xdmf"))
    return {p.stem: str(p) for p in any_xdmf}


def _to_physical_time(timesteps: np.ndarray, dt: float) -> np.ndarray:
    """Convert index-like XDMF timesteps to physical time using dt."""
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


def _inject_truth_alias_fields(mesh: meshio.Mesh) -> None:
    """Inject common truth-field aliases expected by default feature_map.

    Some truth datasets use French / legacy names like ``Vitesse`` and
    ``Pression``. This helper exposes compatible scalar aliases (x0, x1, x2,
    x3, x6) only when they are missing.
    """
    pdict = mesh.point_data

    if "x0" not in pdict or "x1" not in pdict:
        if "Vitesse" in pdict:
            velocity = np.asarray(pdict["Vitesse"])
            if velocity.ndim == 2 and velocity.shape[1] >= 2:
                pdict.setdefault("x0", velocity[:, 0])
                pdict.setdefault("x1", velocity[:, 1])

    if "x2" not in pdict and "Pression" in pdict:
        pressure = np.asarray(pdict["Pression"])
        pdict.setdefault("x2", pressure.reshape(-1))

    if "x3" not in pdict and "LevelSetObject" in pdict:
        levelset = np.asarray(pdict["LevelSetObject"])
        pdict.setdefault("x3", levelset.reshape(-1))

    if "x6" not in pdict and "NodeType" in pdict:
        nodetype = np.asarray(pdict["NodeType"])
        pdict.setdefault("x6", nodetype.reshape(-1))


def _normalize_case_id(case_id: str) -> str:
    """Normalize case ids across naming conventions.

    Examples:
        graph_10000 -> 10000
        cylinders_10000 -> 10000
        10000 -> 10000
    """
    match = re.search(r"(\d+)$", str(case_id))
    return match.group(1) if match else str(case_id)


def _get_triangles(mesh: meshio.Mesh) -> np.ndarray:
    """Return the first triangle cell block as an (E, 3) int array."""
    for block in mesh.cells:
        if block.type == "triangle":
            return block.data
    raise ValueError("No triangle cells found in mesh")


def compute_boundary_edges(
    triangles: np.ndarray,
    nodetype: np.ndarray,
    obstacle_value: int = int(NodeType.OBSTACLE),
) -> np.ndarray:
    """Find boundary edges where both endpoints are obstacle nodes.

    Parameters
    ----------
    triangles : (E, 3) int
    nodetype : (N,) int
    obstacle_value : int

    Returns
    -------
    (B, 2) int — ordered boundary edges (each row: [n0, n1]).
    """
    obstacle_mask = nodetype.astype(int) == obstacle_value
    edges_all = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )
    # Keep edges where both nodes are obstacle
    both_obstacle = obstacle_mask[edges_all[:, 0]] & obstacle_mask[edges_all[:, 1]]
    candidate_edges = edges_all[both_obstacle]

    # Only keep edges that appear exactly once (true boundary, not interior obstacle edges)
    sorted_edges = np.sort(candidate_edges, axis=1)
    _, idx, counts = np.unique(
        sorted_edges, axis=0, return_index=True, return_counts=True
    )
    boundary_edges = candidate_edges[idx[counts == 1]]
    return boundary_edges


# ============================================================================
# Gradient computation
# ============================================================================


def tri_gradients(
    points: np.ndarray,
    triangles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-triangle gradient operators and areas for P1 elements.

    Returns
    -------
    grad_op : (E, 3, 2)  — gradient basis for each triangle.
    areas : (E,) — signed area of each triangle.
    tri_areas : (E,) — absolute area of each triangle.
    """
    pts = points[:, :2]
    v0 = pts[triangles[:, 0]]
    v1 = pts[triangles[:, 1]]
    v2 = pts[triangles[:, 2]]

    # 2 * signed area
    det = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (
        v1[:, 1] - v0[:, 1]
    )
    area2 = det
    areas = 0.5 * np.abs(det)

    # Gradient basis functions  dN_i/dx, dN_i/dy
    # dN0/dx = (y1 - y2) / det,  dN0/dy = (x2 - x1) / det
    # dN1/dx = (y2 - y0) / det,  dN1/dy = (x0 - x2) / det
    # dN2/dx = (y0 - y1) / det,  dN2/dy = (x1 - x0) / det
    inv_det = 1.0 / (area2 + 1e-30)

    grad_op = np.zeros((len(triangles), 3, 2))
    grad_op[:, 0, 0] = (v1[:, 1] - v2[:, 1]) * inv_det
    grad_op[:, 0, 1] = (v2[:, 0] - v1[:, 0]) * inv_det
    grad_op[:, 1, 0] = (v2[:, 1] - v0[:, 1]) * inv_det
    grad_op[:, 1, 1] = (v0[:, 0] - v2[:, 0]) * inv_det
    grad_op[:, 2, 0] = (v0[:, 1] - v1[:, 1]) * inv_det
    grad_op[:, 2, 1] = (v1[:, 0] - v0[:, 0]) * inv_det

    return grad_op, areas, areas


def nodal_gradient(
    scalar: np.ndarray,
    points: np.ndarray,
    triangles: np.ndarray,
    grad_op: np.ndarray,
    tri_areas: np.ndarray,
) -> np.ndarray:
    """Compute area-weighted nodal gradient of a scalar field.

    Parameters
    ----------
    scalar : (N,)
    points : (N, 2 or 3)
    triangles : (E, 3)
    grad_op : (E, 3, 2)
    tri_areas : (E,)

    Returns
    -------
    (N, 2) — gradient at each node.
    """
    N = len(scalar)
    # Element-wise gradient  (E, 2)
    elem_vals = scalar[triangles]  # (E, 3)
    elem_grad = np.einsum("ek,ekd->ed", elem_vals, grad_op)  # (E, 2)

    # Area-weighted scatter to nodes
    grad_sum = np.zeros((N, 2), dtype=np.float64)
    area_sum = np.zeros(N, dtype=np.float64)
    for k in range(3):
        np.add.at(grad_sum, triangles[:, k], elem_grad * tri_areas[:, None])
        np.add.at(area_sum, triangles[:, k], tri_areas)

    mask = area_sum > 0
    grad_sum[mask] /= area_sum[mask, None]
    return grad_sum


def triangle_velocity_gradient(
    vx: np.ndarray,
    vy: np.ndarray,
    points: np.ndarray,
    triangles: np.ndarray,
    grad_op: np.ndarray,
    tri_areas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nodal gradient of each velocity component.

    Returns ``(grad_vx, grad_vy)`` each of shape ``(N, 2)``.
    """
    g_vx = nodal_gradient(vx, points, triangles, grad_op, tri_areas)
    g_vy = nodal_gradient(vy, points, triangles, grad_op, tri_areas)
    return g_vx, g_vy


# ============================================================================
# Drag / Lift computation
# ============================================================================


def compute_drag_lift(
    points: np.ndarray,
    triangles: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    pressure: np.ndarray,
    levelset: np.ndarray,
    nodetype: np.ndarray,
    mu: float = 1.0,
    rho: float = 1.0,
    U_inf: float = 1.0,
    D: float = 1.0,
) -> Tuple[float, float]:
    """Compute drag and lift on the obstacle boundary.

    The outward-pointing normal at each boundary edge is determined from the
    *levelset* gradient direction (pointing away from the body).

    Parameters
    ----------
    points : (N, 2 or 3)
    triangles : (E, 3)
    vx, vy : (N,) — velocity components.
    pressure : (N,) — pressure field.
    levelset : (N,) — signed distance (or quantity whose gradient indicates outward normal).
    nodetype : (N,) — node classification.
    mu : float — dynamic viscosity.
    rho : float — density.
    U_inf : float — freestream velocity for coefficient normalisation.
    D : float — reference length (diameter) for coefficient normalisation.

    Returns
    -------
    (C_D, C_L) — drag and lift coefficients.
    """
    boundary_edges = compute_boundary_edges(triangles, nodetype)
    if len(boundary_edges) == 0:
        return 0.0, 0.0

    grad_op, _, tri_areas = tri_gradients(points, triangles)

    # Velocity gradients at nodes
    grad_vx, grad_vy = triangle_velocity_gradient(
        vx, vy, points, triangles, grad_op, tri_areas
    )

    # Levelset gradient for normal orientation
    grad_ls = nodal_gradient(levelset, points, triangles, grad_op, tri_areas)

    drag_total = 0.0
    lift_total = 0.0

    pts2d = points[:, :2]

    for n0, n1 in boundary_edges:
        edge_vec = pts2d[n1] - pts2d[n0]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-15:
            continue

        # Tangent and candidate normal
        tangent = edge_vec / edge_len
        normal_candidate = np.array([-tangent[1], tangent[0]])

        # Orient outward using levelset gradient at edge midpoint
        mid_grad_ls = 0.5 * (grad_ls[n0] + grad_ls[n1])
        if np.dot(normal_candidate, mid_grad_ls) < 0:
            normal_candidate = -normal_candidate

        nx, ny = normal_candidate

        # Average quantities at edge midpoint
        p_mid = 0.5 * (pressure[n0] + pressure[n1])
        gvx_mid = 0.5 * (grad_vx[n0] + grad_vx[n1])  # (du/dx, du/dy)
        gvy_mid = 0.5 * (grad_vy[n0] + grad_vy[n1])  # (dv/dx, dv/dy)

        # Stress tensor:  sigma_ij = -p delta_ij + mu (dui/dxj + duj/dxi)
        # Traction:  t_i = sigma_ij n_j
        # t_x = -p*nx + mu*(2*du/dx*nx + (du/dy + dv/dx)*ny)
        # t_y = -p*ny + mu*((du/dy + dv/dx)*nx + 2*dv/dy*ny)
        tx = -p_mid * nx + mu * (2.0 * gvx_mid[0] * nx + (gvx_mid[1] + gvy_mid[0]) * ny)
        ty = -p_mid * ny + mu * ((gvx_mid[1] + gvy_mid[0]) * nx + 2.0 * gvy_mid[1] * ny)

        drag_total += tx * edge_len
        lift_total += ty * edge_len

    # Normalise to coefficients
    q = 0.5 * rho * U_inf**2 * D
    if q > 0:
        C_D = drag_total / q
        C_L = lift_total / q
    else:
        C_D = drag_total
        C_L = lift_total

    return C_D, C_L


# ============================================================================
# Time-series force computation
# ============================================================================


def _get_field(mesh: meshio.Mesh, name: str, field_map: Dict[str, str]) -> np.ndarray:
    """Resolve a logical field name via the config map, falling back to direct lookup."""
    mapped = field_map.get(name, name)
    if mapped in mesh.point_data:
        return mesh.point_data[mapped]
    if name in mesh.point_data:
        return mesh.point_data[name]
    raise KeyError(
        f"Field '{name}' (mapped='{mapped}') not found in mesh point_data. "
        f"Available: {list(mesh.point_data.keys())}"
    )


def compute_series_forces(
    meshes: List[meshio.Mesh],
    timesteps: np.ndarray,
    feature_map: Dict[str, str],
    mu: float = 1.0,
    rho: float = 1.0,
    U_inf: float = 1.0,
    D: float = 1.0,
    n_workers: int = 4,
    start_step: int = 10,
) -> pd.DataFrame:
    """Compute drag / lift for all timesteps of a case.

    Parameters
    ----------
    meshes : list[meshio.Mesh]
    timesteps : (T,) array
    feature_map : dict
        Mapping from logical names (``velocity_x``, ``velocity_y``, ``pressure``,
        ``levelset``, ``nodetype``) to the actual field names in ``mesh.point_data``.
    mu, rho, U_inf, D : float
        Physical / normalisation parameters.
    n_workers : int
        Number of threads for parallel computation.

    Returns
    -------
    DataFrame with columns ``rollout_step, time, drag, lift``.
    """
    triangles = _get_triangles(meshes[0])
    points = meshes[0].points

    def _process_step(k: int) -> Tuple[int, float, float, float]:
        mesh = meshes[k]
        vx = _get_field(mesh, "velocity_x", feature_map)
        vy = _get_field(mesh, "velocity_y", feature_map)
        p = _get_field(mesh, "pressure", feature_map)
        ls = _get_field(mesh, "levelset", feature_map)
        nt = _get_field(mesh, "nodetype", feature_map)
        cd, cl = compute_drag_lift(
            points, triangles, vx, vy, p, ls, nt, mu, rho, U_inf, D
        )
        return int(k), float(timesteps[k]), cd, cl

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_step, k): k for k in range(len(meshes))}
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x[0])
    df = pd.DataFrame(results, columns=["rollout_step", "time", "drag", "lift"])
    if start_step > 0:
        df = df[df["rollout_step"] >= int(start_step)].reset_index(drop=True)
    return df


# ============================================================================
# End-to-end computation from config
# ============================================================================


def run_forces_computation(
    config: Dict[str, Any],
    output_dir: str,
    n_workers: int = 4,
) -> None:
    """Compute drag / lift for every model × case in the config.

    Produces:
    * ``forces/<model>/<case>.csv`` — per-timestep drag / lift.
    * ``forces_summary.csv``        — final-timestep comparison.
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

    # Feature map — maps logical names → actual field names in the XDMF
    feature_map = config.get(
        "feature_map",
        {
            "velocity_x": "x0",
            "velocity_y": "x1",
            "pressure": "x2",
            "levelset": "x3",
            "nodetype": "x6",
        },
    )

    # Physical parameters
    phys = config.get("physical_parameters", {})
    mu = phys.get("mu", 0.01)
    rho = phys.get("rho", 1.0)
    U_inf = phys.get("U_inf", 1.0)
    D = phys.get("D", 1.0)
    force_start_step = int(phys.get("force_start_step", 10))
    force_avg_window = int(phys.get("force_avg_window", 300))

    forces_dir = os.path.join(output_dir, "forces")
    os.makedirs(forces_dir, exist_ok=True)

    summary_rows = []

    for model_name in model_names:
        model_folder = _resolve_model_folder(pred_folder, model_name)
        cases = _discover_cases(model_folder, base_name, fallback_base)
        if not cases:
            raise ValueError(
                f"No .xdmf files found for model '{model_name}' in '{model_folder}'. "
                f"Tried base names '{base_name}'"
                + (f" and '{fallback_base}'" if fallback_base else "")
                + "."
            )
        model_out = os.path.join(forces_dir, model_name)
        os.makedirs(model_out, exist_ok=True)

        for case_id, xdmf_path in tqdm(cases.items(), desc=f"Forces [{model_name}]"):
            meshes, timesteps = xdmf_to_meshes(xdmf_path)
            timesteps = _to_physical_time(timesteps, dt=dt)
            df = compute_series_forces(
                meshes,
                timesteps,
                feature_map,
                mu=mu,
                rho=rho,
                U_inf=U_inf,
                D=D,
                n_workers=n_workers,
                start_step=force_start_step,
            )
            csv_path = os.path.join(model_out, f"{case_id}.csv")
            df.to_csv(csv_path, index=False)

            # Summary: final value + average on the last force_avg_window steps
            if df.empty:
                continue
            last = df.iloc[-1]
            tail = df.tail(min(force_avg_window, len(df)))
            summary_rows.append(
                {
                    "model": model_name,
                    "case": case_id,
                    "drag_final": last["drag"],
                    "lift_final": last["lift"],
                    "drag_mean_last_window": tail["drag"].mean(),
                    "lift_mean_last_window": tail["lift"].mean(),
                    "avg_window_steps": int(len(tail)),
                    "force_start_step": int(force_start_step),
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, "forces_summary.csv"), index=False
    )
    print(f"[compute_forces] Results → {output_dir}")


# ============================================================================
# Ground-truth force computation from a truth XDMF
# ============================================================================


def compute_truth_forces(
    truth_folder: str,
    case_base_name: Optional[str],
    feature_map: Dict[str, str],
    output_dir: str,
    mu: float = 0.01,
    rho: float = 1.0,
    U_inf: float = 1.0,
    D: float = 1.0,
    n_workers: int = 4,
    start_step: int = 10,
) -> None:
    """Compute forces on ground-truth XDMFs (for comparison)."""
    cases = _discover_cases(
        truth_folder,
        preferred_base_name=case_base_name or "",
        fallback_base_name=None,
    )
    os.makedirs(output_dir, exist_ok=True)

    for case_id, xdmf_path in tqdm(cases.items(), desc="Truth forces"):
        meshes, timesteps = xdmf_to_meshes(xdmf_path)
        for mesh in meshes:
            _inject_truth_alias_fields(mesh)
        timesteps = _to_physical_time(timesteps, dt=1.0)
        df = compute_series_forces(
            meshes,
            timesteps,
            feature_map,
            mu=mu,
            rho=rho,
            U_inf=U_inf,
            D=D,
            n_workers=n_workers,
            start_step=start_step,
        )
        normalized_case_id = _normalize_case_id(case_id)
        df.to_csv(os.path.join(output_dir, f"{normalized_case_id}.csv"), index=False)
    print(f"[compute_forces] Truth forces → {output_dir}")


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute drag / lift forces from GNN prediction XDMFs."
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
        default="./force_results",
        help="Output directory for CSV results.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Thread pool size for parallel timestep computation.",
    )
    p.add_argument(
        "--truth-folder",
        default=None,
        help="If given, also compute forces on ground-truth XDMFs in this folder.",
    )
    p.add_argument(
        "--truth-base-name",
        default=None,
        help="Base name for ground-truth XDMF files.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_json(args.parameters)

    run_forces_computation(config, args.directory, n_workers=args.workers)

    if args.truth_folder:
        feature_map = config.get(
            "feature_map",
            {
                "velocity_x": "x0",
                "velocity_y": "x1",
                "pressure": "x2",
                "levelset": "x3",
                "nodetype": "x6",
            },
        )
        dataset_params = config.get("dataset_parameters", {})
        truth_base_name = (
            args.truth_base_name
            or dataset_params.get("truth_base_name")
            or dataset_params.get("prediction_base_name")
            or ""
        )
        phys = config.get("physical_parameters", {})
        compute_truth_forces(
            truth_folder=args.truth_folder,
            case_base_name=truth_base_name,
            feature_map=feature_map,
            output_dir=os.path.join(args.directory, "forces", "truth"),
            mu=phys.get("mu", 0.01),
            rho=phys.get("rho", 1.0),
            U_inf=phys.get("U_inf", 1.0),
            D=phys.get("D", 1.0),
            n_workers=args.workers,
            start_step=int(phys.get("force_start_step", 10)),
        )


if __name__ == "__main__":
    main()
