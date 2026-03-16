from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from postprocess.utils.xdmf_io import (
    NodeType,
    discover_xdmf_cases,
    ensure_dir,
    load_json,
    read_xdmf_series,
)


def _triangles(mesh) -> np.ndarray:
    for cell in mesh.cells:
        if cell.type == "triangle":
            return cell.data
    raise ValueError("No triangle cells found")


def _boundary_edges(triangles: np.ndarray, nodetype: np.ndarray) -> np.ndarray:
    obs = nodetype.astype(int) == int(NodeType.OBSTACLE)
    edges = np.vstack(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    edges = edges[obs[edges[:, 0]] & obs[edges[:, 1]]]
    sorted_edges = np.sort(edges, axis=1)
    uniq, idx, cnt = np.unique(
        sorted_edges, axis=0, return_index=True, return_counts=True
    )
    _ = uniq
    return edges[idx[cnt == 1]]


def _tri_grad(points: np.ndarray, tri: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = points[:, :2]
    a = p[tri[:, 0]]
    b = p[tri[:, 1]]
    c = p[tri[:, 2]]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (
        b[:, 1] - a[:, 1]
    )
    inv = 1.0 / (det + 1e-30)
    grad = np.zeros((len(tri), 3, 2), dtype=float)
    grad[:, 0, 0] = (b[:, 1] - c[:, 1]) * inv
    grad[:, 0, 1] = (c[:, 0] - b[:, 0]) * inv
    grad[:, 1, 0] = (c[:, 1] - a[:, 1]) * inv
    grad[:, 1, 1] = (a[:, 0] - c[:, 0]) * inv
    grad[:, 2, 0] = (a[:, 1] - b[:, 1]) * inv
    grad[:, 2, 1] = (b[:, 0] - a[:, 0]) * inv
    area = 0.5 * np.abs(det)
    return grad, area


def _nodal_grad(
    field: np.ndarray, tri: np.ndarray, grad_op: np.ndarray, area: np.ndarray
) -> np.ndarray:
    elem_grad = np.einsum("ek,ekd->ed", field[tri], grad_op)
    gsum = np.zeros((len(field), 2), dtype=float)
    asum = np.zeros(len(field), dtype=float)
    for j in range(3):
        np.add.at(gsum, tri[:, j], elem_grad * area[:, None])
        np.add.at(asum, tri[:, j], area)
    out = np.zeros_like(gsum)
    mask = asum > 0
    out[mask] = gsum[mask] / asum[mask, None]
    return out


def _drag_lift(points, tri, edges, vx, vy, p, levelset, mu=1e-3):
    grad_op, area = _tri_grad(points, tri)
    gvx = _nodal_grad(vx, tri, grad_op, area)
    gvy = _nodal_grad(vy, tri, grad_op, area)
    gls = _nodal_grad(levelset, tri, grad_op, area)

    pts = points[:, :2]
    fx = 0.0
    fy = 0.0
    for n0, n1 in edges:
        e = pts[n1] - pts[n0]
        el = np.linalg.norm(e)
        if el < 1e-15:
            continue
        t = e / el
        n = np.array([-t[1], t[0]])
        gl = 0.5 * (gls[n0] + gls[n1])
        if np.dot(n, gl) < 0:
            n = -n
        nx, ny = n
        pm = 0.5 * (p[n0] + p[n1])
        gu = 0.5 * (gvx[n0] + gvx[n1])
        gv = 0.5 * (gvy[n0] + gvy[n1])
        tx = -pm * nx + mu * (2.0 * gu[0] * nx + (gu[1] + gv[0]) * ny)
        ty = -pm * ny + mu * ((gu[1] + gv[0]) * nx + 2.0 * gv[1] * ny)
        fx += tx * el
        fy += ty * el
    return fx, fy


def run(config_path: str) -> None:
    cfg = load_json(config_path)
    model_name = cfg["model_name"]
    model_root = Path(cfg["output_dir"]) / model_name
    xdmf_dir = model_root / "xdmf"
    forces_dir = Path(ensure_dir(model_root / "forces"))

    cases = discover_xdmf_cases(xdmf_dir)
    to_compute = []
    for case_id in cases.keys():
        fpath = forces_dir / f"forces_{case_id}.csv"
        if fpath.exists():
            print(f"[compute_forces] cache hit: {case_id}")
        else:
            to_compute.append(case_id)

    print(f"[compute_forces] recompute cases: {to_compute}")

    mu = float(cfg.get("force_mu", 1e-3))
    workers = int(cfg.get("force_workers", 4))

    sorted_case_ids = sorted(cases.keys(), key=lambda x: int(x) if x.isdigit() else x)
    for case_id in tqdm(sorted_case_ids, desc="Force cases"):
        out_csv = forces_dir / f"forces_{case_id}.csv"
        if out_csv.exists():
            continue
        meshes, _ = read_xdmf_series(cases[case_id])
        tri = _triangles(meshes[0])
        pts = meshes[0].points
        edges = _boundary_edges(
            tri, np.asarray(meshes[0].point_data["nodetype"]).reshape(-1)
        )

        def _step(k: int):
            mesh = meshes[k]
            vx = np.asarray(mesh.point_data["vx"]).reshape(-1)
            vy = np.asarray(mesh.point_data["vy"]).reshape(-1)
            p = np.asarray(mesh.point_data["p"]).reshape(-1)
            vx_t = np.asarray(mesh.point_data["vx_targ"]).reshape(-1)
            vy_t = np.asarray(mesh.point_data["vy_targ"]).reshape(-1)
            p_t = np.asarray(mesh.point_data["p_targ"]).reshape(-1)
            ls = np.asarray(mesh.point_data["levelset"]).reshape(-1)
            fx_p, fy_p = _drag_lift(pts, tri, edges, vx, vy, p, ls, mu=mu)
            fx_t, fy_t = _drag_lift(pts, tri, edges, vx_t, vy_t, p_t, ls, mu=mu)
            return k, -fx_p, -fy_p, -fx_t, -fy_t

        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_step, k) for k in range(len(meshes))]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Timesteps {case_id}",
                leave=False,
            ):
                rows.append(future.result())
        rows.sort(key=lambda x: x[0])

        df = pd.DataFrame(
            rows, columns=["timestep", "fx_pred", "fy_pred", "fx_targ", "fy_targ"]
        )
        df.insert(0, "case_id", case_id)
        df.insert(0, "model_name", model_name)
        df.to_csv(out_csv, index=False)

    summary_rows = []
    for csv_path in sorted(forces_dir.glob("forces_*.csv")):
        df = pd.read_csv(csv_path)
        case_id = str(df["case_id"].iloc[0])
        summary_rows.append(
            {
                "model_name": model_name,
                "case_id": case_id,
                "fx_pred_mean": df["fx_pred"].mean(),
                "fy_pred_mean": df["fy_pred"].mean(),
                "fx_targ_mean": df["fx_targ"].mean(),
                "fy_targ_mean": df["fy_targ"].mean(),
                "fx_pred_std": df["fx_pred"].std(),
                "fy_pred_std": df["fy_pred"].std(),
                "fx_targ_std": df["fx_targ"].std(),
                "fy_targ_std": df["fy_targ"].std(),
            }
        )
    pd.DataFrame(summary_rows).to_csv(forces_dir / "forces_summary.csv", index=False)
    print(f"[compute_forces] Results written to {forces_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute force time-series from postprocessed XDMFs"
    )
    parser.add_argument("config", help="Unified JSON config file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
