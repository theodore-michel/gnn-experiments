"""
process_xdmf.py — Create processed XDMF files for ParaView-ready visualization.

This script reproduces the core behavior of scripts/postprocess_gnn.py in a
clean CLI module:
- rename selected scalar fields,
- handle x/y timestep shift (y at t-1 aligns with x at t),
- create vector fields for streamlines,
- optionally create velocity norms,
- write processed time-series XDMF files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from ..utils.xdmf_io import (
        create_norm_field,
        create_vector_field,
        gather_cases,
        load_json,
        meshes_to_xdmf,
        xdmf_to_meshes,
    )
except ImportError:
    from postprocess.utils.xdmf_io import (
        create_norm_field,
        create_vector_field,
        gather_cases,
        load_json,
        meshes_to_xdmf,
        xdmf_to_meshes,
    )


def _resolve_model_folder(pred_folder: str, model_name: str) -> str:
    nested = os.path.join(pred_folder, model_name)
    return nested if os.path.isdir(nested) else pred_folder


def _discover_cases(
    case_folder: str,
    preferred_base_name: str,
    fallback_base_name: Optional[str] = None,
) -> Dict[str, str]:
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


def _build_processed_meshes(
    meshes: List[Any],
    shift_steps: int,
) -> List[Any]:
    """Create processed mesh list with aligned pred/targ fields and vectors.

    Raw convention:
      - x0, x1, x2: prediction-aligned scalar fields
      - y0, y1, y2: target fields one step ahead

    Alignment used here:
      For mesh index k>=shift, compare x*[k] with y*[k-shift].
    """
    if shift_steps < 0:
        raise ValueError("shift_steps must be >= 0")

    out = []
    start = shift_steps
    for k in range(start, len(meshes)):
        m_cur = meshes[k]
        m_ref = meshes[k - shift_steps] if shift_steps > 0 else meshes[k]

        proc = m_cur.copy()
        pd = proc.point_data

        # scalar pred/targ fields
        if "x0" in m_cur.point_data:
            pd["Vx_pred"] = m_cur.point_data["x0"]
        if "x1" in m_cur.point_data:
            pd["Vy_pred"] = m_cur.point_data["x1"]
        if "x2" in m_cur.point_data:
            pd["P_pred"] = m_cur.point_data["x2"]

        if "y0" in m_ref.point_data:
            pd["Vx_targ"] = m_ref.point_data["y0"]
        if "y1" in m_ref.point_data:
            pd["Vy_targ"] = m_ref.point_data["y1"]
        if "y2" in m_ref.point_data:
            pd["P_targ"] = m_ref.point_data["y2"]

        # keep geometric helpers when available
        for aux in ("x3", "x4", "x5", "x6"):
            if aux in m_cur.point_data:
                pd[aux] = m_cur.point_data[aux]

        # vector + norm for streamlines/visualization
        if "Vx_pred" in pd and "Vy_pred" in pd:
            create_vector_field(proc, "V_vect_pred", ["Vx_pred", "Vy_pred"], fill=True)
            create_norm_field(proc, "V_pred", ["Vx_pred", "Vy_pred"])
        if "Vx_targ" in pd and "Vy_targ" in pd:
            create_vector_field(proc, "V_vect_targ", ["Vx_targ", "Vy_targ"], fill=True)
            create_norm_field(proc, "V_targ", ["Vx_targ", "Vy_targ"])

        out.append(proc)

    return out


def run_process_xdmf(config: Dict[str, Any], output_dir: str) -> None:
    dataset_params = config["dataset_parameters"]
    model_params = config["model_parameters"]

    model_names = model_params["name"] if isinstance(model_params["name"], list) else [model_params["name"]]
    base_name = model_params.get("final_base_name", "pred_")
    fallback_base = dataset_params.get("prediction_base_name")
    pred_folder = dataset_params["prediction_folder"]
    dt = dataset_params.get("dt", 1.0)
    shift_steps = model_params.get("truth_shift_steps", 1)

    processed_root = os.path.join(output_dir, "processed_xdmf")
    os.makedirs(processed_root, exist_ok=True)

    for model_name in model_names:
        model_folder = _resolve_model_folder(pred_folder, model_name)
        cases = _discover_cases(model_folder, base_name, fallback_base)
        if not cases:
            raise ValueError(f"No XDMF cases found in {model_folder}")

        model_out = os.path.join(processed_root, model_name)
        os.makedirs(model_out, exist_ok=True)

        for case_id, path in cases.items():
            meshes, timesteps = xdmf_to_meshes(path)
            timesteps = _to_physical_time(timesteps, dt)
            proc_meshes = _build_processed_meshes(meshes, shift_steps=shift_steps)
            proc_times = timesteps[shift_steps:] if shift_steps > 0 else timesteps
            out_path = os.path.join(model_out, f"processed_{case_id}.xdmf")
            meshes_to_xdmf(out_path, proc_meshes, proc_times, drop_first=False, verbose=False)

    print(f"[process_xdmf] Processed XDMFs written to {processed_root}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create processed XDMF files with aligned/vectorized fields.")
    p.add_argument("-p", "--parameters", required=True, help="JSON config file path.")
    p.add_argument("-d", "--directory", required=True, help="Output root directory.")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_json(args.parameters)
    run_process_xdmf(cfg, args.directory)


if __name__ == "__main__":
    main()
