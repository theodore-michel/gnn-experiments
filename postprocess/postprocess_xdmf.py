from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import meshio
import numpy as np
from tqdm import tqdm

from postprocess.utils.xdmf_io import (
    crop_rollout,
    discover_xdmf_cases,
    ensure_dir,
    feature_key_for_semantic,
    latest_x_feature_key,
    load_json,
    normalize_case_id,
    read_case_levelset_from_dataset,
    read_xdmf_series,
    stacked_vector,
    write_xdmf_series,
)


def _build_processed_meshes(
    meshes: List[meshio.Mesh],
    timesteps: np.ndarray,
    feature_map: Dict[str, str],
    target_step_offset: int,
    levelset_source: str,
    dataset_dir: str,
    case_id: str,
    rollout_steps: Optional[int],
) -> tuple[List[meshio.Mesh], np.ndarray]:
    meshes, timesteps = crop_rollout(meshes, timesteps, rollout_steps)
    if target_step_offset < 0:
        raise ValueError("target_step_offset must be >= 0")
    if len(meshes) <= target_step_offset:
        return [], np.array([], dtype=float)

    levelset_key = feature_key_for_semantic(feature_map, "levelset")
    nodetype_key = latest_x_feature_key(feature_map)

    levelset_dataset = None
    if levelset_source == "dataset" or levelset_key is None:
        levelset_dataset = read_case_levelset_from_dataset(dataset_dir, case_id)

    out_meshes: List[meshio.Mesh] = []
    out_times: List[float] = []

    for t in tqdm(
        range(target_step_offset, len(meshes)),
        desc=f"Align {case_id}",
        leave=False,
    ):
        pred = meshes[t]
        targ = meshes[t - target_step_offset]

        vx = np.asarray(pred.point_data["x0"]).reshape(-1)
        vy = np.asarray(pred.point_data["x1"]).reshape(-1)
        p = np.asarray(pred.point_data["x2"]).reshape(-1)

        vx_targ = np.asarray(targ.point_data["y0"]).reshape(-1)
        vy_targ = np.asarray(targ.point_data["y1"]).reshape(-1)
        p_targ = np.asarray(targ.point_data["y2"]).reshape(-1)

        if levelset_dataset is not None:
            levelset = levelset_dataset
        else:
            levelset = np.asarray(pred.point_data[levelset_key]).reshape(-1)

        nodetype = np.asarray(pred.point_data[nodetype_key]).reshape(-1)

        point_data = {
            "vx": vx,
            "vy": vy,
            "v_pred": stacked_vector(vx, vy),
            "p": p,
            "levelset": levelset,
            "nodetype": nodetype,
            "vx_targ": vx_targ,
            "vy_targ": vy_targ,
            "v_targ": stacked_vector(vx_targ, vy_targ),
            "p_targ": p_targ,
        }

        out_meshes.append(
            meshio.Mesh(points=pred.points, cells=pred.cells, point_data=point_data)
        )
        out_times.append(float(timesteps[t]))

    return out_meshes, np.asarray(out_times, dtype=float)


def run(config_path: str) -> None:
    cfg = load_json(config_path)

    pred_dir = cfg["pred_dir"]
    dataset_dir = cfg["dataset_dir"]
    model_name = cfg["model_name"]
    output_dir = cfg["output_dir"]
    feature_map = cfg["feature_map"]
    target_step_offset = int(cfg.get("target_step_offset", 1))
    rollout_steps = cfg.get("rollout_steps")
    levelset_source = cfg.get("levelset_source", "dataset")

    model_root = Path(ensure_dir(Path(output_dir) / model_name))
    out_xdmf_dir = Path(ensure_dir(model_root / "xdmf"))

    pred_prefix = cfg.get("prediction_base_name")
    cases = discover_xdmf_cases(pred_dir, pred_prefix)
    if not cases:
        raise FileNotFoundError(f"No prediction XDMFs found in {pred_dir}")

    written_count = 0
    skipped_count = 0
    for case_id, xdmf_path in tqdm(cases.items(), total=len(cases), desc="Cases"):
        meshes, timesteps = read_xdmf_series(xdmf_path)
        proc_meshes, proc_times = _build_processed_meshes(
            meshes=meshes,
            timesteps=timesteps,
            feature_map=feature_map,
            target_step_offset=target_step_offset,
            levelset_source=levelset_source,
            dataset_dir=dataset_dir,
            case_id=normalize_case_id(case_id),
            rollout_steps=rollout_steps,
        )
        if not proc_meshes:
            skipped_count += 1
            continue
        write_xdmf_series(
            out_xdmf_dir / f"{normalize_case_id(case_id)}.xdmf", proc_meshes, proc_times
        )
        written_count += 1

    if written_count == 0:
        raise RuntimeError(
            "postprocess_xdmf generated 0 processed cases. "
            "This often means target_step_offset is too large for the available "
            "prediction timesteps (e.g. 1-step predictions with target_step_offset=1)."
        )

    print(
        f"[postprocess_xdmf] Wrote {written_count} processed XDMFs to {out_xdmf_dir} "
        f"(skipped {skipped_count} cases with empty aligned rollout)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Postprocess raw prediction XDMF files"
    )
    parser.add_argument("config", help="Unified JSON config file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
