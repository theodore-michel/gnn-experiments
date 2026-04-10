#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from typing import Dict, Iterable, Tuple

import meshio
import numpy as np
from scipy.spatial import cKDTree


def extract_case_id(src_name: str) -> str:
    """Extract case id from source predict filename like cylinders_10000.xdmf."""
    stem = Path(src_name).stem
    if "_" not in stem:
        raise ValueError(f"Unsupported source filename format: {src_name}")
    return stem.split("_")[-1]


def normalize_like(arr: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Reshape/validate replacement array to match target field shape."""
    arr = np.asarray(arr)
    target = np.asarray(target)

    if arr.shape == target.shape:
        return arr

    if target.ndim == 1 and arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]

    if target.ndim == 2 and target.shape[1] == 1 and arr.ndim == 1:
        return arr[:, None]

    raise ValueError(
        f"Cannot align replacement shape {arr.shape} to target shape {target.shape}"
    )


def same_cells(
    cells_a: Iterable[meshio.CellBlock], cells_b: Iterable[meshio.CellBlock]
) -> bool:
    a = list(cells_a)
    b = list(cells_b)
    if len(a) != len(b):
        return False
    for ca, cb in zip(a, b):
        if ca.type != cb.type:
            return False
        if ca.data.shape != cb.data.shape:
            return False
        if not np.array_equal(ca.data, cb.data):
            return False
    return True


def copy_source_case_files(src_xdmf: Path, out_xdmf: Path) -> None:
    """Copy source xdmf+h5 pair unchanged to output location."""
    out_xdmf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_xdmf, out_xdmf)

    src_h5 = src_xdmf.with_suffix(".h5")
    out_h5 = out_xdmf.with_suffix(".h5")
    if src_h5.exists():
        shutil.copy2(src_h5, out_h5)


def build_point_index_map(
    src_points: np.ndarray, init_points: np.ndarray
) -> np.ndarray:
    """Return indices mapping source node order -> initializer node order."""
    if src_points.shape != init_points.shape:
        raise ValueError(
            f"Point array shape mismatch: source={src_points.shape}, init={init_points.shape}"
        )

    tree = cKDTree(init_points)
    dist, idx = tree.query(src_points, k=1)

    if np.max(dist) > 1e-12:
        raise ValueError(
            "Mesh points mismatch: nearest-neighbor distance exceeds tolerance "
            f"(max={float(np.max(dist))})"
        )

    if len(np.unique(idx)) != len(idx):
        raise ValueError(
            "Mesh point mapping is not one-to-one between source and initializer"
        )

    return idx.astype(np.int64)


def substitute_case(
    src_xdmf: Path,
    init_xdmf: Path,
    out_xdmf: Path,
    source_v_field: str,
    source_p_field: str,
    init_v_field: str,
    init_p_field: str,
    verbose: bool,
) -> None:
    with meshio.xdmf.TimeSeriesReader(str(src_xdmf)) as src_reader:
        src_points, src_cells = src_reader.read_points_cells()
        src_steps = src_reader.num_steps
        src_series = [src_reader.read_data(i) for i in range(src_steps)]

    if src_steps < 1:
        raise ValueError(f"Source has no timesteps: {src_xdmf}")

    with meshio.xdmf.TimeSeriesReader(str(init_xdmf)) as init_reader:
        init_points, init_cells = init_reader.read_points_cells()
        init_steps = init_reader.num_steps
        init_t0, init_pdata0, _ = init_reader.read_data(0)

    if init_steps < 1:
        raise ValueError(f"Initializer has no timesteps: {init_xdmf}")

    point_map = build_point_index_map(src_points, init_points)

    if not same_cells(src_cells, init_cells) and verbose:
        print(
            f"[INFO] {src_xdmf.stem}: source/init cell ordering differs; "
            "using coordinate-based point remap for substitution"
        )

    t0, src_pdata0, src_cdata0 = src_series[0]

    if source_v_field not in src_pdata0:
        raise KeyError(
            f"Missing source velocity field '{source_v_field}' in {src_xdmf}"
        )
    if source_p_field not in src_pdata0:
        raise KeyError(
            f"Missing source pressure field '{source_p_field}' in {src_xdmf}"
        )
    if init_v_field not in init_pdata0:
        raise KeyError(
            f"Missing initializer velocity field '{init_v_field}' in {init_xdmf}"
        )
    if init_p_field not in init_pdata0:
        raise KeyError(
            f"Missing initializer pressure field '{init_p_field}' in {init_xdmf}"
        )

    init_v_mapped = np.asarray(init_pdata0[init_v_field])[point_map]
    init_p_mapped = np.asarray(init_pdata0[init_p_field])[point_map]

    src_pdata0[source_v_field] = normalize_like(
        init_v_mapped, src_pdata0[source_v_field]
    )
    src_pdata0[source_p_field] = normalize_like(
        init_p_mapped, src_pdata0[source_p_field]
    )

    out_xdmf.parent.mkdir(parents=True, exist_ok=True)

    prev_cwd = os.getcwd()
    try:
        os.chdir(out_xdmf.parent)
        with meshio.xdmf.TimeSeriesWriter(out_xdmf.name) as writer:
            writer.write_points_cells(src_points, src_cells)
            writer.write_data(t0, point_data=src_pdata0, cell_data=src_cdata0)

            for t, pdata, cdata in src_series[1:]:
                writer.write_data(t, point_data=pdata, cell_data=cdata)
    finally:
        os.chdir(prev_cwd)

    if verbose:
        print(
            "[OK] "
            f"{src_xdmf.name} -> {out_xdmf.name} | "
            f"step0 {source_v_field}<={init_v_field}, {source_p_field}<={init_p_field} | "
            f"init_t0={init_t0}"
        )


def build_for_regime(
    re: int,
    source_template: str,
    init_template: str,
    output_template: str,
    source_v_field: str,
    source_p_field: str,
    init_v_field: str,
    init_p_field: str,
    verbose: bool,
) -> Tuple[int, int, int]:
    src_dir = Path(source_template.format(re=re))
    init_dir = Path(init_template.format(re=re))
    out_dir = Path(output_template.format(re=re))

    if not src_dir.is_dir():
        raise FileNotFoundError(f"Missing source predict directory: {src_dir}")
    if not init_dir.is_dir():
        raise FileNotFoundError(f"Missing initializer xdmf directory: {init_dir}")

    src_xdmfs = sorted(src_dir.glob("*.xdmf"))
    if not src_xdmfs:
        raise FileNotFoundError(f"No source xdmf files found in {src_dir}")

    init_by_case: Dict[str, Path] = {p.stem: p for p in sorted(init_dir.glob("*.xdmf"))}
    if not init_by_case:
        raise FileNotFoundError(f"No initializer xdmf files found in {init_dir}")

    processed = 0
    missing = 0
    incompatible = 0

    if verbose:
        print("\n" + "=" * 88)
        print(f"Re1e{re}: source={src_dir}")
        print(f"Re1e{re}: init  ={init_dir}")
        print(f"Re1e{re}: out   ={out_dir}")
        print("=" * 88)

    for src_xdmf in src_xdmfs:
        case_id = extract_case_id(src_xdmf.name)
        init_xdmf = init_by_case.get(case_id)

        if init_xdmf is None:
            missing += 1
            print(
                f"[MISSING] Re1e{re} case {case_id}: no initializer xdmf {case_id}.xdmf"
            )
            continue

        out_xdmf = out_dir / src_xdmf.name

        try:
            substitute_case(
                src_xdmf=src_xdmf,
                init_xdmf=init_xdmf,
                out_xdmf=out_xdmf,
                source_v_field=source_v_field,
                source_p_field=source_p_field,
                init_v_field=init_v_field,
                init_p_field=init_p_field,
                verbose=verbose,
            )
        except Exception as exc:
            incompatible += 1
            copy_source_case_files(src_xdmf, out_xdmf)
            print(
                f"[WARN] Re1e{re} case {case_id}: substitution skipped, copied source unchanged ({exc})"
            )
            continue

        processed += 1

    return processed, missing, incompatible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create combined datasets where the first timestep fields in full-trajectory predict "
            "XDMFs are replaced by 1-step initializer predictions."
        )
    )
    parser.add_argument(
        "--res",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="Reynolds regimes to process (default: 2 3 4)",
    )
    parser.add_argument(
        "--source-template",
        default="/scratch-big/tmichel/GNN/DATASETS/dataset_onecyl_Re1e{re}_gmsh_trunc/predict",
        help="Template for source full-trajectory predict directory",
    )
    parser.add_argument(
        "--init-template",
        default=(
            "/scratch-big/tmichel/GNN/RUNS/NEWCYL/ONECYL_ARTICLE/gnn-experiments/"
            "results_ReX_1step/onecyl_Re{re}_1step_VPLN/xdmf"
        ),
        help="Template for initializer 1-step model xdmf directory",
    )
    parser.add_argument(
        "--output-template",
        default=(
            "/scratch-big/tmichel/GNN/DATASETS/DATASETS_COMBINED_INIT_ITER/"
            "dataset_onecyl_Re1e{re}_gmsh_trunc/predict"
        ),
        help="Template for output combined predict directory",
    )
    parser.add_argument(
        "--source-v-field", default="Vitesse", help="Velocity field in source xdmf"
    )
    parser.add_argument(
        "--source-p-field", default="Pression", help="Pressure field in source xdmf"
    )
    parser.add_argument(
        "--init-v-field", default="v_pred", help="Velocity field in initializer xdmf"
    )
    parser.add_argument(
        "--init-p-field", default="p", help="Pressure field in initializer xdmf"
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any source case is missing in initializer outputs",
    )
    parser.add_argument(
        "--strict-incompatible",
        action="store_true",
        help="Fail if any case cannot be substituted due to mesh incompatibility",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose per-case logging"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    total_processed = 0
    total_missing = 0
    total_incompatible = 0

    for re in args.res:
        processed, missing, incompatible = build_for_regime(
            re=re,
            source_template=args.source_template,
            init_template=args.init_template,
            output_template=args.output_template,
            source_v_field=args.source_v_field,
            source_p_field=args.source_p_field,
            init_v_field=args.init_v_field,
            init_p_field=args.init_p_field,
            verbose=args.verbose,
        )
        total_processed += processed
        total_missing += missing
        total_incompatible += incompatible

    print("\n" + "=" * 88)
    print(
        "Completed combined dataset generation | "
        f"substituted={total_processed} | missing={total_missing} | incompatible={total_incompatible}"
    )
    print("=" * 88)

    if args.strict_missing and total_missing > 0:
        raise SystemExit(2)
    if args.strict_incompatible and total_incompatible > 0:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
