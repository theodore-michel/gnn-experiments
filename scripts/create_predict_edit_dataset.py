#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import meshio
import numpy as np


def build_modified_step0(
    point_data: dict[str, np.ndarray],
    velocity_field: str,
    pressure_field: str,
) -> dict[str, np.ndarray]:
    if velocity_field not in point_data:
        raise KeyError(
            f"Missing velocity field '{velocity_field}' in step-0 point_data"
        )
    if pressure_field not in point_data:
        raise KeyError(
            f"Missing pressure field '{pressure_field}' in step-0 point_data"
        )

    out = {k: np.asarray(v).copy() for k, v in point_data.items()}
    velocity = np.asarray(point_data[velocity_field])
    if velocity.ndim != 2 or velocity.shape[1] < 1:
        raise ValueError(
            f"Expected vector velocity field with shape (N, C>=1), got {velocity.shape}"
        )

    new_velocity = np.zeros_like(velocity)
    new_velocity[:, 0] = 1.0
    out[velocity_field] = new_velocity
    out[pressure_field] = np.zeros_like(np.asarray(point_data[pressure_field]))
    return out


def process_case(
    src_xdmf: Path,
    out_xdmf: Path,
    velocity_field: str,
    pressure_field: str,
    verbose: bool,
) -> None:
    with meshio.xdmf.TimeSeriesReader(str(src_xdmf)) as reader:
        points, cells = reader.read_points_cells()
        if reader.num_steps < 1:
            raise ValueError(f"No timesteps found in {src_xdmf}")
        t0, pdata0, cdata0 = reader.read_data(0)

    modified_pdata0 = build_modified_step0(
        point_data=pdata0,
        velocity_field=velocity_field,
        pressure_field=pressure_field,
    )
    original_pdata0 = {k: np.asarray(v).copy() for k, v in pdata0.items()}

    out_xdmf.parent.mkdir(parents=True, exist_ok=True)

    prev_cwd = os.getcwd()
    try:
        # meshio writes a sidecar HDF5 file using relative paths.
        os.chdir(out_xdmf.parent)
        with meshio.xdmf.TimeSeriesWriter(out_xdmf.name) as writer:
            writer.write_points_cells(points, cells)
            writer.write_data(0.0, point_data=modified_pdata0, cell_data=cdata0)
            writer.write_data(0.2, point_data=original_pdata0, cell_data=cdata0)
    finally:
        os.chdir(prev_cwd)

    if verbose:
        print(
            f"[OK] {src_xdmf.name} -> {out_xdmf.name} "
            f"(prepended edited step, preserved source step-0)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recreate a predict dataset with only step-0 kept, duplicated to 2 steps, "
            "and step-0 fields overwritten (velocity=1, pressure=0)."
        )
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--velocity-field", default="Vitesse")
    parser.add_argument("--pressure-field", default="Pression")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    src_xdmfs = sorted(args.source_dir.glob("*.xdmf"))
    if not src_xdmfs:
        raise FileNotFoundError(f"No xdmf files found in source: {args.source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for src_xdmf in src_xdmfs:
        out_xdmf = args.output_dir / src_xdmf.name
        process_case(
            src_xdmf=src_xdmf,
            out_xdmf=out_xdmf,
            velocity_field=args.velocity_field,
            pressure_field=args.pressure_field,
            verbose=args.verbose,
        )
        processed += 1

    print(
        "Completed predict_edit generation | "
        f"cases={processed} | source={args.source_dir} | output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
