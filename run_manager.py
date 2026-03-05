#!/usr/bin/env python3
"""
run_manager.py — CLI for GNN experiment tracking.

Commands:
    register   Snapshot metadata from a run folder into a YAML record.
    update     Pull metrics from wandb / SLURM accounting into a record.
    relaunch   Reconstruct a run folder from a record and re-submit.
    compare    Print a ranked table of runs.
    sync-sheet (stub)

Usage:
    python run_manager.py register --run-dir /path/to/run [--base-commit SHA] [--notes TEXT]
    python run_manager.py update   --run-id ID
    python run_manager.py relaunch --run-id ID [--dry-run]
    python run_manager.py compare  [--dataset re2_full] [--sort-by val/rollout_rmse] [--top 10]
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml
from rich.console import Console
from rich.table import Table

# ── Paths ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRY_DIR = SCRIPT_DIR / "registry"
REGISTRY_DIR.mkdir(exist_ok=True)

# Default clean graph-physics reference.  Override with env var GRAPH_PHYSICS_REF.
DEFAULT_BASE_REF = os.environ.get(
    "GRAPH_PHYSICS_REF",
    str(SCRIPT_DIR.parent / "graph-physics"),
)

console = Console()

# ── Helpers ─────────────────────────────────────────────────────────────


def _load_record(run_id: str) -> Dict[str, Any]:
    """Load a run record YAML by run_id."""
    path = REGISTRY_DIR / f"{run_id}.yaml"
    if not path.exists():
        raise click.ClickException(f"Run record not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _save_record(record: Dict[str, Any]) -> Path:
    """Save a run record YAML."""
    run_id = record["run_id"]
    path = REGISTRY_DIR / f"{run_id}.yaml"
    with open(path, "w") as f:
        yaml.dump(record, f, default_flow_style=False, sort_keys=False, width=120)
    return path


def _generate_run_id(dataset_label: str) -> str:
    """Generate a human-readable run ID like re100_full_20250305_001."""
    today = datetime.datetime.now().strftime("%Y%m%d")
    prefix = f"{dataset_label}_{today}"
    existing = sorted(REGISTRY_DIR.glob(f"{prefix}_*.yaml"))
    seq = 1
    if existing:
        last = existing[-1].stem
        match = re.search(r"_(\d+)$", last)
        if match:
            seq = int(match.group(1)) + 1
    return f"{prefix}_{seq:03d}"


def _infer_dataset_label(params: Dict[str, Any]) -> str:
    """Produce a short dataset label like 're100_full' from the parameters JSON."""
    xdmf = params.get("dataset", {}).get("xdmf_folder", "")
    train_path = params.get("dataset", {}).get("train_path", "")
    path_str = xdmf or train_path

    # Infer Re case from path
    re_label = "reX"
    for pattern, label in [
        (r"Re1e2|_Re2", "re100"),
        (r"Re1e3|_Re3", "re1000"),
        (r"Re1e4|_Re4", "re10000"),
    ]:
        if re.search(pattern, path_str, re.IGNORECASE):
            re_label = label
            break

    # Infer variant from folder name or default
    variant = "full"  # default
    if "mini" in path_str.lower():
        variant = "mini"

    return f"{re_label}_{variant}"


def _infer_re_case(params: Dict[str, Any]) -> str:
    """Return short Re case tag like Re2, Re3, Re4."""
    xdmf = params.get("dataset", {}).get("xdmf_folder", "")
    train_path = params.get("dataset", {}).get("train_path", "")
    path_str = xdmf or train_path
    for pattern, tag in [
        (r"Re1e2|_Re2", "Re2"),
        (r"Re1e3|_Re3", "Re3"),
        (r"Re1e4|_Re4", "Re4"),
    ]:
        if re.search(pattern, path_str, re.IGNORECASE):
            return tag
    return "ReX"


def _infer_variant(params: Dict[str, Any]) -> str:
    xdmf = params.get("dataset", {}).get("xdmf_folder", "")
    train_path = params.get("dataset", {}).get("train_path", "")
    path_str = xdmf or train_path
    return "mini" if "mini" in path_str.lower() else "full"


def _compute_diff(run_dir: str, base_ref: str) -> str:
    """Compute a unified diff of run_dir vs base_ref (clean graph-physics)."""
    run_dir = os.path.abspath(run_dir)
    base_ref = os.path.abspath(base_ref)

    if not os.path.isdir(base_ref):
        return f"# WARNING: base reference not found at {base_ref}; diff skipped.\n"

    # Files to compare: everything except transient/output dirs
    exclude_patterns = [
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=*.pyc",
        "--exclude=wandb",
        "--exclude=meshes*",
        "--exclude=checkpoints",
        "--exclude=backup",
        "--exclude=*.ckpt",
        "--exclude=*.log",
        "--exclude=*.egg-info",
    ]

    cmd = [
        "diff",
        "-ruN",
        *exclude_patterns,
        base_ref,
        run_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # diff returns 0 = identical, 1 = differences, 2 = error
    if result.returncode == 2:
        return f"# ERROR running diff: {result.stderr.strip()}\n"
    return result.stdout or "# No differences found.\n"


def _find_wandb_run(run_dir: str) -> Dict[str, Optional[str]]:
    """Find wandb run_id and offline folder from the run directory."""
    wandb_dir = os.path.join(run_dir, "wandb")
    info: Dict[str, Optional[str]] = {
        "run_id": None,
        "project": None,
        "offline_dir": None,
    }

    if not os.path.isdir(wandb_dir):
        return info

    # Find offline-run-* directories, take the latest
    offline_dirs = sorted(glob.glob(os.path.join(wandb_dir, "offline-run-*")))
    if not offline_dirs:
        # Also try run-* pattern
        offline_dirs = sorted(glob.glob(os.path.join(wandb_dir, "run-*")))
    if not offline_dirs:
        return info

    latest = offline_dirs[-1]
    info["offline_dir"] = latest

    # Extract run ID from directory name: offline-run-YYYYMMDD_HHMMSS-<run_id>
    match = re.search(r"-([a-z0-9]+)$", os.path.basename(latest))
    if match:
        info["run_id"] = match.group(1)

    # Try reading config.yaml for project name
    config_path = os.path.join(latest, "files", "config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                _ = yaml.safe_load(f)
            # wandb config.yaml may have project under _wandb or at top level
        except Exception:
            pass

    # Try reading wandb-metadata.json for extra info
    meta_path = os.path.join(latest, "files", "wandb-metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            info["gpu_type"] = meta.get("gpu")
            info["host"] = meta.get("host")
            slurm = meta.get("slurm", {})
            if slurm:
                info["slurm_job_id"] = slurm.get("job_id")
                info["slurm_partition"] = slurm.get("job_partition")
                info["slurm_node"] = slurm.get("cluster_name")
        except Exception:
            pass

    # Try reading wandb-summary.json for metrics
    summary_path = os.path.join(latest, "files", "wandb-summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                info["summary"] = json.load(f)
        except Exception:
            pass

    return info


def _parse_job_sh(job_sh_content: str) -> Dict[str, Any]:
    """Extract SBATCH directives and training command args from job.sh."""
    info: Dict[str, Any] = {}
    for line in job_sh_content.splitlines():
        line = line.strip()
        m = re.match(r"#SBATCH\s+--(\S+?)=(.+)", line)
        if m:
            key, val = m.group(1), m.group(2)
            info[key] = val
        m2 = re.match(r"#SBATCH\s+--(\S+)\s+(\S+)", line)
        if m2 and m2.group(1) not in info:
            info[m2.group(1)] = m2.group(2)
    return info


def _find_training_params_file(run_dir: str, job_sh_content: str) -> Optional[str]:
    """Find the training parameters JSON path referenced in job.sh."""
    # Look for --training_parameters_path=... in the job.sh
    match = re.search(r"--training_parameters_path[= ](\S+)", job_sh_content)
    if match:
        rel = match.group(1).strip("'\"")
        full = os.path.join(run_dir, rel)
        if os.path.exists(full):
            return full
    # Fallback: look for first json in training_config/
    tc_dir = os.path.join(run_dir, "training_config")
    if os.path.isdir(tc_dir):
        jsons = sorted(glob.glob(os.path.join(tc_dir, "*.json")))
        if jsons:
            return jsons[0]
    return None


def _extract_training_args(job_sh_content: str) -> Dict[str, Any]:
    """Extract CLI training arguments from the first python -m graphphysics.train block."""
    args: Dict[str, Any] = {}
    # Collect key=value flags
    for m in re.finditer(r"--(\w+)[= ]([^\s\\]+)", job_sh_content):
        key, val = m.group(1), m.group(2)
        # Try numeric conversion
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        args[key] = val
    # Boolean flags (no value)
    for m in re.finditer(
        r"--(no_edge_feature|resume_training|use_previous_data)\b", job_sh_content
    ):
        args[m.group(1)] = True
    return args


# ── Commands ────────────────────────────────────────────────────────────


@click.group()
def cli():
    """GNN experiment run manager."""
    pass


# ────────────────────────────── register ─────────────────────────────────


@cli.command()
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to the run folder.",
)
@click.option("--base-commit", default=None, help="graph-physics base commit hash.")
@click.option(
    "--base-ref", default=None, help="Path to clean graph-physics checkout for diff."
)
@click.option("--notes", default="", help="Free-text notes.")
@click.option(
    "--params-file",
    default=None,
    help="Override: path to parameters JSON inside run dir.",
)
def register(
    run_dir: str,
    base_commit: Optional[str],
    base_ref: Optional[str],
    notes: str,
    params_file: Optional[str],
):
    """Snapshot all metadata from RUN_DIR into a new YAML record."""
    run_dir = os.path.abspath(run_dir)
    base_ref = base_ref or DEFAULT_BASE_REF

    # --- Read job.sh ---
    job_sh_path = os.path.join(run_dir, "job.sh")
    if not os.path.exists(job_sh_path):
        raise click.ClickException(f"job.sh not found in {run_dir}")
    with open(job_sh_path) as f:
        job_sh_content = f.read()

    sbatch_info = _parse_job_sh(job_sh_content)
    train_args = _extract_training_args(job_sh_content)

    # --- Find & load training parameters JSON ---
    if params_file:
        params_path = (
            os.path.join(run_dir, params_file)
            if not os.path.isabs(params_file)
            else params_file
        )
    else:
        params_path = _find_training_params_file(run_dir, job_sh_content)
    if not params_path or not os.path.exists(params_path):
        raise click.ClickException(
            "Training parameters JSON not found. Searched job.sh and training_config/."
        )

    with open(params_path) as f:
        params = json.load(f)

    # --- Base commit ---
    if not base_commit:
        bc_file = os.path.join(run_dir, "BASE_COMMIT")
        if os.path.exists(bc_file):
            with open(bc_file) as f:
                base_commit = f.read().strip()

    # --- Dataset info ---
    ds_label = _infer_dataset_label(params)
    ds = params.get("dataset", {})
    dataset_path = (
        ds.get("xdmf_folder") or ds.get("train_path") or ds.get("h5_path", "")
    )
    meta_path = ds.get("meta_path", "")

    # --- Generate run ID ---
    run_id = _generate_run_id(ds_label)

    # --- Source diff ---
    source_diff = _compute_diff(run_dir, base_ref)

    # --- wandb info ---
    wb = _find_wandb_run(run_dir)

    # --- Model info ---
    model = params.get("model", {})

    # --- Loss info ---
    loss_cfg = params.get("loss", {})

    # --- Noise info ---
    preproc = params.get("transformations", {}).get("preprocessing", {})
    noise_amp = preproc.get("noise")
    if not isinstance(noise_amp, list):
        noise_amp = [noise_amp] if noise_amp is not None else []

    # --- Features (best-effort from node_input_size + known field lists) ---
    input_size = model.get("node_input_size", 0)
    features_list = _infer_feature_names(run_dir, input_size)

    # --- Checkpoint ---
    save_name = train_args.get("model_save_name", "model")
    ckpt_path = os.path.join(run_dir, "checkpoints", f"{save_name}.ckpt")
    if not os.path.exists(ckpt_path):
        # Search for any .ckpt
        ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "*.ckpt"))
        ckpt_path = ckpts[0] if ckpts else ckpt_path

    # --- Metrics from wandb summary (pre-populate if available) ---
    summary = wb.get("summary", {})
    metrics_final = {
        "val_rollout_rmse": summary.get("val_all_rollout_rmse"),
        "val_1step_rmse": summary.get("val_1step_rmse"),
        "train_loss": summary.get("train_multiloss_step"),
    }

    # --- Build record ---
    record: Dict[str, Any] = {
        "run_id": run_id,
        "parent_run_id": None,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": wb.get("slurm_job_id") or sbatch_info.get("job-name"),
        #
        "base_commit": base_commit,
        "source_diff": source_diff,
        "run_dir": run_dir,
        #
        "dataset": {
            "name": "onecyl",
            "re_case": _infer_re_case(params),
            "variant": _infer_variant(params),
            "path": dataset_path,
            "meta_path": meta_path,
        },
        #
        "model": {
            "type": model.get("type"),
            "message_passing_num": model.get("message_passing_num"),
            "hidden_size": model.get("hidden_size"),
            "node_input_size": model.get("node_input_size"),
            "output_size": model.get("output_size"),
            "edge_input_size": model.get("edge_input_size", 0),
            "num_heads": model.get("num_heads"),
        },
        #
        "features": {
            "input_fields": features_list,
            "no_edge_feature": bool(train_args.get("no_edge_feature", False)),
        },
        #
        "loss": {
            "type": loss_cfg.get("type", ["l2loss"]),
            "weights": loss_cfg.get("weights", [1.0]),
            "gradient_method": loss_cfg.get("gradient_method"),
        },
        #
        "noise": {
            "amplitudes": noise_amp,
            "index_start": preproc.get("noise_index_start", []),
            "index_end": preproc.get("noise_index_end", []),
        },
        #
        "training": {
            "num_epochs": train_args.get("num_epochs", 10),
            "init_lr": train_args.get("init_lr", 0.001),
            "batch_size": train_args.get("batch_size", 2),
            "warmup": train_args.get("warmup", 1000),
            "seed": train_args.get("seed", 42),
            "num_workers": train_args.get("num_workers", 2),
            "prefetch_factor": train_args.get("prefetch_factor", 2),
            "project_name": train_args.get("project_name", "my_project"),
        },
        #
        "parameters_json": params,
        "job_sh": job_sh_content,
        #
        "slurm": {
            "partition": sbatch_info.get("partition"),
            "gres": sbatch_info.get("gres"),
            "gpu_type": wb.get("gpu_type"),
            "wall_clock_seconds": None,
            "node": sbatch_info.get("nodelist"),
        },
        #
        "checkpoint": {
            "path": ckpt_path,
            "save_name": save_name,
        },
        #
        "wandb": {
            "run_id": wb.get("run_id"),
            "url": None,
            "project": train_args.get("project_name", "onecyl_article"),
        },
        #
        "metrics": {
            "final": metrics_final,
            "best": {
                "val_rollout_rmse": None,
                "val_1step_rmse": None,
                "train_loss": None,
            },
        },
        #
        "notes": notes,
    }

    path = _save_record(record)
    console.print(f"[green]Registered[/green] {run_id}  →  {path}")
    click.echo(run_id)  # machine-readable output on last line
    return run_id


def _infer_feature_names(run_dir: str, input_size: int) -> List[str]:
    """Best-effort inference of feature names from the onecyl.py build_features."""
    onecyl_path = os.path.join(run_dir, "graphphysics", "external", "onecyl.py")
    features: List[str] = []
    if os.path.exists(onecyl_path):
        with open(onecyl_path) as f:
            src = f.read()
        # Heuristic: look for variable assignments that feed into torch.cat
        if "current_velocity" in src and "levelset" in src:
            if "# levelset" in src or "levelset" not in src.split("torch.cat")[0]:
                # levelset commented out
                features = ["velocity_xy", "pressure", "pos_xy", "nodetype"]
            else:
                features = ["velocity_xy", "pressure", "levelset", "pos_xy", "nodetype"]
        elif "current_velocity" in src:
            features = ["velocity_xy", "pressure", "pos_xy", "nodetype"]
    if not features:
        features = [f"feature_{i}" for i in range(input_size)]
    return features


# ──────────────────────────── update ─────────────────────────────────────


@cli.command()
@click.option("--run-id", required=True, help="Run ID to update.")
def update(run_id: str):
    """Pull metrics from wandb / SLURM accounting into the record."""
    record = _load_record(run_id)

    # --- wandb metrics (offline: read from summary file) ---
    wb_run_id = record.get("wandb", {}).get("run_id")
    run_dir = record.get("run_dir", "")

    # Try offline summary first
    summary = _read_wandb_summary_offline(run_dir, wb_run_id)
    if summary:
        record["metrics"]["final"]["val_rollout_rmse"] = summary.get(
            "val_all_rollout_rmse"
        )
        record["metrics"]["final"]["val_1step_rmse"] = summary.get("val_1step_rmse")
        record["metrics"]["final"]["train_loss"] = summary.get("train_multiloss_step")

    # Try online wandb API for best metrics
    if wb_run_id:
        try:
            best = _fetch_wandb_best_metrics(
                record["wandb"].get("project", ""), wb_run_id
            )
            if best:
                record["metrics"]["best"].update(best)
                # Also fill URL
                record["wandb"]["url"] = best.get("url")
        except Exception as e:
            console.print(f"[yellow]wandb API unavailable:[/yellow] {e}")

    # --- SLURM accounting ---
    slurm_job_id = record.get("slurm_job_id")
    if slurm_job_id:
        slurm_info = _fetch_sacct_info(str(slurm_job_id))
        if slurm_info:
            record["slurm"]["wall_clock_seconds"] = slurm_info.get("elapsed_seconds")
            record["slurm"]["gpu_type"] = slurm_info.get("gpu_type") or record[
                "slurm"
            ].get("gpu_type")
            record["slurm"]["partition"] = slurm_info.get("partition") or record[
                "slurm"
            ].get("partition")
            record["slurm"]["node"] = slurm_info.get("node") or record["slurm"].get(
                "node"
            )

    # --- Also re-read wandb-metadata.json for GPU info ---
    wb_info = _find_wandb_run(run_dir)
    if wb_info.get("gpu_type"):
        record["slurm"]["gpu_type"] = wb_info["gpu_type"]

    _save_record(record)
    console.print(f"[green]Updated[/green] {run_id}")


def _read_wandb_summary_offline(
    run_dir: str, wb_run_id: Optional[str]
) -> Optional[Dict]:
    """Read wandb-summary.json from offline run directory."""
    wandb_dir = os.path.join(run_dir, "wandb")
    if not os.path.isdir(wandb_dir):
        return None

    # Find the matching offline dir
    candidates = sorted(glob.glob(os.path.join(wandb_dir, "offline-run-*")))
    candidates += sorted(glob.glob(os.path.join(wandb_dir, "run-*")))
    if wb_run_id:
        matching = [d for d in candidates if wb_run_id in os.path.basename(d)]
        if matching:
            candidates = matching

    for d in reversed(candidates):
        summary_path = os.path.join(d, "files", "wandb-summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                return json.load(f)
    return None


def _fetch_wandb_best_metrics(project: str, wb_run_id: str) -> Optional[Dict]:
    """Fetch best metrics from wandb API. Returns None if API unavailable."""
    try:
        import wandb  # noqa: F811

        api = wandb.Api()
        # Try common entity patterns
        run = None
        for entity in [None, os.environ.get("WANDB_ENTITY")]:
            try:
                path = (
                    f"{entity}/{project}/{wb_run_id}"
                    if entity
                    else f"{project}/{wb_run_id}"
                )
                run = api.run(path)
                break
            except Exception:
                continue
        if run is None:
            return None

        url = run.url
        history = run.history(
            keys=[
                "val_all_rollout_rmse",
                "val_1step_rmse",
                "train_multiloss_step",
            ]
        )

        best: Dict[str, Any] = {"url": url}
        if "val_all_rollout_rmse" in history.columns:
            col = history["val_all_rollout_rmse"].dropna()
            if len(col):
                best["val_rollout_rmse"] = float(col.min())
        if "val_1step_rmse" in history.columns:
            col = history["val_1step_rmse"].dropna()
            if len(col):
                best["val_1step_rmse"] = float(col.min())
        if "train_multiloss_step" in history.columns:
            col = history["train_multiloss_step"].dropna()
            if len(col):
                best["train_loss"] = float(col.min())

        return best
    except ImportError:
        return None
    except Exception:
        return None


def _fetch_sacct_info(job_id: str) -> Optional[Dict]:
    """Fetch SLURM job info via sacct."""
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,Partition,NodeList,Elapsed,AllocGRES,State",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        info: Dict[str, Any] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 6:
                continue
            jid = parts[0]
            # Take the batch/main row (no .0 suffix, or exact match)
            if jid == job_id or jid == f"{job_id}.batch":
                info["partition"] = parts[1] or info.get("partition")
                info["node"] = parts[2] or info.get("node")
                elapsed = parts[3]  # HH:MM:SS or D-HH:MM:SS
                info["elapsed_seconds"] = _elapsed_to_seconds(elapsed)
                info["gpu_type"] = parts[4] if parts[4] else None
        return info if info else None
    except Exception:
        return None


def _elapsed_to_seconds(elapsed: str) -> Optional[int]:
    """Convert SLURM elapsed string (D-HH:MM:SS or HH:MM:SS) to seconds."""
    if not elapsed:
        return None
    days = 0
    if "-" in elapsed:
        d, elapsed = elapsed.split("-", 1)
        days = int(d)
    parts = elapsed.split(":")
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + s


# ──────────────────────────── relaunch ───────────────────────────────────


@cli.command()
@click.option("--run-id", required=True, help="Run ID to relaunch.")
@click.option(
    "--override-dataset", default=None, help="Override dataset path in parameters."
)
@click.option("--override-notes", default=None, help="Override notes for the new run.")
@click.option("--dry-run", is_flag=True, help="Print plan without executing.")
def relaunch(
    run_id: str,
    override_dataset: Optional[str],
    override_notes: Optional[str],
    dry_run: bool,
):
    """Reconstruct a run folder from a record and re-submit via sbatch."""
    record = _load_record(run_id)
    base_ref = os.environ.get("GRAPH_PHYSICS_REF", DEFAULT_BASE_REF)

    if not os.path.isdir(base_ref):
        if dry_run:
            console.print(
                f"[yellow]Warning: base ref not found at {base_ref}; proceeding with dry-run output only.[/yellow]"
            )
        else:
            raise click.ClickException(
                f"Clean graph-physics reference not found at {base_ref}. "
                f"Set GRAPH_PHYSICS_REF env var to the correct path."
            )

    # --- Create new run folder ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder_name = f"relaunch_{run_id}_{timestamp}"
    parent_dir = (
        os.path.dirname(record["run_dir"]) if record.get("run_dir") else os.getcwd()
    )
    new_run_dir = os.path.join(parent_dir, new_folder_name)

    if dry_run:
        console.print(f"[cyan]DRY RUN[/cyan] — would create: {new_run_dir}")
    else:
        os.makedirs(new_run_dir, exist_ok=True)

    # --- Copy clean base ---
    if dry_run:
        console.print(f"  cp -r {base_ref}/* → {new_run_dir}/")
    else:
        # Copy base, excluding .git
        for item in os.listdir(base_ref):
            if item == ".git":
                continue
            src = os.path.join(base_ref, item)
            dst = os.path.join(new_run_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    # --- Apply stored diff ---
    source_diff = record.get("source_diff", "")
    if (
        source_diff
        and not source_diff.startswith("# No differences")
        and not source_diff.startswith("# WARNING")
    ):
        if dry_run:
            console.print(f"  patch -p1 < (stored diff, {len(source_diff)} chars)")
        else:
            # Write diff to temp file and apply
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".patch", delete=False
            ) as tmp:
                tmp.write(source_diff)
                tmp_path = tmp.name
            try:
                # The diff was computed with base_ref and run_dir as absolute paths.
                # We need to strip the leading path components.
                subprocess.run(
                    ["patch", "-p0", "--no-backup-if-mismatch", "-d", "/"],
                    input=source_diff,
                    capture_output=True,
                    text=True,
                )
                # Also try with -p1 relative to new_run_dir
                # Replace old paths in diff
                adapted_diff = source_diff
                original_base = None
                for line in source_diff.splitlines()[:20]:
                    m = re.match(r"^--- (.+?)(?:\t|$)", line)
                    if m:
                        p = m.group(1)
                        # Remove trailing filename to get the base dir
                        original_base = os.path.dirname(p)
                        break

                if original_base:
                    adapted_diff = source_diff.replace(
                        record.get("run_dir", ""), new_run_dir
                    ).replace(base_ref, new_run_dir)

                result = subprocess.run(
                    ["patch", "-p0", "--no-backup-if-mismatch", "-d", "/"],
                    input=adapted_diff,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    console.print(
                        f"[yellow]Warning: patch had issues:[/yellow] {result.stderr[:200]}"
                    )
            finally:
                os.unlink(tmp_path)

    # --- Write parameters JSON ---
    params = record.get("parameters_json", {})
    if override_dataset:
        if "xdmf_folder" in params.get("dataset", {}):
            params["dataset"]["xdmf_folder"] = override_dataset
        elif "train_path" in params.get("dataset", {}):
            params["dataset"]["train_path"] = override_dataset

    # Determine params filename from original record
    params_filename = None
    job_sh_content = record.get("job_sh", "")
    m = re.search(r"--training_parameters_path[= ](\S+)", job_sh_content)
    if m:
        params_filename = m.group(1).strip("'\"")
    if not params_filename:
        params_filename = "training_config/params.json"

    if dry_run:
        console.print(f"  write {params_filename}")
    else:
        params_path = os.path.join(new_run_dir, params_filename)
        os.makedirs(os.path.dirname(params_path), exist_ok=True)
        with open(params_path, "w") as f:
            json.dump(params, f, indent=4)

    # --- Write job.sh ---
    if dry_run:
        console.print("  write job.sh")
    else:
        with open(os.path.join(new_run_dir, "job.sh"), "w") as f:
            f.write(job_sh_content)

    # --- Submit ---
    sbatch_cmd = f"cd {new_run_dir} && sbatch job.sh"
    if dry_run:
        console.print(f"  {sbatch_cmd}")
        console.print("[cyan]DRY RUN complete — no submission.[/cyan]")
        return

    result = subprocess.run(
        ["sbatch", "job.sh"],
        capture_output=True,
        text=True,
        cwd=new_run_dir,
    )

    new_slurm_id = None
    if result.returncode == 0:
        m = re.search(r"(\d+)", result.stdout)
        if m:
            new_slurm_id = m.group(1)
        console.print(f"[green]Submitted[/green] SLURM job {new_slurm_id}")
    else:
        console.print(f"[red]sbatch failed:[/red] {result.stderr}")

    # --- Register new run as child ---
    notes = override_notes or f"Relaunched from {run_id}"
    # Write a temporary params file and job.sh so register can find them
    new_record = _register_child(
        new_run_dir=new_run_dir,
        parent_run_id=run_id,
        params=params,
        job_sh_content=job_sh_content,
        slurm_job_id=new_slurm_id,
        notes=notes,
        base_commit=record.get("base_commit"),
    )
    console.print(
        f"[green]Registered child[/green] {new_record['run_id']}  (parent: {run_id})"
    )


def _register_child(
    new_run_dir: str,
    parent_run_id: str,
    params: Dict,
    job_sh_content: str,
    slurm_job_id: Optional[str],
    notes: str,
    base_commit: Optional[str],
) -> Dict[str, Any]:
    """Create a record for a relaunched child run."""
    ds_label = _infer_dataset_label(params)
    run_id = _generate_run_id(ds_label)
    model = params.get("model", {})
    loss_cfg = params.get("loss", {})
    preproc = params.get("transformations", {}).get("preprocessing", {})
    noise_amp = preproc.get("noise")
    if not isinstance(noise_amp, list):
        noise_amp = [noise_amp] if noise_amp is not None else []

    sbatch_info = _parse_job_sh(job_sh_content)
    train_args = _extract_training_args(job_sh_content)

    record: Dict[str, Any] = {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": slurm_job_id,
        "base_commit": base_commit,
        "source_diff": "# Same as parent run; see parent record.\n",
        "run_dir": new_run_dir,
        "dataset": {
            "name": "onecyl",
            "re_case": _infer_re_case(params),
            "variant": _infer_variant(params),
            "path": params.get("dataset", {}).get("xdmf_folder")
            or params.get("dataset", {}).get("train_path", ""),
            "meta_path": params.get("dataset", {}).get("meta_path", ""),
        },
        "model": {
            "type": model.get("type"),
            "message_passing_num": model.get("message_passing_num"),
            "hidden_size": model.get("hidden_size"),
            "node_input_size": model.get("node_input_size"),
            "output_size": model.get("output_size"),
            "edge_input_size": model.get("edge_input_size", 0),
            "num_heads": model.get("num_heads"),
        },
        "features": {
            "input_fields": _infer_feature_names(
                new_run_dir, model.get("node_input_size", 0)
            ),
            "no_edge_feature": bool(train_args.get("no_edge_feature", False)),
        },
        "loss": {
            "type": loss_cfg.get("type", ["l2loss"]),
            "weights": loss_cfg.get("weights", [1.0]),
            "gradient_method": loss_cfg.get("gradient_method"),
        },
        "noise": {
            "amplitudes": noise_amp,
            "index_start": preproc.get("noise_index_start", []),
            "index_end": preproc.get("noise_index_end", []),
        },
        "training": {
            "num_epochs": train_args.get("num_epochs", 10),
            "init_lr": train_args.get("init_lr", 0.001),
            "batch_size": train_args.get("batch_size", 2),
            "warmup": train_args.get("warmup", 1000),
            "seed": train_args.get("seed", 42),
            "num_workers": train_args.get("num_workers", 2),
            "prefetch_factor": train_args.get("prefetch_factor", 2),
            "project_name": train_args.get("project_name", "my_project"),
        },
        "parameters_json": params,
        "job_sh": job_sh_content,
        "slurm": {
            "partition": sbatch_info.get("partition"),
            "gres": sbatch_info.get("gres"),
            "gpu_type": None,
            "wall_clock_seconds": None,
            "node": sbatch_info.get("nodelist"),
        },
        "checkpoint": {
            "path": os.path.join(
                new_run_dir,
                "checkpoints",
                f"{train_args.get('model_save_name', 'model')}.ckpt",
            ),
            "save_name": train_args.get("model_save_name", "model"),
        },
        "wandb": {
            "run_id": None,
            "url": None,
            "project": train_args.get("project_name", "onecyl_article"),
        },
        "metrics": {
            "final": {
                "val_rollout_rmse": None,
                "val_1step_rmse": None,
                "train_loss": None,
            },
            "best": {
                "val_rollout_rmse": None,
                "val_1step_rmse": None,
                "train_loss": None,
            },
        },
        "notes": notes,
    }

    _save_record(record)
    return record


# ──────────────────────────── compare ────────────────────────────────────


@cli.command()
@click.option(
    "--dataset",
    "dataset_filter",
    default=None,
    help="Filter by dataset label (e.g. re100_full, Re2).",
)
@click.option(
    "--sort-by", "sort_key", default="val/rollout_rmse", help="Metric to sort by."
)
@click.option("--top", "top_n", default=None, type=int, help="Show only top N runs.")
@click.option("--all-metrics", is_flag=True, help="Show all metric columns.")
def compare(
    dataset_filter: Optional[str],
    sort_key: str,
    top_n: Optional[int],
    all_metrics: bool,
):
    """Print a ranked table of all runs."""
    records = []
    for yaml_file in sorted(REGISTRY_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            rec = yaml.safe_load(f)
        if rec:
            records.append(rec)

    if not records:
        console.print("[yellow]No runs registered yet.[/yellow]")
        return

    # --- Filter ---
    if dataset_filter:
        filt = dataset_filter.lower()
        records = [
            r
            for r in records
            if filt in r.get("run_id", "").lower()
            or filt in (r.get("dataset", {}).get("re_case", "") or "").lower()
            or filt in (r.get("dataset", {}).get("variant", "") or "").lower()
        ]

    # --- Sort ---
    sort_map = {
        "val/rollout_rmse": lambda r: r.get("metrics", {})
        .get("final", {})
        .get("val_rollout_rmse")
        or float("inf"),
        "val/1step_rmse": lambda r: r.get("metrics", {})
        .get("final", {})
        .get("val_1step_rmse")
        or float("inf"),
        "train/loss": lambda r: r.get("metrics", {}).get("final", {}).get("train_loss")
        or float("inf"),
        "created_at": lambda r: r.get("created_at", ""),
    }
    key_fn = sort_map.get(sort_key, sort_map["val/rollout_rmse"])
    records.sort(key=key_fn)

    if top_n:
        records = records[:top_n]

    # --- Table ---
    table = Table(title="GNN Experiment Runs", show_lines=True)
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Date", style="dim")
    table.add_column("Dataset", no_wrap=True)
    table.add_column("Loss terms")
    table.add_column("val/rollout", justify="right", style="green")
    table.add_column("val/1step", justify="right", style="green")
    table.add_column("train/loss", justify="right")
    if all_metrics:
        table.add_column("best roll", justify="right", style="bold green")
        table.add_column("best 1step", justify="right", style="bold green")
    table.add_column("GPU")
    table.add_column("Wall (h)", justify="right")
    table.add_column("wandb")
    table.add_column("Notes", max_width=30)

    for r in records:
        final = r.get("metrics", {}).get("final", {})
        best = r.get("metrics", {}).get("best", {})
        ds = r.get("dataset", {})
        slurm = r.get("slurm", {})
        wb = r.get("wandb", {})
        loss_terms = r.get("loss", {}).get("type", [])
        loss_short = (
            ",".join(
                t.replace("l2loss", "L2")
                .replace("gradient", "G")
                .replace("convection", "C")
                .replace("divergence", "D")
                .replace("pressure", "P")
                for t in loss_terms
            )
            if loss_terms
            else "-"
        )

        wall_h = ""
        if slurm.get("wall_clock_seconds"):
            wall_h = f"{slurm['wall_clock_seconds'] / 3600:.1f}"

        row = [
            r.get("run_id", "?"),
            (r.get("created_at", "")[:10]),
            f"{ds.get('re_case', '?')} {ds.get('variant', '?')}",
            loss_short,
            _fmt_metric(final.get("val_rollout_rmse")),
            _fmt_metric(final.get("val_1step_rmse")),
            _fmt_metric(final.get("train_loss")),
        ]
        if all_metrics:
            row += [
                _fmt_metric(best.get("val_rollout_rmse")),
                _fmt_metric(best.get("val_1step_rmse")),
            ]
        row += [
            slurm.get("gpu_type", "-") or "-",
            wall_h or "-",
            wb.get("run_id", "-") or "-",
            (r.get("notes", "") or "")[:30],
        ]

        table.add_row(*row)

    console.print(table)


def _fmt_metric(val) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        if abs(val) < 0.001:
            return f"{val:.2e}"
        return f"{val:.5f}"
    return str(val)


# ──────────────────────────── sync-sheet ─────────────────────────────────


@cli.command("sync-sheet")
def sync_sheet():
    """TODO: Sync run records to a Google Sheet or Excel file."""
    console.print("[yellow]sync-sheet is not yet implemented.[/yellow]")
    # TODO: Implement Google Sheets / Excel export
    #   - Read all YAML records from registry/
    #   - Map to spreadsheet columns
    #   - Use gspread or openpyxl to push
    raise SystemExit(0)


# ── Entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
