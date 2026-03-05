#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# launch.sh — Submit a GNN training run with automatic experiment tracking.
#
# Usage:
#   ./launch.sh <run-folder> [params-file] [--base-commit SHA] [--notes "..."]
#
# Arguments:
#   run-folder   Path to the run folder (copy of graph-physics with edits).
#   params-file  (Optional) Relative path to the training parameters JSON
#                inside the run folder. If omitted, inferred from job.sh.
#
# Example:
#   ./launch.sh ../onecyl-T-Re2-VPLN-lgud training_config/onecyl_Re2.json \
#       --notes "Baseline Re=100 with physics loss"
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_MANAGER="${SCRIPT_DIR}/run_manager.py"

# ── Parse arguments ─────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run-folder> [params-file] [--base-commit SHA] [--notes TEXT]"
    exit 1
fi

RUN_DIR="$(cd "$1" && pwd)"
shift

PARAMS_FILE=""
BASE_COMMIT=""
NOTES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-commit)
            BASE_COMMIT="$2"; shift 2 ;;
        --notes)
            NOTES="$2"; shift 2 ;;
        -*)
            echo "Unknown option: $1"; exit 1 ;;
        *)
            PARAMS_FILE="$1"; shift ;;
    esac
done

# ── Activate conda environment ──────────────────────────────────────────
# Try common conda init paths
if [[ -z "${CONDA_EXE:-}" ]]; then
    for candidate in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh" \
        "/scratch-fast/$(whoami)/miniconda3/etc/profile.d/conda.sh"; do
        if [[ -f "$candidate" ]]; then
            source "$candidate"
            break
        fi
    done
fi
conda activate graph

# ── Register the run (snapshot metadata before submission) ──────────────
REGISTER_ARGS=( --run-dir "$RUN_DIR" )
[[ -n "$BASE_COMMIT" ]] && REGISTER_ARGS+=( --base-commit "$BASE_COMMIT" )
[[ -n "$NOTES" ]]       && REGISTER_ARGS+=( --notes "$NOTES" )
[[ -n "$PARAMS_FILE" ]] && REGISTER_ARGS+=( --params-file "$PARAMS_FILE" )

echo ">> Registering run from ${RUN_DIR} ..."
RUN_ID=$(python "$RUN_MANAGER" register "${REGISTER_ARGS[@]}" | tail -1)
echo ">> Registered as: ${RUN_ID}"

# ── Submit via sbatch ───────────────────────────────────────────────────
echo ">> Submitting job.sh from ${RUN_DIR} ..."
cd "$RUN_DIR"
SBATCH_OUTPUT=$(sbatch job.sh 2>&1)
echo ">> sbatch output: ${SBATCH_OUTPUT}"

# Extract SLURM job ID (e.g. "Submitted batch job 123456")
SLURM_JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP '\d+' | tail -1)

if [[ -z "$SLURM_JOB_ID" ]]; then
    echo ">> WARNING: could not parse SLURM job ID from sbatch output."
else
    echo ">> SLURM Job ID: ${SLURM_JOB_ID}"

    # Write SLURM job ID back into the run record
    RECORD_FILE="${SCRIPT_DIR}/registry/${RUN_ID}.yaml"
    if [[ -f "$RECORD_FILE" ]]; then
        # Use python to update the YAML (avoids fragile sed on multiline YAML)
        python -c "
import yaml, sys
path = sys.argv[1]
job_id = sys.argv[2]
with open(path) as f:
    rec = yaml.safe_load(f)
rec['slurm_job_id'] = job_id
with open(path, 'w') as f:
    yaml.dump(rec, f, default_flow_style=False, sort_keys=False, width=120)
" "$RECORD_FILE" "$SLURM_JOB_ID"
        echo ">> Updated ${RUN_ID} with slurm_job_id=${SLURM_JOB_ID}"
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Run ID     : ${RUN_ID}"
echo "  SLURM Job  : ${SLURM_JOB_ID:-unknown}"
echo "  Run Dir    : ${RUN_DIR}"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "After the job completes, update metrics with:"
echo "  python ${RUN_MANAGER} update --run-id ${RUN_ID}"
