#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_postprocess.sh — Pipeline launcher for GNN prediction post-processing.
#
# Runs error computation, force computation, and plotting in sequence.
# Designed for use on HPC (SLURM) or locally.
#
# Usage:
#   bash run_postprocess.sh -p config.json [-d results_dir] [--forces] [--compare]
#
# Flags:
#   -p, --parameters   JSON config file (required)
#   -d, --directory     Output root directory (default: ./postprocess_results)
#   --forces            Also compute drag / lift forces
#   --truth-folder      Ground-truth XDMF folder for force comparison
#   --sensor-errors     Also compute sensor-level cumulated errors
#   --no-sensor-errors  Disable sensor-level extraction and plots
#   --metric            Error metric for sensors: AE (default) or SE
#   --article-style     Use minimal annotation for camera-ready figures
#   --format            Plot format(s): png (default), pdf, svg
#   --workers           Thread count for force computation (default: 4)
#   --load-data         Re-use cached sensor JSON instead of re-extracting
#   --skip-errors       Skip error computation (plot only)
#   --skip-forces       Skip force computation (plot only)
#   --skip-plots        Skip plotting
#   --skip-xdmf         Skip processed-XDMF export
#   --compact-sensors   Use compact sensor plot layout (3x2)
#   --compare           Compare multiple models from config (default: auto)
# ---------------------------------------------------------------------------
set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────
CONFIG=""
OUTDIR="./postprocess_results"
DO_FORCES=false
TRUTH_FOLDER=""
SENSOR_ERRORS=true
METRIC="AE"
ARTICLE_STYLE=false
PLOT_FMT="png"
WORKERS=4
LOAD_DATA=false
SKIP_ERRORS=false
SKIP_FORCES=false
SKIP_PLOTS=false
SKIP_XDMF=false
COMPACT_SENSORS=false
COMPARE_MODE="auto"

# ── parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--parameters)    CONFIG="$2"; shift 2 ;;
        -d|--directory)     OUTDIR="$2"; shift 2 ;;
        --forces)           DO_FORCES=true; shift ;;
        --truth-folder)     TRUTH_FOLDER="$2"; shift 2 ;;
        --sensor-errors)    SENSOR_ERRORS=true; shift ;;
        --no-sensor-errors) SENSOR_ERRORS=false; shift ;;
        --metric)           METRIC="$2"; shift 2 ;;
        --article-style)    ARTICLE_STYLE=true; shift ;;
        --format)           PLOT_FMT="$2"; shift 2 ;;
        --workers)          WORKERS="$2"; shift 2 ;;
        --load-data)        LOAD_DATA=true; shift ;;
        --skip-errors)      SKIP_ERRORS=true; shift ;;
        --skip-forces)      SKIP_FORCES=true; shift ;;
        --skip-plots)       SKIP_PLOTS=true; shift ;;
        --skip-xdmf)        SKIP_XDMF=true; shift ;;
        --compact-sensors)  COMPACT_SENSORS=true; shift ;;
        --compare)          COMPARE_MODE="true"; shift ;;
        -h|--help)
            head -28 "$0" | tail -25
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --parameters / -p is required."
    exit 1
fi

# ── activate conda ────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate graph 2>/dev/null || true
fi

# Resolve script directory (location of this file)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Detect compare mode from config when not forced
if [[ "${COMPARE_MODE}" == "auto" ]]; then
    if grep -Eq '"name"[[:space:]]*:[[:space:]]*\[' "${CONFIG}"; then
        COMPARE_MODE="true"
    else
        COMPARE_MODE="false"
    fi
fi

echo "============================================================"
echo "  GNN Post-processing Pipeline"
echo "  Config : ${CONFIG}"
echo "  Output : ${OUTDIR}"
echo "  Compare: ${COMPARE_MODE}"
echo "============================================================"
mkdir -p "${OUTDIR}"

# ── 0. Processed XDMF export ─────────────────────────────────────────────
if [[ "$SKIP_XDMF" == false ]]; then
    echo ""
    echo "── Step 0: Exporting processed XDMF files ──"
    CMD="python -m postprocess.metrics.process_xdmf -p ${CONFIG} -d ${OUTDIR}"
    echo "  Running: ${CMD}"
    eval "${CMD}"
else
    echo "── Skipping processed XDMF export ──"
fi

# ── 1. Error computation ─────────────────────────────────────────────────
if [[ "$SKIP_ERRORS" == false ]]; then
    echo ""
    echo "── Step 1: Computing RMSE errors ──"
    CMD="python -m postprocess.metrics.compute_errors -p ${CONFIG} -d ${OUTDIR}"
    if [[ "$SENSOR_ERRORS" == true ]]; then
        CMD="${CMD} --sensor-errors --metric ${METRIC}"
    fi
    if [[ "$LOAD_DATA" == true ]]; then
        CMD="${CMD} --load-data"
    fi
    echo "  Running: ${CMD}"
    eval "${CMD}"
else
    echo "── Skipping error computation ──"
fi

# ── 2. Force computation ─────────────────────────────────────────────────
if [[ "$DO_FORCES" == true && "$SKIP_FORCES" == false ]]; then
    echo ""
    echo "── Step 2: Computing drag / lift forces ──"
    CMD="python -m postprocess.metrics.compute_forces -p ${CONFIG} -d ${OUTDIR} --workers ${WORKERS}"
    if [[ -n "$TRUTH_FOLDER" ]]; then
        CMD="${CMD} --truth-folder ${TRUTH_FOLDER}"
    fi
    echo "  Running: ${CMD}"
    eval "${CMD}"
elif [[ "$DO_FORCES" == false ]]; then
    echo "── Skipping force computation (pass --forces to enable) ──"
fi

# ── 3. Plotting ──────────────────────────────────────────────────────────
if [[ "$SKIP_PLOTS" == false ]]; then
    echo ""
    echo "── Step 3: Generating plots ──"
    CMD="python -m postprocess.visualization.plot_results -d ${OUTDIR} --format ${PLOT_FMT}"
    if [[ "$ARTICLE_STYLE" == true ]]; then
        CMD="${CMD} --article-style"
    fi
    if [[ "$DO_FORCES" == true && -n "$TRUTH_FOLDER" ]]; then
        CMD="${CMD} --truth-dir ${OUTDIR}/forces/truth"
    fi
    if [[ "$COMPACT_SENSORS" == true ]]; then
        CMD="${CMD} --compact-sensors"
    fi
    echo "  Running: ${CMD}"
    eval "${CMD}"
else
    echo "── Skipping plots ──"
fi

echo ""
echo "============================================================"
echo "  Pipeline complete. Results in: ${OUTDIR}"
echo "============================================================"
