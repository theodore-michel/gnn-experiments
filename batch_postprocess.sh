#!/bin/bash
###############################################################################
# Local Batch Postprocessing Script
#
# Process multiple configs sequentially on local machine/login node.
# This is equivalent to the SLURM script but without job submission overhead.
#
# Usage:
#   bash batch_postprocess.sh config1.json config2.json config3.json
#
# The script will:
#   1. Validate all configs exist
#   2. Process each sequentially with run_postprocess.sh
#   3. Log all output and timing to stdout/stderr
#   4. Stop on first error (set -e)
#
# Example:
#   bash batch_postprocess.sh \
#     postprocess/config.model_A.json \
#     postprocess/config.model_B.json \
#     postprocess/config.model_C.json
#
###############################################################################

set -euo pipefail

if [[ $# -eq 0 ]]; then
  cat >&2 <<'EOF'
Usage: bash batch_postprocess.sh config1.json [config2.json ...]

Examples:
  bash batch_postprocess.sh postprocess/config.model_A.json
  bash batch_postprocess.sh \
    postprocess/config.model_A.json \
    postprocess/config.model_B.json \
    postprocess/config.model_C.json

Configs will be processed sequentially. Stop on first error.
EOF
  exit 1
fi

# Activate conda if available
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate gnnpostprocess || conda activate graph || true
fi

# Validate all configs exist before processing
echo "Validating configs..."
for config in "$@"; do
  if [[ ! -f "$config" ]]; then
    echo "[ERROR] Config file not found: $config" >&2
    exit 1
  fi
  echo "  ✓ $config"
done

echo ""
echo "Processing ${#@} config(s) sequentially..."
echo ""

# Track failure count
failed=0

# Process each config
for config in "$@"; do
  echo "================================================================================"
  echo "Config: $config"
  echo "Start:  $(date)"
  echo "================================================================================"
  
  if bash run_postprocess.sh "$config"; then
    echo "================================================================================"
    echo "✓ Completed: $config"
    echo "End:     $(date)"
    echo "================================================================================"
  else
    echo "================================================================================"
    echo "✗ FAILED: $config"
    echo "================================================================================"
    ((failed++))
    if [[ $failed -eq 1 ]]; then
      echo "[ERROR] Stopping on first failure. Fix and rerun." >&2
      exit 1
    fi
  fi
  
  echo ""
done

echo "================================================================================"
echo "All ${#@} configs processed successfully!"
echo "================================================================================"
