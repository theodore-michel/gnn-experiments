#!/bin/bash
#
#SBATCH --job-name=gnn_postprocess
#SBATCH --output=slurm_postprocess_%j.log
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-core=1
#SBATCH --threads-per-core=1
#SBATCH --time=06:00:00

###############################################################################
# SLURM Batch Postprocessing Script
#
# Submits multiple postprocessing configs to run SERIALLY on a single SLURM job.
# This avoids I/O contention and makes efficient use of cluster resources.
#
# Usage:
#   bash submit_postprocess.slurm.sh config1.json config2.json config3.json
#
# The script will:
#   1. Create a temporary batch file listing all configs
#   2. Submit it to SLURM
#   3. Configs are processed sequentially (each waits for previous to complete)
#   4. Output logged to slurm_postprocess_<job_id>.log
#
# To monitor:
#   tail -f slurm_postprocess_<job_id>.log
#
# To customize resource allocation, edit the SBATCH directives:
#   --nodes            Number of compute nodes (1 for serial postprocessing)
#   --ntasks           Total number of tasks (typically 32 for full node)
#   --time             Wall-clock time limit (e.g., 06:00:00 = 6 hours)
#   --partition/--qos  Queue assignment (adjust to your cluster)
#
###############################################################################

set -euo pipefail

if [[ $# -eq 0 ]]; then
  cat >&2 <<'EOF'
Usage: bash submit_postprocess.slurm.sh config1.json [config2.json ...]

Examples:
  bash submit_postprocess.slurm.sh postprocess/config.model_A.json
  bash submit_postprocess.slurm.sh \
    postprocess/config.model_A.json \
    postprocess/config.model_B.json \
    postprocess/config.model_C.json

All configs will be processed sequentially in a single SLURM job.
EOF
  exit 1
fi

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate graph || true
fi

# Process each config sequentially
for config in "$@"; do
  if [[ ! -f "$config" ]]; then
    echo "[ERROR] Config file not found: $config" >&2
    exit 1
  fi
  
  echo ""
  echo "================================================================================"
  echo "Processing: $config"
  echo "Start time: $(date)"
  echo "================================================================================"
  
  bash run_postprocess.sh "$config" || {
    echo "[ERROR] Failed to process $config" >&2
    exit 1
  }
  
  echo "================================================================================"
  echo "Completed: $config"
  echo "End time: $(date)"
  echo "================================================================================"
  echo ""
done

echo "================================================================================"
echo "All configs processed successfully!"
echo "================================================================================"
