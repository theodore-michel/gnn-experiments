#!/bin/bash
#
#SBATCH --job-name=postprocess_init_iter
#SBATCH --output=postprocess_init_iter.log
#
#SBATCH --nodes 1
#SBATCH --ntasks 32
#SBATCH --ntasks-per-node=32
#SBATCH --ntasks-per-core=1
#SBATCH --threads-per-core=1
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#SBATCH --time=2:00:00
#SBATCH --begin=now
#SBATCH --chdir=/scratch-big/tmichel/GNN/RUNS/NEWCYL/ONECYL_ARTICLE/gnn-experiments

set -euo pipefail

echo "==> Init-iter postprocess SLURM job started at $(date)"
echo "==> Working directory: $(pwd)"
echo "==> SLURM Job ID: ${SLURM_JOB_ID:-N/A}"
echo ""

# The target script handles environment activation and the full rebuild.
bash rerun_postprocess_init_iter.sh

echo ""
echo "==> Init-iter postprocess SLURM job completed at $(date)"