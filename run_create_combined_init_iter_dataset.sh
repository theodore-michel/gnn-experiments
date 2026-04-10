#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/scratch-fast/tmichel/miniconda3/envs/gnnpostprocess/bin/python}"
RES="${RES:-2 3 4}"

SOURCE_TEMPLATE="${SOURCE_TEMPLATE:-}"
if [[ -z "${SOURCE_TEMPLATE}" ]]; then
  SOURCE_TEMPLATE='/scratch-big/tmichel/GNN/DATASETS/dataset_onecyl_Re1e{re}_gmsh_trunc/predict'
fi

INIT_TEMPLATE="${INIT_TEMPLATE:-}"
if [[ -z "${INIT_TEMPLATE}" ]]; then
  INIT_TEMPLATE='/scratch-big/tmichel/GNN/RUNS/NEWCYL/ONECYL_ARTICLE/gnn-experiments/results_ReX_1step/onecyl_Re{re}_1step_VPLN/xdmf'
fi

OUTPUT_TEMPLATE="${OUTPUT_TEMPLATE:-}"
if [[ -z "${OUTPUT_TEMPLATE}" ]]; then
  OUTPUT_TEMPLATE='/scratch-big/tmichel/GNN/DATASETS/DATASETS_COMBINED_INIT_ITER/dataset_onecyl_Re1e{re}_gmsh_trunc/predict'
fi
SOURCE_V_FIELD="${SOURCE_V_FIELD:-Vitesse}"
SOURCE_P_FIELD="${SOURCE_P_FIELD:-Pression}"
INIT_V_FIELD="${INIT_V_FIELD:-v_pred}"
INIT_P_FIELD="${INIT_P_FIELD:-p}"
VERBOSE="${VERBOSE:-1}"
STRICT_MISSING="${STRICT_MISSING:-0}"
STRICT_INCOMPATIBLE="${STRICT_INCOMPATIBLE:-0}"

read -r -a RES_ARR <<< "${RES}"

ARGS=(
  --res "${RES_ARR[@]}"
  --source-template "${SOURCE_TEMPLATE}"
  --init-template "${INIT_TEMPLATE}"
  --output-template "${OUTPUT_TEMPLATE}"
  --source-v-field "${SOURCE_V_FIELD}"
  --source-p-field "${SOURCE_P_FIELD}"
  --init-v-field "${INIT_V_FIELD}"
  --init-p-field "${INIT_P_FIELD}"
)

if [[ "${VERBOSE}" == "1" ]]; then
  ARGS+=(--verbose)
fi
if [[ "${STRICT_MISSING}" == "1" ]]; then
  ARGS+=(--strict-missing)
fi
if [[ "${STRICT_INCOMPATIBLE}" == "1" ]]; then
  ARGS+=(--strict-incompatible)
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/create_combined_init_iter_dataset.py" "${ARGS[@]}" "$@"
