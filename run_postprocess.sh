#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate graph || true
fi

PLOTS_ONLY=false
COMPARE=false
CONFIG=""
OUT_OVERRIDE=""
ONLY_PLOTS="all"
SENSOR_DROP_LAST_COLUMN=false
MODELS=()
NICKNAMES=()

print_help() {
  cat <<'EOF'
Usage:
  bash run_postprocess.sh config.json
  bash run_postprocess.sh --plots-only config.json [--output_dir DIR] [--only sensors|rmse|forces|all] [--sensor-drop-last-column]
  bash run_postprocess.sh --compare config.json --models DIR1 DIR2 [...] [--nicknames N1 N2 ...] --output_dir DIR [--only sensors|rmse|forces|all] [--sensor-drop-last-column]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    --plots-only)
      PLOTS_ONLY=true
      shift
      ;;
    --compare)
      COMPARE=true
      shift
      ;;
    --models)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --nicknames)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        NICKNAMES+=("$1")
        shift
      done
      ;;
    --output_dir)
      OUT_OVERRIDE="$2"
      shift 2
      ;;
    --only)
      ONLY_PLOTS="$2"
      shift 2
      ;;
    --sensor-drop-last-column)
      SENSOR_DROP_LAST_COLUMN=true
      shift
      ;;
    *)
      if [[ -z "$CONFIG" ]]; then
        CONFIG="$1"
        shift
      else
        echo "Unknown argument: $1"
        exit 1
      fi
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  print_help
  exit 1
fi

if [[ "$ONLY_PLOTS" != "all" && "$ONLY_PLOTS" != "sensors" && "$ONLY_PLOTS" != "rmse" && "$ONLY_PLOTS" != "forces" ]]; then
  echo "--only must be one of: sensors, rmse, forces, all"
  exit 1
fi

if [[ "$COMPARE" == true ]]; then
  if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "--compare requires --models"
    exit 1
  fi
  if [[ -z "$OUT_OVERRIDE" ]]; then
    echo "--compare requires --output_dir"
    exit 1
  fi
  CMD=(python -m postprocess.compare_models "$CONFIG" --models "${MODELS[@]}" --output_dir "$OUT_OVERRIDE")
  if [[ ${#NICKNAMES[@]} -gt 0 ]]; then
    CMD+=(--nicknames "${NICKNAMES[@]}")
  fi
  if [[ "$ONLY_PLOTS" != "all" ]]; then
    CMD+=(--only "$ONLY_PLOTS")
  fi
  if [[ "$SENSOR_DROP_LAST_COLUMN" == true ]]; then
    CMD+=(--sensor-drop-last-column)
  fi
  "${CMD[@]}"
  exit 0
fi

if [[ "$PLOTS_ONLY" == true ]]; then
  CMD=(python -m postprocess.plot_results "$CONFIG")
  if [[ -n "$OUT_OVERRIDE" ]]; then
    CMD+=(--output_dir "$OUT_OVERRIDE")
  fi
  if [[ "$ONLY_PLOTS" != "all" ]]; then
    CMD+=(--only "$ONLY_PLOTS")
  fi
  if [[ "$SENSOR_DROP_LAST_COLUMN" == true ]]; then
    CMD+=(--sensor-drop-last-column)
  fi
  "${CMD[@]}"
  exit 0
fi

python -m postprocess.postprocess_xdmf "$CONFIG"
python -m postprocess.extract_sensors "$CONFIG"
python -m postprocess.compute_errors "$CONFIG"
python -m postprocess.compute_forces "$CONFIG"
if [[ -n "$OUT_OVERRIDE" ]]; then
  python -m postprocess.plot_results "$CONFIG" --output_dir "$OUT_OVERRIDE"
else
  python -m postprocess.plot_results "$CONFIG"
fi
