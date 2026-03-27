#!/usr/bin/env bash
set -euo pipefail

# Recompute postprocess artifacts for all 1step models.
# For 1-step inference we only regenerate postprocessed XDMFs and RMSE tables.
# Raw prediction XDMF paths are read from config files and are not modified here.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate gnnpostprocess || conda activate graph || true
fi

CONFIGS=(
  "configs/config.re2_1step_VPLN.json"
  "configs/config.re2_1step_VPN.json"
  "configs/config.re3_1step_VPLN.json"
  "configs/config.re3_1step_VPN.json"
  "configs/config.re4_1step_VPLN.json"
  "configs/config.re4_1step_VPN.json"
)

echo "==> Recomputing postprocessed XDMFs and RMSE tables for ${#CONFIGS[@]} 1step configs"

for cfg in "${CONFIGS[@]}"; do
  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] Missing config: $cfg" >&2
    exit 1
  fi

  read -r model_name output_dir <<EOF
$(python - <<PY
import json
with open("$cfg", "r", encoding="utf-8") as f:
    c = json.load(f)
print(c["model_name"], c["output_dir"])
PY
)
EOF

  model_root="$output_dir/$model_name"

  echo ""
  echo "================================================================"
  echo "Config: $cfg"
  echo "Model:  $model_name"
  echo "Root:   $model_root"
  echo "================================================================"

  # Clean old artifacts so everything is regenerated from scratch
  rm -rf \
    "$model_root/postprocessed" \
    "$model_root/errors" \
    "$model_root/figures" \
    "$model_root/figures_cropped_sensors" \
    "$model_root/sensors" \
    "$model_root/forces"

  # Recompute only the artifacts relevant to 1-step rollouts.
  python -m postprocess.postprocess_xdmf "$cfg"
  python -m postprocess.compute_errors "$cfg"

done

echo ""
echo "==> Done. All 1step postprocessed XDMFs and RMSE tables rebuilt from scratch."
