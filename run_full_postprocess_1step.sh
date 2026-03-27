#!/usr/bin/env bash
set -euo pipefail

# Recompute postprocess artifacts for all 1step models
# (postprocessed XDMFs, sensors, errors, forces, plots) from scratch.
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

echo "==> Recomputing postprocessed XDMFs, errors/forces/plots for ${#CONFIGS[@]} 1step configs"

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
    "$model_root/sensors" \
    "$model_root/errors" \
    "$model_root/forces" \
    "$model_root/figures" \
    "$model_root/figures_cropped_sensors"

  # Recompute all postprocess artifacts from scratch
  python -m postprocess.postprocess_xdmf "$cfg"
  python -m postprocess.extract_sensors "$cfg"
  python -m postprocess.compute_errors "$cfg"
  python -m postprocess.compute_forces "$cfg"
  python -m postprocess.plot_results "$cfg"

done

echo ""
echo "==> Rebuilding 1step comparison plots (VPLN vs VPN for Re2/Re3/Re4)"

rm -rf \
  "results_ReX_1step/comparison_Re2" \
  "results_ReX_1step/comparison_Re3" \
  "results_ReX_1step/comparison_Re4"

# Re2 comparisons
python -m postprocess.compare_models \
  configs/config.re2_1step_VPLN.json \
  --models \
    results_ReX_1step/onecyl_Re2_1step_VPLN \
    results_ReX_1step/onecyl_Re2_1step_VPN \
  --nicknames "Re2 VPLN" "Re2 VPN" \
  --output_dir results_ReX_1step/comparison_Re2

python -m postprocess.compare_models \
  configs/config.re2_1step_VPLN.json \
  --models \
    results_ReX_1step/onecyl_Re2_1step_VPLN \
    results_ReX_1step/onecyl_Re2_1step_VPN \
  --nicknames "Re2 VPLN" "Re2 VPN" \
  --output_dir results_ReX_1step/comparison_Re2/comparison_Re2_cropped_sensors \
  --only sensors \
  --sensor-drop-last-column

# Re3 comparisons
python -m postprocess.compare_models \
  configs/config.re3_1step_VPLN.json \
  --models \
    results_ReX_1step/onecyl_Re3_1step_VPLN \
    results_ReX_1step/onecyl_Re3_1step_VPN \
  --nicknames "Re3 VPLN" "Re3 VPN" \
  --output_dir results_ReX_1step/comparison_Re3

python -m postprocess.compare_models \
  configs/config.re3_1step_VPLN.json \
  --models \
    results_ReX_1step/onecyl_Re3_1step_VPLN \
    results_ReX_1step/onecyl_Re3_1step_VPN \
  --nicknames "Re3 VPLN" "Re3 VPN" \
  --output_dir results_ReX_1step/comparison_Re3/comparison_Re3_cropped_sensors \
  --only sensors \
  --sensor-drop-last-column

# Re4 comparisons
python -m postprocess.compare_models \
  configs/config.re4_1step_VPLN.json \
  --models \
    results_ReX_1step/onecyl_Re4_1step_VPLN \
    results_ReX_1step/onecyl_Re4_1step_VPN \
  --nicknames "Re4 VPLN" "Re4 VPN" \
  --output_dir results_ReX_1step/comparison_Re4

python -m postprocess.compare_models \
  configs/config.re4_1step_VPLN.json \
  --models \
    results_ReX_1step/onecyl_Re4_1step_VPLN \
    results_ReX_1step/onecyl_Re4_1step_VPN \
  --nicknames "Re4 VPLN" "Re4 VPN" \
  --output_dir results_ReX_1step/comparison_Re4/comparison_Re4_cropped_sensors \
  --only sensors \
  --sensor-drop-last-column

echo ""
echo "==> Done. All 1step postprocessed XDMFs, errors, forces, and comparisons rebuilt from scratch."
