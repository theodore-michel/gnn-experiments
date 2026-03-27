#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate gnnpostprocess || conda activate graph || true
fi

CONFIGS=(
  "configs/config.re2trunc_allnoise_vpln_l.json"
  "configs/config.re2trunc_allnoise_vpln_ld.json"
  "configs/config.re2trunc_allnoise_vpn_l.json"
  "configs/config.re3trunc_allnoise_vpln_l.json"
  "configs/config.re3trunc_allnoise_vpln_ld.json"
  "configs/config.re3trunc_allnoise_vpn_l.json"
  "configs/config.re4trunc_allnoise_vpln_l.json"
  "configs/config.re4trunc_allnoise_vpln_ld.json"
  "configs/config.re4trunc_allnoise_vpn_l.json"
)

echo "==> Rebuilding single-model plots from existing CSV data"
for cfg in "${CONFIGS[@]}"; do
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
  figures_dir="$model_root/figures"
  cropped_dir="$model_root/figures_cropped_sensors"

  echo ""
  echo "================================================================"
  echo "Config: $cfg"
  echo "Model:  $model_name"
  echo "================================================================"

  rm -rf "$figures_dir" "$cropped_dir"

  # Full figure suite from existing sensors/errors/forces outputs.
  python -m postprocess.plot_results "$cfg"

  # Cropped sensor-only figures.
  python -m postprocess.plot_results \
    "$cfg" \
    --only sensors \
    --sensor-drop-last-column \
    --output_dir "$cropped_dir"
done

echo ""
echo "==> Rebuilding comparison plots from existing CSV data"
rm -rf \
  "results_ReX_allnoise/comparison_Re2" \
  "results_ReX_allnoise/comparison_Re3" \
  "results_ReX_allnoise/comparison_Re4"

# Re2
python -m postprocess.compare_models \
  configs/config.re2trunc_allnoise_vpln_l.json \
  --models \
    results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPN_l \
    results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l \
    results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_ld \
  --nicknames "Re2 VPN-l" "Re2 VPLN-l" "Re2 VPLN-ld" \
  --output_dir results_ReX_allnoise/comparison_Re2

python -m postprocess.compare_models \
  configs/config.re2trunc_allnoise_vpln_l.json \
  --models \
    results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPN_l \
    results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l \
    results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_ld \
  --nicknames "Re2 VPN-l" "Re2 VPLN-l" "Re2 VPLN-ld" \
  --output_dir results_ReX_allnoise/comparison_Re2/comparison_Re2_cropped_sensors \
  --only sensors \
  --sensor-drop-last-column

# Re3
python -m postprocess.compare_models \
  configs/config.re3trunc_allnoise_vpln_l.json \
  --models \
    results_ReX_allnoise/onecyl_Re3trunc_allnoise_VPN_l \
    results_ReX_allnoise/onecyl_Re3trunc_allnoise_VPLN_l \
    results_ReX_allnoise/onecyl_Re3trunc_allnoise_VPLN_ld \
  --nicknames "Re3 VPN-l" "Re3 VPLN-l" "Re3 VPLN-ld" \
  --output_dir results_ReX_allnoise/comparison_Re3

python -m postprocess.compare_models \
  configs/config.re3trunc_allnoise_vpln_l.json \
  --models \
    results_ReX_allnoise/onecyl_Re3trunc_allnoise_VPN_l \
    results_ReX_allnoise/onecyl_Re3trunc_allnoise_VPLN_l \
    results_ReX_allnoise/onecyl_Re3trunc_allnoise_VPLN_ld \
  --nicknames "Re3 VPN-l" "Re3 VPLN-l" "Re3 VPLN-ld" \
  --output_dir results_ReX_allnoise/comparison_Re3/comparison_Re3_cropped_sensors \
  --only sensors \
  --sensor-drop-last-column

# Re4
python -m postprocess.compare_models \
  configs/config.re4trunc_allnoise_vpln_l.json \
  --models \
    results_ReX_allnoise/onecyl_Re4trunc_allnoise_VPN_l \
    results_ReX_allnoise/onecyl_Re4trunc_allnoise_VPLN_l \
    results_ReX_allnoise/onecyl_Re4trunc_allnoise_VPLN_ld \
  --nicknames "Re4 VPN-l" "Re4 VPLN-l" "Re4 VPLN-ld" \
  --output_dir results_ReX_allnoise/comparison_Re4

python -m postprocess.compare_models \
  configs/config.re4trunc_allnoise_vpln_l.json \
  --models \
    results_ReX_allnoise/onecyl_Re4trunc_allnoise_VPN_l \
    results_ReX_allnoise/onecyl_Re4trunc_allnoise_VPLN_l \
    results_ReX_allnoise/onecyl_Re4trunc_allnoise_VPLN_ld \
  --nicknames "Re4 VPN-l" "Re4 VPLN-l" "Re4 VPLN-ld" \
  --output_dir results_ReX_allnoise/comparison_Re4/comparison_Re4_cropped_sensors \
  --only sensors \
  --sensor-drop-last-column

echo ""
echo "==> Done. All single and comparison plots refreshed from existing data."
