#!/usr/bin/env bash
set -euo pipefail

# Recompute postprocess artifacts for the all-noise init-iter configs and
# compare them against the matching trunc all-noise VPLN-ld results.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate gnnpostprocess || conda activate graph || true
fi

RE_LIST="${RE_LIST:-2 3 4}"

read -r -a RE_ARR <<< "${RE_LIST}"

echo "==> Recomputing postprocess artifacts for init-iter all-noise configs: Re${RE_ARR[*]}"

for re in "${RE_ARR[@]}"; do
  cfg="configs/config.re${re}_allnoise_vpln_ld_init-iter.json"
  trunc_cfg="configs/config.re${re}trunc_allnoise_vpln_ld.json"

  if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] Missing config: $cfg" >&2
    exit 1
  fi
  if [[ ! -f "$trunc_cfg" ]]; then
    echo "[ERROR] Missing trunc comparison config: $trunc_cfg" >&2
    exit 1
  fi

  read -r init_model_name init_output_dir <<EOF
$(python - <<PY
import json
with open("$cfg", "r", encoding="utf-8") as f:
    c = json.load(f)
print(c["model_name"], c["output_dir"])
PY
)
EOF

  read -r trunc_model_name trunc_output_dir <<EOF
$(python - <<PY
import json
with open("$trunc_cfg", "r", encoding="utf-8") as f:
    c = json.load(f)
print(c["model_name"], c["output_dir"])
PY
)
EOF

  init_model_root="$init_output_dir/$init_model_name"
  trunc_model_root="$trunc_output_dir/$trunc_model_name"

  echo ""
  echo "================================================================"
  echo "Config: $cfg"
  echo "Model:  $init_model_name"
  echo "Root:   $init_model_root"
  echo "Trunc:  $trunc_model_root"
  echo "================================================================"

  rm -rf \
    "$init_model_root/postprocessed" \
    "$init_model_root/sensors" \
    "$init_model_root/errors" \
    "$init_model_root/forces" \
    "$init_model_root/figures" \
    "$init_model_root/figures_cropped_sensors"

  python -m postprocess.postprocess_xdmf "$cfg"
  python -m postprocess.extract_sensors "$cfg"
  python -m postprocess.compute_errors "$cfg"
  python -m postprocess.compute_forces "$cfg"
  python -m postprocess.plot_results "$cfg"

  compare_root="results_ReX_allnoise_full/comparison_Re${re}_trunc_vs_init_iter"
  rm -rf "$compare_root" "$compare_root/cropped_sensors"

  python -m postprocess.compare_models \
    "$cfg" \
    --models \
      "$trunc_model_root" \
      "$init_model_root" \
    --nicknames \
      "Re${re} VPLN-ld truth-iter" \
      "Re${re} VPLN-ld init-iter" \
    --output_dir "$compare_root"

  python -m postprocess.compare_models \
    "$cfg" \
    --models \
      "$trunc_model_root" \
      "$init_model_root" \
    --nicknames \
      "Re${re} VPLN-ld truth-iter" \
      "Re${re} VPLN-ld init-iter" \
    --output_dir "$compare_root/cropped_sensors" \
    --only sensors \
    --sensor-drop-last-column

done

echo ""
echo "==> Done. Init-iter postprocess artifacts and trunc-vs-init-iter comparisons rebuilt from scratch."