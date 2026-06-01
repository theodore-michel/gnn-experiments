# Getting Started

This repository expects a working Python environment plus access to prediction XDMF/HDF5 files, a matching truth dataset, and a `configs_pool` pickle for sensor placement and case ordering.

## Prerequisites

| Requirement | Why it is needed |
|---|---|
| Python 3.11 | Matches `environment.yml` |
| `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `meshio`, `h5py` | Required by the active pipeline |
| `conda` or `pip` | Environment setup |
| XDMF/HDF5 outputs | Raw predictions to postprocess |
| Truth dataset XDMFs | Levelset injection when `levelset_source` is `dataset` |
| `configs_pool` pickle | Auto sensor placement and Reynolds-based sorting |

## Install

Recommended setup:

```bash
cd gnn-experiments
conda env create -f environment.yml
conda activate gnnpostprocess
```

If you already have a suitable Python environment:

```bash
pip install -r requirements.txt
```

Sanity check:

```bash
python -m postprocess.postprocess_xdmf --help
```

## Minimal End-To-End Run

Use one of the existing config files as a template. The active pipeline only needs a single JSON file:

```bash
bash run_postprocess.sh configs/config.re2trunc_allnoise_vpln_l.json
```

This will create:

| Output | Location |
|---|---|
| Standardized XDMFs | `results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l/xdmf/` |
| Sensor CSV | `results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l/sensors/sensor_data.csv` |
| RMSE tables | `results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l/errors/` |
| Force CSVs | `results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l/forces/` |
| Figures | `results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l/figures/` |

## New Model Checklist

To process a new prediction set, copy an existing config and change these fields exactly:

| Key | What to point it at |
|---|---|
| `pred_dir` | Folder containing the raw prediction `.xdmf` files |
| `dataset_dir` | Truth dataset folder used for levelset injection if needed |
| `configs_pool` | Pickle with case metadata and geometry |
| `model_name` | Output folder name under `output_dir` |
| `model_shortname` | Plot legend label |
| `output_dir` | Root results directory |
| `prediction_base_name` | File prefix such as `graph_` |
| `feature_map` | Mapping from `x*` fields to semantics |
| `target_map` | Mapping from `y*` fields to semantics |

If any of those paths are wrong, the pipeline typically fails early with `FileNotFoundError`, `KeyError`, or empty case discovery.

## Fast Reruns

If the CSV outputs already exist and you only want figures:

```bash
bash run_postprocess.sh --plots-only configs/config.re2trunc_allnoise_vpln_l.json
```

If you want to compare multiple already-processed models:

```bash
bash run_postprocess.sh --compare configs/config.re2trunc_allnoise_vpln_l.json \
  --models results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPN_l \
           results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l \
  --output_dir ./results_ReX_allnoise/comparison_example
```

---
Prev: [Home](Home.md) | Next: [Repository Structure](Repository-Structure.md)
