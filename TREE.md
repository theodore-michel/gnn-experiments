# gnn-experiments — Postprocessing Tree

`gnn-experiments/`
- `README.md` — Postprocessing workflow, assumptions, and usage.
- `TREE.md` — Repository tree with one-line file descriptions.
- `environment.yml` — Conda environment specification (`graph`).
- `run_postprocess.sh` — Top-level launcher for full pipeline, plots-only, and compare modes.
- `postprocess/__init__.py` — Package marker.
- `postprocess/config.example.json` — Annotated unified config template.
- `postprocess/postprocess_xdmf.py` — Script 1: raw prediction XDMF to standardized XDMF.
- `postprocess/extract_sensors.py` — Script 2: sensor extraction to flat CSV.
- `postprocess/compute_errors.py` — Script 3: RMSE and cumulative metrics to CSV.
- `postprocess/compute_forces.py` — Script 4: drag/lift force extraction to CSV with caching.
- `postprocess/plot_results.py` — Script 5: single-model publication figures from CSV only.
- `postprocess/compare_models.py` — Script 6: multi-model comparison figures from CSV only.
- `postprocess/utils/__init__.py` — Utilities package marker.
- `postprocess/utils/xdmf_io.py` — Consolidated XDMF I/O, case discovery, geometry/sensor helpers.
- `postprocess/metrics/` — Legacy modules kept for backward compatibility.
- `postprocess/visualization/` — Legacy plotting modules kept for backward compatibility.
