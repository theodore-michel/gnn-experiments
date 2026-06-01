# Home

This repository postprocesses Graph Neural Network predictions for 2D cylinder-flow rollouts. The active workflow takes raw prediction XDMF/HDF5 pairs, standardizes the fields, extracts sensor signals, computes RMSE and force summaries, and renders publication-ready plots.

```text
raw prediction XDMF/H5
        |
        v
postprocess_xdmf.py
        |
        +--> xdmf/<case>.xdmf + .h5
        |
        +--> extract_sensors.py --> sensors/sensor_data.csv
        |
        +--> compute_errors.py   --> errors/*.csv
        |
        +--> compute_forces.py   --> forces/*.csv
        |
        +--> plot_results.py      --> figures/*.png
        |
        +--> compare_models.py    --> comparison figures
```

The code path above is the one used by the root-level shell scripts. The older `postprocess/metrics/` and `postprocess/visualization/` packages are present for compatibility, but the current launchers do not call them.

## What To Read First

1. [Getting Started](Getting-Started.md) for setup and a minimal end-to-end run.
2. [Repository Structure](Repository-Structure.md) for the annotated file tree.
3. [Configuration](Configuration.md) for every config key used by the active pipeline.
4. [Workflow Guide](Workflow-Guide.md) for a full worked example.

## Main Entry Points

| Task | Command |
|---|---|
| Full postprocess for one model | `bash run_postprocess.sh postprocess/config.my_model.json` |
| Replot existing outputs only | `bash run_postprocess.sh --plots-only postprocess/config.my_model.json` |
| Compare multiple models | `bash run_postprocess.sh --compare postprocess/config.base.json --models DIR1 DIR2 --output_dir ./comparison` |
| Batch local run | `bash batch_postprocess.sh config.A.json config.B.json` |
| Batch SLURM run | `bash run_full_postprocess_slurm.sh` |

## Wiki Map

| Page | Purpose |
|---|---|
| [Getting Started](Getting-Started.md) | Installation and first run |
| [Repository Structure](Repository-Structure.md) | File-by-file layout |
| [Configuration](Configuration.md) | JSON parameter reference |
| [Error Metrics](Error-Metrics.md) | RMSE and force metric definitions |
| [XDMF-HDF5 Postprocessing](XDMF-HDF5-Postprocessing.md) | Input/output structure for XDMF and HDF5 |
| [Plotting](Plotting.md) | Figure types and controls |
| [CLI Reference](CLI-Reference.md) | Command-line interfaces |
| [Batch Scripts](Batch-Scripts.md) | Shell orchestration scripts |
| [Model Comparison](Model-Comparison.md) | Multi-model comparison workflow |
| [Workflow Guide](Workflow-Guide.md) | Full worked pipeline |
| [Extending the Codebase](Extending-the-Codebase.md) | How to add new stages or outputs |

---
Next: [Getting Started](Getting-Started.md)
