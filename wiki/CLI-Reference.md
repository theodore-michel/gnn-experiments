# CLI Reference

This page lists every Python file in the repo that exposes an `argparse` interface.

## Active Pipeline

### `python -m postprocess.postprocess_xdmf`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `config` | positional string | n/a | yes | Unified JSON config file |

Example:

```bash
python -m postprocess.postprocess_xdmf configs/config.re2trunc_allnoise_vpln_l.json
```

### `python -m postprocess.extract_sensors`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `config` | positional string | n/a | yes | Unified JSON config file |

Example:

```bash
python -m postprocess.extract_sensors configs/config.re2trunc_allnoise_vpln_l.json
```

### `python -m postprocess.compute_errors`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `config` | positional string | n/a | yes | Unified JSON config file |

Example:

```bash
python -m postprocess.compute_errors configs/config.re2trunc_allnoise_vpln_l.json
```

### `python -m postprocess.compute_forces`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `config` | positional string | n/a | yes | Unified JSON config file |

Example:

```bash
python -m postprocess.compute_forces configs/config.re2trunc_allnoise_vpln_l.json
```

### `python -m postprocess.plot_results`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `config` | positional string | n/a | yes | Unified JSON config file |
| `--output_dir` | string | `None` | no | Custom figure output directory |
| `--only` | choice | `all` | no | Restrict to `all`, `sensors`, `rmse`, or `forces` |
| `--sensor-drop-last-column` | flag | `false` | no | Force 3x2 sensor layout |

Example:

```bash
python -m postprocess.plot_results configs/config.re2trunc_allnoise_vpln_l.json --only sensors
```

### `python -m postprocess.compare_models`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `config` | positional string | n/a | yes | Unified JSON config file; used for `configs_pool` and plot settings |
| `--models` | list of strings | n/a | yes | Model output directories (`output_dir/model_name`) |
| `--nicknames` | list of strings | `None` | no | Optional display labels, one per model |
| `--output_dir` | string | n/a | yes | Comparison output directory |
| `--only` | choice | `all` | no | Restrict to `all`, `sensors`, `rmse`, or `forces` |
| `--sensor-drop-last-column` | flag | `false` | no | Force 3x2 sensor layout |

Example:

```bash
python -m postprocess.compare_models configs/config.re2trunc_allnoise_vpln_l.json \
  --models results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPN_l \
           results_ReX_allnoise/onecyl_Re2trunc_allnoise_VPLN_l \
  --nicknames "VPN-l" "VPLN-l" \
  --output_dir ./comparison
```

## Dataset Helpers

### `python scripts/create_combined_init_iter_dataset.py`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `--res` | list of int | `2 3 4` | no | Reynolds regimes to process |
| `--source-template` | string | scratch dataset template | no | Template for source full-trajectory prediction directories |
| `--init-template` | string | init-iter results template | no | Template for 1-step initializer XDMFs |
| `--output-template` | string | combined dataset template | no | Template for output prediction directories |
| `--source-v-field` | string | `Vitesse` | no | Velocity field name in source XDMFs |
| `--source-p-field` | string | `Pression` | no | Pressure field name in source XDMFs |
| `--init-v-field` | string | `v_pred` | no | Velocity field name in initializer XDMFs |
| `--init-p-field` | string | `p` | no | Pressure field name in initializer XDMFs |
| `--strict-missing` | flag | `false` | no | Fail if any source case has no initializer match |
| `--strict-incompatible` | flag | `false` | no | Fail if any mesh substitution is incompatible |
| `--verbose` | flag | `false` | no | Enable per-case logging |

### `python scripts/create_predict_edit_dataset.py`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `--source-dir` | Path | n/a | yes | Directory of source XDMF files |
| `--output-dir` | Path | n/a | yes | Destination directory |
| `--velocity-field` | string | `Vitesse` | no | Velocity field to overwrite in step 0 |
| `--pressure-field` | string | `Pression` | no | Pressure field to overwrite in step 0 |
| `--verbose` | flag | `false` | no | Print per-case status |

## Legacy Compatibility CLIs

### `python -m postprocess.metrics.process_xdmf`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `-p`, `--parameters` | string | n/a | yes | JSON config file |
| `-d`, `--directory` | string | n/a | yes | Output root directory |

### `python -m postprocess.metrics.compute_errors`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `-p`, `--parameters` | string | n/a | yes | JSON config file |
| `-d`, `--directory` | string | `./error_results` | no | Output root directory |
| `--sensor-errors` | flag | `false` | no | Also compute sensor-level cumulative errors |
| `--metric` | choice | `AE` | no | Sensor error metric: `AE` or `SE` |
| `--load-data` | flag | `false` | no | Reuse cached sensor JSON if present |

### `python -m postprocess.metrics.compute_forces`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `-p`, `--parameters` | string | n/a | yes | JSON config file |
| `-d`, `--directory` | string | `./force_results` | no | Output root directory |
| `--workers` | int | `4` | no | Thread pool size |
| `--truth-folder` | string | `None` | no | Optional truth XDMF folder |
| `--truth-base-name` | string | `None` | no | Optional truth file base name |

### `python -m postprocess.visualization.plot_results`

| Argument | Type | Default | Required | Description |
|---|---|---:|---|---|
| `-d`, `--directory` | string | n/a | yes | Root results directory |
| `-o`, `--output` | string | `None` | no | Plot output directory |
| `--article-style` | flag | `false` | no | Remove decorative titles and annotations |
| `--forces-dir` | string | `None` | no | Alternate forces directory |
| `--truth-dir` | string | `None` | no | Truth force directory |
| `--format` | list of strings | `png` | no | Output formats |
| `--compact-sensors` | flag | `false` | no | Use compact 3x2 sensor layout |

Example of the legacy plotter:

```bash
python -m postprocess.visualization.plot_results -d ./results_ReX_allnoise --article-style
```

---
Prev: [Plotting](Plotting.md) | Next: [Batch Scripts](Batch-Scripts.md)
