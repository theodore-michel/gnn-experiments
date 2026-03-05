# postprocess/visualization

Plotting, ParaView/XDMF formatting, and video generation from GNN predictions.

## Intended scope

- Matplotlib/seaborn plots: loss curves, metric evolution, comparison bar charts.
- ParaView-compatible XDMF files with error overlay fields.
- Animated `.mp4` / `.gif` of flow field evolution (predicted vs. ground truth).
- Snapshot images at key timesteps for paper figures.
- Integration with `pyvista` for programmatic 3D rendering.

## Conventions

- Plotting scripts read data from `postprocess/metrics/` outputs or raw predictions.
- Figures are saved to a `figures/` directory with descriptive filenames.
- Use consistent color maps and style across all visualizations.
