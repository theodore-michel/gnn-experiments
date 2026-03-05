# gnn-experiments — Directory Tree
#
# Each line: relative path — one-line description

gnn-experiments/
├── run_manager.py              # CLI tool (click): register, update, relaunch, compare, sync-sheet
├── launch.sh                   # Shell wrapper: register + sbatch + capture SLURM job ID
├── environment.yml             # Conda env spec (name: graph) with all dependencies
├── schema.yaml                 # Annotated YAML template documenting every run-record field
├── README.md                   # Setup guide, workflow docs, field reference
├── TREE.md                     # This file — repository directory tree with descriptions
├── registry/                   # Per-run YAML records (one file per registered run, git-tracked)
│   └── <run_id>.yaml           # Example: re100_full_20260305_001.yaml
└── postprocess/                # Post-processing scaffold (structure only, implementation TBD)
    ├── __init__.py             # Package marker
    ├── metrics/                # Error statistics & custom metric computation
    │   ├── __init__.py         # Package marker
    │   └── README.md           # Scope: RMSE, MAE, drag/lift, spatial error maps
    ├── visualization/          # Plotting, ParaView XDMF export, video generation
    │   ├── __init__.py         # Package marker
    │   └── README.md           # Scope: matplotlib plots, pyvista renders, mp4/gif
    └── utils/                  # Shared I/O utilities: XDMF readers, mesh loading
        ├── __init__.py         # Package marker
        └── README.md           # Scope: meshio/h5py readers, trajectory matching
