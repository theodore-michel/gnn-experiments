# postprocess/utils

Shared utilities for post-processing: data loading, prediction I/O, XDMF parsing.

## Intended scope

- XDMF/HDF5 readers: load prediction and ground-truth fields into numpy arrays.
- Mesh utilities: node coordinates, connectivity, boundary masks.
- I/O helpers: find prediction directories, match pred↔GT trajectories.
- Common constants: field names, node type enums, physical parameters.

## Conventions

- Functions should be stateless and testable in isolation.
- Use `meshio` and `h5py` for I/O, consistent with the `graphphysics` codebase.
