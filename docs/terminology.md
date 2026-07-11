# Terminology

This document defines the key terminology used in the nbody-pipeline codebase to avoid confusion.

## Snapshot vs HDF5 File

### Snapshot (snap)

A **snapshot** refers to simulation data at **ONE specific moment in time**. It represents the state of all particles, binaries, and scalar quantities at a single time point `TTOT`.

**Key characteristics:**
- Contains data at a single `TTOT` value
- Can be extracted from an HDF5 file using `get_snapshot_at_t(df_dict, ttot)`
- Examples in code:
  - `single_df_at_t`: Single star data at time `t`
  - `binary_df_at_t`: Binary star data at time `t`
  - `get_snapshot_at_t()`: Function to extract data at specific time

### HDF5 File (hdf5/hdf)

An **HDF5 file** (`.h5part`) is a container that stores **MULTIPLE snapshots** from the simulation. Each file typically contains data for multiple time points (default: 8 snapshots per file).

**Key characteristics:**
- Contains multiple `Step#` groups, each representing a different snapshot
- Each `Step#` corresponds to a different `TTOT` value
- File naming: `snap.40_<TIME>.h5part` where `<TIME>` indicates the approximate time range
- Examples in code:
  - `hdf5_file_path`: Path to an HDF5 file
  - `df_dict`: Dictionary returned by `read_file()` containing data from ALL snapshots in the file
  - `dataframes_from_hdf5_file()`: Function to read entire HDF5 file

## Usage Guidelines

### When to use "snapshot" terminology:
- Referring to data at a specific time point
- Extracting or filtering data by `TTOT`
- Working with `single_df_at_t`, `binary_df_at_t`, etc.

### When to use "HDF5 file" terminology:
- Referring to the physical `.h5part` files
- Reading or processing entire files
- Iterating through files in a directory
- File I/O operations

## Code Examples

```python
# Reading an HDF5 file (contains multiple snapshots)
hdf5_file_path = "/path/to/snap.40_1.234.h5part"
df_dict = processor.read_file(hdf5_file_path, simu_name)

# Extracting a specific snapshot at time t
ttot = 1.5
single_df_at_t, binary_df_at_t, is_valid = processor.get_snapshot_at_t(df_dict, ttot)

# Processing all snapshots within the HDF5 file
for ttot in df_dict['scalars']['TTOT'].unique():
    single_df, binary_df, _ = processor.get_snapshot_at_t(df_dict, ttot)
    # ... process this snapshot ...
```

## Common Patterns

1. **File iteration**: Loop over HDF5 files in a directory
2. **Snapshot iteration**: Loop over time points (`TTOT`) within each file
3. **Data extraction**: Get data for a specific snapshot from the file's data

```python
# Pattern: Process all snapshots from all HDF5 files
for hdf5_file in hdf5_files:                      # Iterate over HDF5 files
    df_dict = processor.read_file(hdf5_file)      # Read entire file
    for ttot in df_dict['scalars']['TTOT']:       # Iterate over snapshots
        snapshot_data = get_snapshot_at_t(df_dict, ttot)  # Extract one snapshot
        process_snapshot(snapshot_data)            # Process this moment in time
```

## Historical Note

This terminology was standardized to resolve confusion where "snap" was sometimes used to refer to HDF5 files (because they are named snap.XXX.h5part in NBODY6++GPU output).
