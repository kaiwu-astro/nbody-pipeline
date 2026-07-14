# nbody-pipeline API Reference

## Configuration

### `nbody_pipeline.config.ConfigManager`

Main configuration class for managing simulation settings. The packaged default
ships no site-specific paths (`paths.simulations`/`plot_dir`/`analysis_cache_dir`),
so `ConfigManager()`/examples below assume a user config (see
[`config.example.yaml`](../config.example.yaml)) has been supplied via `config_path=`,
`--config`, `NBODY_CONFIG`, or `./nbody_config.yaml` — otherwise construction
raises a `ValueError` naming the missing keys.

```python
from nbody_pipeline.config import ConfigManager, load_config

# Load with custom YAML
config = ConfigManager(config_path="my_config.yaml")

# Equivalent convenience function
config = load_config("my_config.yaml")
```

## I/O Operations

### `nbody_pipeline.io.HDF5FileProcessor`

Read and process HDF5 simulation files.

### `nbody_pipeline.io.LagrFileProcessor`

Process Lagrangian radii files.

### `nbody_pipeline.io.text_parsers`

Functions for parsing text-based simulation output files.

## Analysis

### `nbody_pipeline.analysis.ParticleTracker`

Track individual particles through simulation snapshots.

### `nbody_pipeline.analysis.BinaryStellarTypeExtractor`

Extract complete processed binary rows where either component matches a StellarType abbreviation or KW code.

```python
from nbody_pipeline.analysis import BinaryStellarTypeExtractor
from nbody_pipeline.config import ConfigManager

config = ConfigManager()
extractor = BinaryStellarTypeExtractor(config)
bh_binaries = extractor.load_binaries_with_stellar_type("20sb", stellar_type="BH")
ns_binaries = extractor.load_binaries_with_stellar_type("20sb", kw="13")
```

Specify exactly one of `stellar_type` or `kw`. StellarType abbreviations are matched case-insensitively using `default_config.yaml` `stellar_types`.
Pass `force=True` to rebuild the analysis cache from HDF5 files.

### `nbody_pipeline.analysis.BTypeBinaryExtractor`

Extract complete processed binary rows where either component satisfies the project B-type main-sequence criteria: `Bin KW* == 1`, `10500 <= Bin Teff* <= 31500`, and `2.75 <= Bin M* <= 17.7`.

```python
from nbody_pipeline.analysis import BTypeBinaryExtractor
from nbody_pipeline.config import ConfigManager

config = ConfigManager()
df = BTypeBinaryExtractor(config).load_b_type_binaries("20sb")
```

The returned table preserves the processed binary rows and adds `b_type_member1`, `b_type_member2`, `b_type_member_count`, `b_type_pair_key`, and `is_primordial_binary`. Results are cached under `paths.analysis_cache_dir`.

### `nbody_pipeline.analysis.InitialTotalMassAnalyzer`

Load the initial cluster mass from `lagr.7` using the `100%` shell total mass:
`total_mass = avmass * nshell` at strict `Time[Myr] == 0.0`.

```python
from nbody_pipeline.analysis import InitialTotalMassAnalyzer
from nbody_pipeline.config import ConfigManager

config = ConfigManager()
mass_msun = InitialTotalMassAnalyzer(config).get_initial_total_mass_msun("20sb")
```

The scalar result is cached under `initial_total_mass/initial_total_mass.feather`.
Pass `force=True` to rebuild it from `lagr.7`.

### `nbody_pipeline.analysis.IntermediateMassBlackHoleAnalyzer`

Scan HDF5 snapshots for intermediate-mass black hole candidates, defined as `KW == 14` and `100 < mass < 1e5` solar mass. Single-star candidates use `M`; binary components use `Bin M1*` and `Bin M2*`. Physical merger lineage is loaded from the concatenated `coll.13` and `coal.24` continuous files, not from the HDF5 `mergers` table.

```python
from nbody_pipeline.analysis import IntermediateMassBlackHoleAnalyzer
from nbody_pipeline.config import ConfigManager

config = ConfigManager()
analyzer = IntermediateMassBlackHoleAnalyzer(config)
summary = analyzer.summarize_simulation("20sb")
snapshots = analyzer.load_imbh_snapshots("20sb")
events = analyzer.load_imbh_merger_events("20sb")
```

`summarize_simulation()` returns `summary`, `objects`, `snapshots`, and `merger_events`. Results are cached under `intermediate_mass_black_hole` as `imbh_snapshots.feather` and `imbh_true_merger_events.feather`.

### `nbody_pipeline.analysis.hdf5_scan.HDF5ScanSession`

Batch compatible HDF5 data-reduction tasks so each HDF5 file is read once per simulation/options group.

```python
from nbody_pipeline.analysis import BTypeBinaryExtractor, BinaryStellarTypeExtractor
from nbody_pipeline.analysis.hdf5_scan import HDF5ScanSession
from nbody_pipeline.config import ConfigManager

config = ConfigManager()
session = HDF5ScanSession(config)
session.add_job(BinaryStellarTypeExtractor(config).build_scan_job("20sb", stellar_type="BH"))
session.add_job(BTypeBinaryExtractor(config).build_scan_job("20sb"))
results = session.run()
```

`HDF5ScanOptions` defaults to tail-incremental cache validation. Use `force=True` to ignore old cache/meta and rebuild from scratch.

Developer extension pattern: new analysis/data-reduction features that scan HDF5 files for small per-snapshot records should implement an `HDF5ScanTask`, then expose it through a thin analysis class inheriting `ScanBackedAnalysisBase`. Keep extraction, merge, cache path, metadata, and freshness semantics in the task; use the base only for option merging and single-job run/cache loading.

### `nbody_pipeline.analysis.CompactBinaryCounter`

Count compact binary categories across snapshots. `summarize_simulation()` returns the existing `{"summary": ..., "details": ...}` structure and caches per-snapshot category hits under `compact_binary_count`.

### `nbody_pipeline.analysis.GalacticOrbitProcessor`

Build cluster galactic-orbit points from scalar HDF5 snapshots. `load_plot_data()` returns de-duplicated rows containing `TTOT`, `Time[Myr]`, `RG(1..3)`, `VG(1..3)`, and source tracking columns. Results are cached under `galactic_orbit`.

### `nbody_pipeline.analysis.GalacticEnergyAngularMomentumProcessor`

Compute per-star galactocentric kinetic energy, potential energy, total energy, and `L_z` for one HDF5 snapshot. `compute_snapshot(single_df_at_t, scalar_row_at_t)` returns a copy of the single-star DataFrame with mass-weighted `E_kin_gal[Msun*(km/s)^2]`, `E_pot_gal[Msun*(km/s)^2]`, `E_gal[Msun*(km/s)^2]`, `L_z_gal[Msun*kpc*km/s]`, and specific (per-unit-mass, mass-independent) `E_kin_gal_specific[(km/s)^2]`, `E_pot_gal_specific[(km/s)^2]`, `E_gal_specific[(km/s)^2]`, `L_z_gal_specific[kpc*km/s]`. The calculation uses `galpy` `MWPotential2014`. Scalar `RG(1-3)`/`VG(1-3)` are raw N-body (Henon) units in the HDF5 scalars table (NBODY6++GPU writes them without re-applying `RSCALE_OUT`/`VSCALE_OUT`, unlike the per-star `X1`/`V1` columns), so they are scaled by that snapshot's `RBAR`/`VSTAR` before being added to the (already-physical) star offsets — mirroring how `RDENS(1-3)` is scaled in `hdf5_reader.py`. `compute_cluster_com(scalar_row_at_t, representative_mass_msun=None)` returns the cluster's own bulk position/velocity in the same E/Lz frame, for overlaying as a reference point on these plots.

`nbody_pipeline.visualization.SingleStarVisualizer` exposes three plots built from this frame: `create_galactic_energy_angular_momentum_plot_jpg` (mass-weighted E vs `L_z`), `create_galactic_energy_angular_momentum_specific_plot_jpg` (specific/unweighted E vs `L_z`), and `create_galactic_kinetic_energy_specific_plot_jpg` (specific kinetic energy vs specific `L_z`, isolating the kinetic term for diagnosing potential-vs-kinetic issues). All three accept an optional `com_point` dict (from `compute_cluster_com`) to overlay the cluster COM.

Known remaining gap: the same "RG/VG are raw N-body units" issue also affects `nbody_pipeline.analysis.galactic_orbit.py` (and its visualizer) and the `rg_x_pc`/`vg_x_kmps` fields in `snapshot_summary.py`/`particle_lake.py`, which were not fixed here because they back existing cached feather/parquet outputs. See memory/todo before trusting physical-unit values from those.

### `nbody_pipeline.analysis.tau_gw`

Calculate gravitational wave merger timescales.

### `nbody_pipeline.analysis.kstar_semantics`

Decode raw `cm_kw` (HDF5 Item 160, binary centre-of-mass `KSTAR`) and member `kw_1`/`kw_2` (Item 158/159)
codes into physical states. The manual's odd/even prose for `cm_kw` is backwards relative to what the
NBODY6++GPU source (`roche.f`/`expel.f`) actually does; this module follows the code: **odd values >=11
mean mass transfer is ongoing right now; even values >=10 mean the binary is between episodes** (see the
module docstring for the full derivation). This matches the pre-existing `hdf5_reader.py`/
`snapshot_binaries.yaml` convention, which needed no fix.

```python
from nbody_pipeline.analysis import decode_cm_kstar, decode_member_kw, annotate_binary_states

decode_cm_kstar(11)          # CmKstarState(mt_ongoing=True, mt_phase_index=1, ...)
decode_member_kw(114)        # (14, True)  -- common-envelope member, base KW 14
annotated_df = annotate_binary_states(binary_df)  # adds mt_ongoing/mt_past/is_relativistic_binary/...
```

### `nbody_pipeline.analysis.physics.binding_energy_nb` / `ebind_over_kt`

`binding_energy_nb(m1_msun, m2_msun, a_au, *, zmbar_msun, rbar_pc)` reproduces `hdf5_reader.py`'s
`Ebind_abs_NBODY` formula exactly (reduced mass over twice the semi-major axis, both in N-body units) for
numerical continuity with existing step1 CSVs. `ebind_over_kt(ebind_nb, eclose_nb)` is a plain division;
**pass `config.ECLOSE_INPUT` (the fixed constant, default 1.0), not the real per-snapshot
`snapshot_scalars.eclose_nb`**, to stay numerically consistent with step1's `Ebind/kT`/`is_hard_binary`
columns. `eclose_nb` genuinely varies over a run (e.g. 0.003-1.0 within 20sb) because `adjust.F` re-tunes
it, so this is a known normalization limitation of the existing pipeline, not a step2 design choice --
fixing it properly (using the real time-varying threshold) is tracked as a separate future project.

## Schema Registry

### `nbody_pipeline.schemas`

VO-safe (`^[a-z][a-z0-9_]*$` + unit-suffixed) column schema definitions for the Parquet/DuckDB analysis layer, one YAML file per table. Each column carries `dtype`, `unit`, `ucd`, `description`, `public`, and `nullable`. See `docs/analysis_architecture.md` for the caching layers this schema registry backs.

```python
from nbody_pipeline.schemas import load_table_schema

schema = load_table_schema("compact_object_history")
schema.column_names()   # -> tuple of VO-safe column names
schema.empty_dataframe()  # correctly-typed empty frame, useful as a cache sentinel
schema.schema_hash()    # sha1 over (name, dtype) pairs; bumps when columns change
schema.validate_dataframe(df)  # raises SchemaValidationError on mismatch
```

Registered tables: `compact_object_history` (per-snapshot rows for compact objects, KW 10-14), `snapshot_summary` (one row per TTOT with population counts and structural parameters).

## Query Layer

### `nbody_pipeline.query`

DuckDB read path over the Parquet feature store written by `ParquetDatasetCacheMixin`/`ParquetTableCacheMixin` tasks (`nbody_pipeline.analysis.parquet_cache`). Prefer this over loading raw HDF5 for anything that only needs the small pre-computed feature tables.

```python
from nbody_pipeline.config import ConfigManager
from nbody_pipeline.query import load_feature, duckdb_connect

config = ConfigManager()

# One simulation, filtered, as a DataFrame:
bh = load_feature(
    config, "20sb", "compact_object_history",
    columns=["ttot", "object_id", "mass_msun"],
    where="kw = 14 AND mass_msun > ?", params=[20.0],
)

# Cross-simulation SQL, one VIEW per feature (union'd on simulation_id):
con = duckdb_connect(config)
con.execute("SELECT simulation_id, count(*) FROM compact_object_history GROUP BY 1").df()
```

`load_feature` and `duckdb_connect` transparently handle both on-disk layouts (a directory of per-source-file Parquet parts, or one merged Parquet table) and raise a friendly error pointing at `python -m nbody_pipeline analyze` when a feature hasn't been built yet for a simulation. `duckdb_connect` skips (rather than errors on) any simulation/feature pair with no data on disk.

## Visualization

### Base Classes

- `BaseVisualizer`: Base class for all visualizers
- `BaseHDF5Visualizer`: Base for HDF5-based visualizations

### Visualizer Classes

- `SingleStarVisualizer`: Visualize single star properties
- `BinaryStarVisualizer`: Visualize binary star systems
- `LagrVisualizer`: Visualize Lagrangian radii evolution
- `GalacticOrbitVisualizer`: Visualize cluster galactic orbits as static 2D projections and Plotly 3D HTML
- `CollCoalVisualizer`: Visualize collision and coalescence events

### `nbody_pipeline.visualization.evolution_path.EvolutionPathVisualizer`

Reusable Dragon-2-style "evolution path" cartoon diagram (see Rizzuto et al. 2023,
doi:10.1093/mnras/stad2292 Fig. 1): a vertical sequence of epochs with member stars drawn as circles
(size ~ mass via `mass_to_display_radius`, colour ~ stellar type via `DEFAULT_KW_COLORS`), binary pairs
linked by a dashed orbit ellipse or a filled common-envelope blob, and same-object continuity across
epochs drawn as connecting lines. The data model (`PathMember`, `PathBinary`, `PathEpoch`,
`EvolutionPath`) is study-agnostic -- callers populate it from whatever analysis they are doing, it does
not know about `kstar_semantics`.

```python
from nbody_pipeline.visualization import (
    EvolutionPath, EvolutionPathVisualizer, PathBinary, PathEpoch, PathMember,
)

path = EvolutionPath(
    title="20sb 428467",
    epochs=[
        PathEpoch(0.0, [PathMember("A", 25.0, 1), PathMember("B", 20.0, 1)],
                  [PathBinary(("A", "B"), a=2.0, e=0.1)], event_label="t0 primordial"),
        PathEpoch(3298.0, [PathMember("A", 8.0, 14), PathMember("C", 15.0, 1)],
                  [PathBinary(("A", "C"), a=1.4, e=0.45)], event_label="exchange"),
    ],
)
EvolutionPathVisualizer(config).plot(path, "evolution_path_20sb_428467.jpg")
```

`assign_member_columns(path)` (a pure, matplotlib-free helper) computes the stable per-`object_id`
horizontal layout: a column, once assigned, never changes; binary members get adjacent columns; a
column vacated by an object that never reappears is reused by a later, unrelated member, so an
exchanged-in partner tends to land next to its new companion.

## Utilities

### `nbody_pipeline.utils`

- `save()`, `read()`: Pickle serialization
- `get_output()`: Execute shell commands
- `log_time`: Decorator for timing functions
- `BlackbodyColorConverter`: Temperature to RGB color conversion
