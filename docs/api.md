# Dragon3 Pipelines API Reference

## Configuration

### `dragon3_pipelines.config.ConfigManager`

Main configuration class for managing simulation settings.

```python
from dragon3_pipelines.config import ConfigManager, load_config

# Load default configuration
config = ConfigManager()

# Load with custom YAML
config = load_config("my_config.yaml")
```

## I/O Operations

### `dragon3_pipelines.io.HDF5FileProcessor`

Read and process HDF5 simulation files.

### `dragon3_pipelines.io.LagrFileProcessor`

Process Lagrangian radii files.

### `dragon3_pipelines.io.text_parsers`

Functions for parsing text-based simulation output files.

## Analysis

### `dragon3_pipelines.analysis.ParticleTracker`

Track individual particles through simulation snapshots.

### `dragon3_pipelines.analysis.BinaryStellarTypeExtractor`

Extract complete processed binary rows where either component matches a StellarType abbreviation or KW code.

```python
from dragon3_pipelines.analysis import BinaryStellarTypeExtractor
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
extractor = BinaryStellarTypeExtractor(config)
bh_binaries = extractor.load_binaries_with_stellar_type("20sb", stellar_type="BH")
ns_binaries = extractor.load_binaries_with_stellar_type("20sb", kw="13")
```

Specify exactly one of `stellar_type` or `kw`. StellarType abbreviations are matched case-insensitively using `default_config.yaml` `stellar_types`.
Pass `force=True` to rebuild the analysis cache from HDF5 files.

### `dragon3_pipelines.analysis.BTypeBinaryExtractor`

Extract complete processed binary rows where either component satisfies the project B-type main-sequence criteria: `Bin KW* == 1`, `10500 <= Bin Teff* <= 31500`, and `2.75 <= Bin M* <= 17.7`.

```python
from dragon3_pipelines.analysis import BTypeBinaryExtractor
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
df = BTypeBinaryExtractor(config).load_b_type_binaries("20sb")
```

The returned table preserves the processed binary rows and adds `b_type_member1`, `b_type_member2`, `b_type_member_count`, `b_type_pair_key`, and `is_primordial_binary`. Results are cached under `paths.analysis_cache_dir`.

### `dragon3_pipelines.analysis.InitialTotalMassAnalyzer`

Load the initial cluster mass from `lagr.7` using the `100%` shell total mass:
`total_mass = avmass * nshell` at strict `Time[Myr] == 0.0`.

```python
from dragon3_pipelines.analysis import InitialTotalMassAnalyzer
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
mass_msun = InitialTotalMassAnalyzer(config).get_initial_total_mass_msun("20sb")
```

The scalar result is cached under `initial_total_mass/initial_total_mass.feather`.
Pass `force=True` to rebuild it from `lagr.7`.

### `dragon3_pipelines.analysis.IntermediateMassBlackHoleAnalyzer`

Scan HDF5 snapshots for intermediate-mass black hole candidates, defined as `KW == 14` and `100 < mass < 1e5` solar mass. Single-star candidates use `M`; binary components use `Bin M1*` and `Bin M2*`. Physical merger lineage is loaded from the concatenated `coll.13` and `coal.24` continuous files, not from the HDF5 `mergers` table.

```python
from dragon3_pipelines.analysis import IntermediateMassBlackHoleAnalyzer
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
analyzer = IntermediateMassBlackHoleAnalyzer(config)
summary = analyzer.summarize_simulation("20sb")
snapshots = analyzer.load_imbh_snapshots("20sb")
events = analyzer.load_imbh_merger_events("20sb")
```

`summarize_simulation()` returns `summary`, `objects`, `snapshots`, and `merger_events`. Results are cached under `intermediate_mass_black_hole` as `imbh_snapshots.feather` and `imbh_true_merger_events.feather`.

### `dragon3_pipelines.analysis.hdf5_scan.HDF5ScanSession`

Batch compatible HDF5 data-reduction tasks so each HDF5 file is read once per simulation/options group.

```python
from dragon3_pipelines.analysis import BTypeBinaryExtractor, BinaryStellarTypeExtractor
from dragon3_pipelines.analysis.hdf5_scan import HDF5ScanSession
from dragon3_pipelines.config import ConfigManager

config = ConfigManager()
session = HDF5ScanSession(config)
session.add_job(BinaryStellarTypeExtractor(config).build_scan_job("20sb", stellar_type="BH"))
session.add_job(BTypeBinaryExtractor(config).build_scan_job("20sb"))
results = session.run()
```

`HDF5ScanOptions` defaults to tail-incremental cache validation. Use `force=True` to ignore old cache/meta and rebuild from scratch.

Developer extension pattern: new analysis/data-reduction features that scan HDF5 files for small per-snapshot records should implement an `HDF5ScanTask`, then expose it through a thin analysis class inheriting `ScanBackedAnalysisBase`. Keep extraction, merge, cache path, metadata, and freshness semantics in the task; use the base only for option merging and single-job run/cache loading.

### `dragon3_pipelines.analysis.CompactBinaryCounter`

Count compact binary categories across snapshots. `summarize_simulation()` returns the existing `{"summary": ..., "details": ...}` structure and caches per-snapshot category hits under `compact_binary_count`.

### `dragon3_pipelines.analysis.GalacticOrbitProcessor`

Build cluster galactic-orbit points from scalar HDF5 snapshots. `load_plot_data()` returns de-duplicated rows containing `TTOT`, `Time[Myr]`, `RG(1..3)`, `VG(1..3)`, and source tracking columns. Results are cached under `galactic_orbit`.

### `dragon3_pipelines.analysis.GalacticEnergyAngularMomentumProcessor`

Compute per-star galactocentric kinetic energy, potential energy, total energy, and `L_z` for one HDF5 snapshot. `compute_snapshot(single_df_at_t, scalar_row_at_t)` returns a copy of the single-star DataFrame with `E_kin_gal[Msun*(km/s)^2]`, `E_pot_gal[Msun*(km/s)^2]`, `E_gal[Msun*(km/s)^2]`, and `L_z_gal[Msun*kpc*km/s]`. The calculation uses `galpy` `MWPotential2014`; scalar `RG/VG` are interpreted as pc/km/s and added directly to star offsets.

### `dragon3_pipelines.analysis.tau_gw`

Calculate gravitational wave merger timescales.

## Schema Registry

### `dragon3_pipelines.schemas`

VO-safe (`^[a-z][a-z0-9_]*$` + unit-suffixed) column schema definitions for the Parquet/DuckDB analysis layer, one YAML file per table. Each column carries `dtype`, `unit`, `ucd`, `description`, `public`, and `nullable`. See `docs/analysis_architecture.md` for the caching layers this schema registry backs.

```python
from dragon3_pipelines.schemas import load_table_schema

schema = load_table_schema("compact_object_history")
schema.column_names()   # -> tuple of VO-safe column names
schema.empty_dataframe()  # correctly-typed empty frame, useful as a cache sentinel
schema.schema_hash()    # sha1 over (name, dtype) pairs; bumps when columns change
schema.validate_dataframe(df)  # raises SchemaValidationError on mismatch
```

Registered tables: `compact_object_history` (per-snapshot rows for compact objects, KW 10-14), `snapshot_summary` (one row per TTOT with population counts and structural parameters).

## Query Layer

### `dragon3_pipelines.query`

DuckDB read path over the Parquet feature store written by `ParquetDatasetCacheMixin`/`ParquetTableCacheMixin` tasks (`dragon3_pipelines.analysis.parquet_cache`). Prefer this over loading raw HDF5 for anything that only needs the small pre-computed feature tables.

```python
from dragon3_pipelines.config import ConfigManager
from dragon3_pipelines.query import load_feature, duckdb_connect

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

`load_feature` and `duckdb_connect` transparently handle both on-disk layouts (a directory of per-source-file Parquet parts, or one merged Parquet table) and raise a friendly error pointing at `python -m dragon3_pipelines analyze` when a feature hasn't been built yet for a simulation. `duckdb_connect` skips (rather than errors on) any simulation/feature pair with no data on disk.

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

## Utilities

### `dragon3_pipelines.utils`

- `save()`, `read()`: Pickle serialization
- `get_output()`: Execute shell commands
- `log_time`: Decorator for timing functions
- `BlackbodyColorConverter`: Temperature to RGB color conversion
