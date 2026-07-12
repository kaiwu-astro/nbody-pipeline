# Unified Analysis Architecture

This document describes the target architecture for data reduction, caching, and
querying in `nbody_pipeline`, and the roadmap toward a future VO (Virtual
Observatory) data release. It supersedes the historical "macroscopic scan /
microscopic per-file plot" module split that used to be documented in
`AGENTS.md`.

## Why this document exists

Two internal design discussions (`HDF5scan和Process重新设计.md`,
`高频读取和VO兼容.md`) identified two compounding problems:

1. **Architecture drift**: the "macro = hdf5_scan, micro = hdf5processor/plotter"
   split was a historical accident, not an intentional module boundary. Several
   tasks (`BTypeBinaryTask`, `BinaryStellarTypeTask`,
   `IntermediateMassBlackHoleTask`) already emit per-object rows from inside the
   scan framework, breaking that split in practice.
2. **Query performance**: a single filtered query across a whole simulation via
   the historical HDF5-loop-into-pandas path takes on the order of an hour. The
   fix is a columnar Parquet analysis layer queried with DuckDB, backed by
   small, purpose-built tables (e.g. compact objects, snapshot summaries)
   rather than a full particle-level table for every query.
3. **Future VO data release**: eventual TAP/ADQL/VOTable publication requires
   SQL-safe, documented column names. Retrofitting internal column names like
   `X [pc]` or `Ebind/kT` after the fact is expensive. New tables should be
   VO-safe and schema-registered from day one.

`HDF5ScanRunner` (`nbody_pipeline/analysis/hdf5_scan.py`) is a mature asset —
single-pass file reads, mtime-based incremental rebuilds, schema-version
invalidation, checkpointing, and parallel execution, all covered by tests. It
is promoted here to be the **one** analysis pipeline, not a "macro-only" tool.

## Data layers

| Layer | Location | Authority | Invalidation |
| --- | --- | --- | --- |
| **L0 - raw HDF5** | source `.h5part` files | authoritative, read-only | n/a |
| **L1 - reader table cache** | `{path}.{table}.df.feather` next to the source file | derived, per-file | **existence-only** (known limitation, see Roadmap #1) |
| **L2 - feature store** | `{analysis_cache_dir}/{simu}/{feature}/` | derived, per-feature | mtime of source files + `schema_version`; legacy features use feather+JSON meta, new features use Parquet+manifest (see PR 3) |
| **L3 - release layer** | future `release/` build output | derived, publishable | rebuilt from L2 whenever `public: true` columns change |

L1 is populated by `nbody_pipeline/io/hdf5_reader.py::read_file` and is a
pure per-file cache of the raw tables plus a fixed set of derived columns. L2
is populated by `HDF5ScanTask` implementations via `HDF5ScanRunner`. L3 does
not exist yet; it is scoped for a later roadmap item.

## Task output-type taxonomy

Every analysis task is an `HDF5ScanTask`, regardless of how much data it emits
per snapshot. Output *cardinality* is not a module boundary, it is a
property of the task's schema:

- **`snapshot_scalar`**: one row (or a handful of rows) per snapshot/TTOT,
  e.g. a time series of a global statistic. Typically cached with
  `ParquetTableCacheMixin` or the legacy `FeatherMetaCacheMixin` (single
  merged table in memory).
- **`object_rows`**: many rows per snapshot, one per object of interest
  (e.g. every compact object). Typically cached with
  `ParquetDatasetCacheMixin` (one Parquet part per source file, never
  merged into memory as a whole).
- **`events`**: irregular, snapshot-independent records (mergers, escapes).
  Roadmap only (see Roadmap #4); will reuse the `object_rows` dataset shape.
- **`plot`**: a rendered figure rather than a data table. Roadmap only (see
  Roadmap #3); today this remains the responsibility of
  `SimulationPlotter.plot_hdf5_file`, which is frozen to its current set of
  per-file visualizers.

All four types are implemented as `HDF5ScanTask` and run through
`HDF5ScanRunner`/`HDF5ScanSession`. There is no separate "microscopic" code
path.

## Normative policies

- **Scan-task-only rule**: any new analysis/data-reduction feature that loops
  over HDF5 files must be implemented as an `HDF5ScanTask` executed by
  `HDF5ScanRunner`. Do not write a new ad hoc file-iteration loop. The
  outer analysis class should subclass `ScanBackedAnalysisBase` and keep
  `build_scan_job()` thin; extraction, merge, cache-path, and meta semantics
  live in the task itself.
- **`read_file` derived-column freeze**: the set of columns `read_file`
  currently derives is frozen. Do not add new derived columns to the reader.
  Heavy or science-specific derived quantities belong in a dedicated feature
  task instead (see Roadmap #2 for why some existing derived columns need to
  move out of the reader).
- **`hdf5_reader_kind = "raw"` for tasks needing untouched source values**: a
  task that must not see `read_file`'s NS/BH display clipping, or must not
  read/write the L1 feather cache (e.g. because the source/archive directory
  should never get new files written next to it), sets the class attribute
  `hdf5_reader_kind = "raw"`. `HDF5ScanRunner` then routes that task's file
  read through `HDF5FileProcessor.read_raw_tables` ->
  `raw_dataframes_from_hdf5_file` instead of `read_tables`. All tasks sharing
  one file read must agree on `hdf5_reader_kind`; the runner raises if they
  do not. See `nbody_pipeline.analysis.particle_lake` for the reference
  usage.
- **`plot_hdf5_file` visualizer freeze**: do not add new visualizers to
  `SimulationPlotter.plot_hdf5_file` until the plot-task registry (Roadmap #3)
  exists.
- **VO-safe naming for new persistent tables**: every new L2 feature table
  uses `snake_case` column names matching `^[a-z][a-z0-9_]*$`, with a unit
  suffix where applicable (`mass_msun`, `x_pc`). Existing internal column
  names (e.g. `X [pc]`, `Ebind/kT`) are left as-is; this rule applies only to
  new tables.
- **Schema YAML required**: every new persistent L2 table must have a
  corresponding schema definition in `nbody_pipeline/schemas/` (see PR 2)
  describing dtype, unit, UCD, description, and public/nullable flags for
  each column.
- **Cache layering discipline**: see the table above. Each layer has exactly
  one writer path (the reader for L1, the owning `HDF5ScanTask` for L2) and a
  documented invalidation rule. Do not read/write a layer's cache files from
  outside its owning code path.

## Query layer

Query L2 feature tables with DuckDB over Parquet (`nbody_pipeline/query.py`,
see PR 5). Only pull small, final results into pandas, do not materialize a
full feature table into memory when a query can be pushed down to DuckDB
(column projection, `WHERE` filters). High-frequency/exploratory queries
should target the small, purpose-built candidate tables (e.g.
`compact_object_history`, `snapshot_summary`) rather than a full particle
table.

## Roadmap

Not implemented in this round; listed here so future work has a documented,
ordered plan. Items are loosely ordered by dependency.

1. **Reader table cache versioning** (prerequisite for everything below):
   `{path}.*.df.feather` is currently existence-only invalidated. Add a
   version marker/sidecar so that changes to `read_file` output can properly
   invalidate the L1 cache.
2. **Reader slimming**: move NS/BH `L*`/`Teff*` clipping (currently in
   `read_file`, which bakes display-only artificial values into the data
   layer) into visualization code; audit `tau_gw`/`Ebind` and migrate them to
   feature tasks if warranted. Depends on #1.
3. **Plotter task registry**: introduce an `output_kind="plot"` scan task
   type, seeded from the static `PlotTarget` registry pattern in
   `visualization/purge.py`. Once it exists, `plot_hdf5_file` becomes legacy
   and is frozen (this freeze is already policy as of this document).
4. **Event tables**: `merger_events` (from the `mergers` table plus
   coll.13/coal.24 lineage) and `escaper_events`, using the same
   `ParquetDatasetCacheMixin` shape and a new schema YAML each. These unlock
   an `is_bound`/escape flag for a future `compact_object_history` v2.
5. **Full particle lake** -- implemented. Four new schema-registered Parquet
   tables cover every snapshot (not just compact objects), in
   `nbody_pipeline/analysis/particle_lake.py`:
   - `snapshot_singles` / `snapshot_binaries` / `snapshot_mergers`
     (`object_rows`, `ParquetDatasetCacheMixin`): raw, force-derivative-free
     column subsets (see the column-drop rationale in each schema YAML's
     descriptions), coordinates/velocities/physical quantities downcast to
     `float32` (ids/`ttot`/`time_myr` stay `int64`/`float64`).
   - `snapshot_scalars` (`snapshot_scalar`, `ParquetTableCacheMixin`): one row
     per TTOT, every valid slot of the raw HDF5 scalars array.

   All four tasks set `hdf5_reader_kind = "raw"` (see Normative policies
   below) so they never touch the L1 feather cache and never apply the NS/BH
   display clipping `read_file` bakes in. The three `object_rows` tasks use
   the Risks #2 worker-direct-write escape hatch unconditionally (not just
   under `parallel=True`): `process_file` calls
   `ParquetDatasetCacheMixin.write_part` itself and returns only
   `{"part", "row_count", "ttot_min", "ttot_max", "file_meta"}`, never the
   full DataFrame.

   Storage: an optional second root, `paths.lake_dir` ->
   `config.lake_dir_of[simu]`, routed via
   `nbody_pipeline.analysis.cache_paths.LAKE_FEATURES` (falls back to
   `analysis_cache_dir` when `paths.lake_dir` is unset, so tests/configs that
   never set it keep working unmodified). Not run by nightly
   `update_analysis_store`; build explicitly with `python -m nbody_pipeline
   analyze --features lake`. See `scripts/lake_preflight.py` for the
   read-only duplicate/overlap check that should run before a
   full-simulation (unsampled, `sample_every_nb_time: null`) rebuild.

   **Cross-file TTOT dedup**: real archived simulations have restart-boundary
   snapshot duplication -- two consecutive run directories both write the
   checkpoint at their shared boundary (`scripts/lake_preflight.py` on
   madnuc's 0sb/20sb/60sb found ~1-8 duplicated TTOT per overlapping
   directory pair, never a whole duplicated directory). Renaming an entire
   run directory `*bad*` for this would discard its other, unique snapshots,
   so dedup instead happens per TTOT, at write time:
   `ParticleLakeProcessor.build_scan_jobs()` calls
   `compute_ttot_dedup_exclusions()`, which reads every selected file's
   `Step#` `Time` attrs (cheap, no dataset reads) and picks the file with the
   latest mtime as authoritative for each contested TTOT (mtime, not the
   run-directory's SLURM job ID, since a resubmitted job can have a *lower*
   job ID than the run it superseded but still finish later). The resulting
   `{path: {ttot to drop}}` map is passed into
   `SnapshotSinglesTask`/`SnapshotBinariesTask`/`SnapshotMergersTask`, which
   drop the loser rows in `_build_rows` before writing their part -- so the
   Parquet output never contains a duplicate `(simulation_id, ttot,
   object_id)`. The map is cached at `<lake_dir>/<simu>/ttot_dedup_map.json`,
   keyed by a hash of the exact file list, so an unchanged incremental
   `analyze --features lake` run does not re-read every file's attrs.
   `snapshot_scalars` needs no such mechanism: its `ParquetTableCacheMixin`
   merge (`replace_ttot_rows`) already deduplicates by TTOT for free.

   **Real-archive robustness** (found running the full three-simulation build
   on madnuc's `0sb`/`20sb`/`60sb`): a year-plus archive of restarted jobs
   contains files a clean synthetic test corpus would not. Two confirmed
   cases and their fixes:
   - Some archived files (`old_run_archive/snap.40/*.h5part` on `0sb`, 435
     files) predate the `"176 Bin Label"`/`"176 Bin cm Name"` dataset
     entirely -- every other `Bin *` column is present. `bin_label` falls
     back to the sentinel `-9` ("unknown", documented in
     `snapshot_binaries.yaml`) instead of `KeyError`-ing the file.
   - A small number of files are genuinely corrupted at the HDF5 layer (e.g.
     h5py `RuntimeError("Unable to get group info (wrong B-tree signature)")`
     on a truncated/interrupted write). `HDF5ScanOptions.skip_unreadable_files`
     (default `False`, so `compact_object_history`/`snapshot_summary` keep the
     original fail-fast-and-checkpoint behavior) is set `True` by default only
     for `ParticleLakeProcessor`: a read failure is logged and treated as an
     empty file (every task's empty-tables path already produces a valid
     result) rather than aborting a multi-hour/multi-terabyte scan, and the
     file is still marked processed so it is not retried every run.
   - `ParquetDatasetCacheMixin.write_part`'s tmp file is now named with a
     per-call-unique suffix (`{pid}.{uuid4 hex}.tmp`), not just
     `{part_name}.tmp`: under ~32-way concurrent writes into one `data/`
     directory, the fixed tmp name occasionally raised a spurious
     `FileNotFoundError` from `os.replace()` on this shared filesystem
     (`old_run_archive` is itself a cross-filesystem symlink into the same
     `/e/data1` mount the lake writes to). A unique tmp path removes any
     dependency on that path being untouched by anything else.
6. **VO release export**: a `release/` builder that exports `public: true`
   columns (per the schema registry) to VOTable via `astropy`; unit/UCD
   metadata is already in place by this point.
7. **TAP/DaCHS pilot**: stand up PostgreSQL + DaCHS in front of the release
   tables.

## Risks and mitigations

1. **Parquet part model vs. the existing single-DataFrame merge protocol**:
   resolved by the pass-through `cache_df` design in `ParquetDatasetCacheMixin`
   (PR 3), `merge_file_result` writes a part file directly and returns the
   `cache_df` unchanged; the runner gained one optional `prepare_full_rebuild`
   hook so a Parquet dataset task can clear stale parts on a forced rebuild.
2. **Worker to main-process pickle volume**: for late evolutionary stages with
   many white dwarfs, per-file compact-object row counts can reach
   10^5-10^6 (and full-snapshot singles/binaries row counts are ~10^6-10^7 per
   file). Mitigation order: tight column projection (done from the start),
   then downcast coordinates/velocities to `float32` if needed (schema
   change), then escape hatch of writing the part directly inside
   `process_file` and returning only `{"part", "row_count", "ttot_min",
   "ttot_max", "file_meta"}` to the main process (part paths are unique per
   source file, so there is no write collision). Implemented as
   `ParquetDatasetCacheMixin.write_part`, used unconditionally (serial or
   parallel) by the particle-lake `object_rows` tasks (Roadmap #5); the pilot
   `CompactObjectHistoryTask` still uses the smaller `"rows"` shape since its
   per-file row counts stay in the 10^2-10^4 range.
3. **Concurrent scan invocations**: e.g. `analyze` and a nightly plotting run
   touching the same simulation concurrently. This has the same exposure as
   the existing feather cache today (atomic `os.replace` prevents corruption,
   last-writer-wins). Policy: only run one scan per simulation at a time.
   A lock file is out of scope for this round.
4. **`simulation_id`**: for now this is the config key (e.g. `0sb`, `20sb`,
   `60sb`). Schema descriptions note that a future `simulations` table will
   formalize this at release time.
5. **RG/VG units**: confirm against `GalacticOrbitVisualizer`'s axis labels
   before finalizing the `snapshot_summary` schema's `_kpc` vs `_pc` column
   suffix, this is a blocking item for that schema YAML.
6. **Nullable `Int64` through Parquet**: round-trips correctly, but the
   schema validator must treat pandas nullable `Int64` as part of the int
   family. Covered by a dedicated dtype-compatibility test.
7. **`pyproject.toml` flat `packages` list**: the wheel build already omits
   subpackages today (`analysis/`, `io/`, etc. are not listed), this is a
   pre-existing gap, not something introduced by this work. This round only
   adds `nbody_pipeline.schemas` to the package list and its YAML files to
   package-data; fixing the rest of the subpackage list is out of scope. A
   future switch to `packages.find` is recommended.

## See also

- [API Reference](api.md)
- `nbody_pipeline/analysis/hdf5_scan.py` - `HDF5ScanRunner`,
  `HDF5ScanSession`, and the `HDF5ScanTask` protocol.
