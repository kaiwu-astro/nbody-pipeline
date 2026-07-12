# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Breaking:** project renamed from `dragon3_pipelines` to `nbody-pipeline`. The
  import package is now `nbody_pipeline`, the installed CLI script is `nbody-plot`
  (was `dragon3-plot`), the config discovery env var is `NBODY_CONFIG` (was
  `DRAGON3_CONFIG`), the discovered config filename is `nbody_config.yaml` (was
  `dragon3_config.yaml`), and the default user cache directory moved to
  `~/.cache/nbody_pipeline/` (was `~/.cache/dragon3_pipelines/`).

### Added
- Full snapshot particle lake (`nbody_pipeline.analysis.particle_lake`): four new
  VO-safe Parquet feature tables -- `snapshot_singles`, `snapshot_binaries`,
  `snapshot_mergers`, `snapshot_scalars` -- covering every snapshot (not just compact
  objects) with raw, unclipped source values (no NS/BH display clipping, no
  force/force-derivative integrator columns). Built via `python -m nbody_pipeline
  analyze --features lake`; not run by the nightly `update_analysis_store` due to its
  size. Requires an optional second storage root, `paths.lake_dir` (see
  `config.example.yaml`); falls back to `analysis_cache_dir` when unset. New
  `scripts/lake_preflight.py` read-only pre-flight check for duplicate/overlapping HDF5
  files before a full-simulation rebuild. Cross-file TTOT duplicates (restart-boundary
  checkpoints written by two consecutive run directories) are resolved at write time,
  per-TTOT (`ParticleLakeProcessor`/`compute_ttot_dedup_exclusions`, cached at
  `<lake_dir>/<simu>/ttot_dedup_map.json`) rather than by excluding whole run
  directories, so no legitimate snapshot data is discarded. `snapshot_binaries.cm_id`
  and `snapshot_mergers.cm_id` are documented as NOT reliably unique within one
  snapshot -- confirmed against real pilot data and NBODY6++GPU source
  (`custom_output.F`) that KS-pair and wide-binary center-of-mass names use
  independent, numerically colliding schemes; use (`object_id_1`, `object_id_2`[,
  `object_id_3`]) as the per-snapshot unique key instead (also now the part-internal
  sort key). Running the full three-simulation build surfaced two more real-archive
  edge cases, both fixed: `snapshot_binaries.bin_label` now falls back to sentinel
  `-9` ("unknown") for files that predate the `Bin Label`/`Bin cm Name` dataset
  entirely (confirmed on `0sb`'s `old_run_archive/snap.40/`, 435 files, every other
  `Bin *` column present), and a new `HDF5ScanOptions.skip_unreadable_files` option
  (default off; on by default only for `ParticleLakeProcessor`) logs and skips
  genuinely corrupted HDF5 files (h5py `RuntimeError`/`OSError`, e.g. "wrong B-tree
  signature") instead of aborting the whole scan; `compact_object_history`/
  `snapshot_summary` keep the original fail-fast-and-checkpoint behavior.
  `ParquetDatasetCacheMixin.write_part` now writes to a per-call-unique tmp
  filename (pid + random suffix) and retries `os.replace()` with backoff on
  `FileNotFoundError` (defense in depth). The actual root cause of the real
  full-archive build's `FileNotFoundError` under `parallel=True`: mid-run
  checkpoints (periodic or crash-flush) ran `write_cache_and_meta` ->
  `_prune_orphan_parts` in the main process while *other* workers could still
  be mid-`write_part`, so a checkpoint's unconditional "not yet in
  `processed_files` == orphan" prune could delete another worker's brand-new
  tmp/part file out from under it. `HDF5ScanTask.write_cache_and_meta` gained
  a `prune_orphans: bool = True` parameter; the runner now passes `False` for
  every mid-run checkpoint and only prunes at the one provably-safe point
  (after every worker's result has been consumed, at the very end of a scan).
  `compact_object_history`/`snapshot_summary`'s feather-backed cache mixins
  accept and ignore the new parameter (no orphan-part concept there). Full
  build completed for `0sb`/`20sb`/`60sb` (12689 source files, ~4.6 TiB
  Parquet output) with post-mortem validation (row-count conservation against
  `snapshot_scalars.n_single/n_binary/n_merger`, uniqueness of `(ttot,
  object_id[, _2])`). Two residual, real-archive findings from the repeated
  crash/retry cycle while landing the fixes above (not from the fixes
  themselves), both documented in detail in
  `docs/analysis_architecture.md` Roadmap #5: a small number of stale
  `processed_files` manifest entries left by the *pre-fix* checkpoint race
  (repaired by removing entries whose declared part is missing on disk and
  rerunning incrementally), and one occurrence of a corrupted file winning a
  cross-file TTOT dedup tie-break it could not honor, silently excluding its
  valid competitor (repaired by hand for the one affected TTOT; the general
  case -- `compute_ttot_dedup_exclusions` picking a winner before anything
  has tried to read it -- remains an open, documented, extremely-rare edge
  case). A separate, small, still-open inconsistency: `snapshot_scalars`'s
  TTOT dedup (`replace_ttot_rows`, "last file processed wins") can disagree
  with the other three tables' (`compute_ttot_dedup_exclusions`, "latest
  mtime wins") for restart-boundary TTOTs with near-but-not-identical
  contributing files; measured impact after the repairs above is under
  10^-4 % of total rows in all three simulations.
- `HDF5FileProcessor.read_raw_tables` / `nbody_pipeline.io.text_parsers.raw_dataframes_from_hdf5_file`:
  an h5py-level raw HDF5 reader (column-projected, source dtypes preserved, no L1
  feather cache writes) for `HDF5ScanTask`s that declare `hdf5_reader_kind = "raw"`.
- `CHANGELOG.md`, `CITATION.cff`, `config.example.yaml`, and a tracked JUWELS/madnuc
  site config (`configs/juwels_madnuc.yaml`).
- `scripts/release.sh` release helper and a versioning/changelog workflow section in
  `AGENTS.md`.
- `requirements.lock` (pinned snapshot of the environment used to produce current
  results) and a config discovery mechanism (`--config` / `DRAGON3_CONFIG` /
  `./dragon3_config.yaml`).

### Changed
- `pyproject.toml` now sources the package version dynamically from
  `dragon3_pipelines.__version__` instead of duplicating it.
- `config/default_config.yaml` no longer ships hardcoded personal JUWELS paths;
  `paths.simulations`/`plot_dir`/`analysis_cache_dir` default to empty, and
  `ConfigManager` raises a clear, actionable error pointing at
  `config.example.yaml` when required paths are missing.
- `BlackbodyColorConverter` and `load_GWTC_catalog` no longer default to a
  developer's personal absolute path; they fall back to a user cache directory
  or a configurable path, respectively.

### Fixed
- `README.md` permissions (was `600`, unreadable by other group members on the
  shared filesystem).
- `HDF5ScanTask` Parquet/feather manifests no longer treat a `scan.parallel` mismatch
  (e.g. a login-node pilot run followed by an sbatch run for the same
  simulation/feature) as an "options changed" full rebuild -- previously this could
  silently delete and reprocess an entire multi-terabyte Parquet dataset.

## [1.0.0] - 2026-07-10

Initial tagged release, consolidating roughly a year of iterative development into a
stable, documented package. Highlights:

### Added
- HDF5 (`.h5part`) simulation data ingestion (`HDF5FileProcessor`) covering
  scalars, singles, binaries, and merger/collision tables, with Feather-based
  read-acceleration caching.
- Particle trajectory tracking (`ParticleTracker`, `ParticleHistoryVisualizer`)
  with per-particle progress checkpointing and parallel processing.
- Binary star analysis: mass ratio, orbital parameters, gravitational-wave
  merger timescale (`tau_gw`), primordial/B-type/compact-binary extraction, and
  IMBH candidate identification.
- Lagrangian radii processing (`LagrFileProcessor`, `LagrVisualizer`) including
  current-mass Lagrangian radii derived directly from HDF5 snapshots.
- Cluster galactic-orbit and galactic energy/angular-momentum analysis built on
  `galpy`'s `MWPotential2014`.
- A unified `HDF5ScanTask` / `HDF5ScanSession` architecture so multiple
  data-reduction tasks share a single pass over HDF5 files, replacing the
  earlier ad hoc macro/micro split.
- A VO-safe Parquet feature store (`compact_object_history`, `snapshot_summary`)
  with a schema registry under `dragon3_pipelines/schemas/` and a DuckDB query
  entry point over the Parquet store.
- CLI entry points: `python -m dragon3_pipelines` (main plotting pipeline),
  `purge` (preview/delete generated plots), and `analyze` (build/refresh the
  Parquet feature store), plus the installed `dragon3-plot` script.
- Extensive visualization suite (single-star, binary-star, Lagrangian,
  collision/coalescence, galactic orbit) and a 250+ test pytest suite with a
  shared `./scripts/ci.sh` entry point, GitHub Actions CI, and pre-commit hooks
  (black, ruff).

[Unreleased]: https://github.com/kaiwu-astro/nbody-pipeline/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kaiwu-astro/nbody-pipeline/releases/tag/v1.0.0
