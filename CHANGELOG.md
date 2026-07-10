# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

[Unreleased]: https://github.com/kaiwu-astro/dragon3_pipeline/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kaiwu-astro/dragon3_pipeline/releases/tag/v1.0.0
