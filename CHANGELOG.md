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
  case). A separate small inconsistency is now fixed: `snapshot_scalars`'s
  TTOT dedup (`replace_ttot_rows`, "last file processed wins") could
  disagree with the other three tables' (`compute_ttot_dedup_exclusions`,
  "latest mtime wins") for restart-boundary TTOTs with near-but-not-identical
  contributing files (measured impact before the fix was under 10^-4 % of
  total rows in all three simulations, but real). `SnapshotScalarsTask` now
  takes and applies the same `excluded_ttot_by_path` as the other three
  tasks; `schema_version` bumped 1 -> 2 to force a retroactive full rebuild
  of just `snapshot_scalars` (the other three tables are untouched).
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

### Added
- `nbody_pipeline.analysis.kstar_semantics`: `decode_cm_kstar`, `decode_member_kw`,
  `annotate_binary_states` -- pure/vectorized decoders for the raw binary
  centre-of-mass `cm_kw` (HDF5 Item 160) and member `kw_1`/`kw_2` (Item 158/159)
  codes, for the Gaia-BH-analog step2 evolution-path study
  (`examples/gaia_bh_formation/step2/`). Also decodes `cm_kw == -1` (`chaos.f`'s
  Mardling 1995 chaotic-tidal-interaction state, unrelated to mass transfer --
  ~11% of rows on real 0sb target-binary data).
- `nbody_pipeline.analysis.physics.binding_energy_nb`/`ebind_over_kt`: binding-energy
  helpers reproducing `hdf5_reader.py`'s existing `Ebind_abs_NBODY`/`Ebind/kT` formulas
  exactly, so step2 binding-energy continuity checks stay numerically comparable to
  the existing step1 CSVs.
- `nbody_pipeline.visualization.evolution_path.EvolutionPathVisualizer` (plus the
  `PathMember`/`PathBinary`/`PathEpoch`/`EvolutionPath` data model and the pure
  `mass_to_display_radius`/`assign_member_columns` helpers): a reusable Dragon-2-style
  ("Formation of an IMBH..." cartoon, Rizzuto et al. 2023) evolution-path diagram
  renderer, decoupled from any specific study.

### Changed
- Clarified, no code change needed: source review of `roche.f`/`expel.f`
  (Nbody6PPGPU-beijing) confirms the existing `cm_kw` odd/even convention already
  used by `hdf5_reader.py`'s `mark_funny_star_binary` and documented in
  `schemas/snapshot_binaries.yaml` (odd >=11 = mass transfer ongoing, even >=10 =
  between episodes) matches the actual runtime behaviour. The NBODY6++GPU manual's
  prose (`Nb6manual.md` line 1431) states this backwards; see the
  `nbody_pipeline.analysis.kstar_semantics` module docstring for the full derivation.
- Documented a known, pre-existing normalization limitation (not fixed here):
  `hdf5_reader.py`'s `Ebind/kT`/`is_hard_binary` columns divide by the fixed config
  constant `config.ECLOSE_INPUT` (default 1.0) rather than the real per-snapshot
  `ECLOSE` value, which genuinely varies over a run (e.g. 0.003-1.0 within 20sb)
  because `adjust.F` re-tunes it. `physics.ebind_over_kt` reproduces the existing
  (constant-based) behaviour for continuity with step1; a proper fix using the real
  time-varying threshold is tracked as a separate future project.

### Changed
- **Breaking semantics:** `hdf5_reader.py`'s `Ebind/kT` binary-table column now divides
  by `TEMPORARY_EBIND_FACTOR * ECLOSE` (the true per-snapshot `ECLOSE`, broadcast per-row
  via `TTOT.map`) instead of the fixed `config.ECLOSE_INPUT`, fixing the normalization
  limitation documented above. `is_hard_binary` keeps its column name (many external
  consumers) but is redefined as `binary_class == "hard"` rather than `Ebind/kT >= 1`.
  `default_config.yaml`'s `limits['Ebind/kT']` widened to `[1.0e-4, 1.0e9]` to
  accommodate the ~1e3-3e5x larger values under the new denominator (20sb has
  `ECLOSE` in ~0.003-1.0); will be tightened in a follow-up commit once real min/max
  values are on record. `physics.ECLOSE_INPUT` config key is kept (still read by older
  step2 scripts) but is annotated as legacy/unused by `hdf5_reader.py`.
- The L1 feather cache (`{hdf5_path}.{table}.df.feather`) has no version marker, so
  `read_file`/`read_tables` now lazily detect and self-heal stale binaries caches
  written before this change (missing `binary_class` column): `read_file` re-reads the
  source HDF5 and rewrites the cache; `read_tables`'s column-projected reads raise and
  fall back to `read_file(write_cache=False)`, since a projection that never requests
  `binary_class` would otherwise never notice the cache was stale.

### Added
- Hard/soft/temporary binary reclassification (2026-07 meeting,
  `examples/soft-hard-temp/meeting.md`): `nbody_pipeline.analysis.physics` gains
  `mean_core_interparticle_distance_au` (NC/RC-geometry core spacing estimate),
  `classify_binaries` (hard = `Bin Label`==1; temporary = wider than core spacing *and*
  weakly bound; else soft), `add_binary_energetics_and_class` (particle-lake read-time
  join/classify helper -- lake tables themselves stay schema-frozen), and
  `drop_temporary_binaries`. `hdf5_reader.py`'s binaries table gains
  `mean_core_interparticle_distance[au]` and `binary_class` (`pd.Categorical`,
  hard/soft/temporary) columns.
- `BinaryStarVisualizer.create_ebind_semi_plot_class_scatter` (new
  `BaseHDF5Visualizer._create_jointplot_class_scatter` engine, parallel to
  `_create_jointplot_density`): Ebind/kT-vs-a scatter of all binaries in a snapshot,
  colored by `binary_class` (color-blind-safe palette), with stacked marginal
  histograms. Registered in `__main__.py`'s binary plotting block and
  `visualization/purge.py`'s `BINARY_TARGETS`.
- `hdf5_reader.py`'s binaries table gains an `eclose_nb` column (the true per-snapshot
  `ECLOSE` broadcast per row) alongside the existing
  `mean_core_interparticle_distance[au]`, so plotting code can label the temporary
  threshold with its actual per-snapshot value instead of just the normalized `y=1`.
- `examples/soft-hard-temp/run_trial.py`: trial script comparing binary-table
  populations before/after `drop_temporary_binaries` cleanup across 9 representative
  20sb snapshots, focused on the compact-object a-vs-primary-mass plot; writes
  `class_counts.csv` with per-snapshot hard/soft/temporary counts and Ebind/kT ranges.

### Fixed
- `_decorator_ebin_semi` (`ebind_vs_a` density/compact-object/class-scatter plots): the
  single horizontal `y=1` line was misleading once hard/soft stopped coming from one
  threshold -- `temporary` requires *both* `a > d_av` *and* `Ebind/kT < 1`, a corner of
  the plot, not a line. Now draws two perpendicular dashed segments (horizontal at
  `y=1` from `x=d_av` rightward, vertical at `x=d_av` from the bottom up to `y=1`)
  outlining that corner, each labeled along its own direction with the actual
  per-snapshot criterion value (`Ebind=1e-3*Eclose=<value>`, `a=d_av=<value>`, 2
  decimal places, scientific notation). The hard/soft/temporary count annotation
  (already using the new three-way `binary_class` definition -- no data bug there)
  moved from the top-right corner to the bottom-left (`ax.transAxes`), since the top
  right is now used by the criterion labels; falls back to bottom-right when the axes
  already has a table (the compact-object-only variant's Stellar Types count table
  also sits at `loc="lower left"`, so both must not claim the same corner).
- `get_all_hdf5_paths`: two different physical files can share the same
  filename-derived sort index when a simulation's data has been partially
  re-generated into a different directory with the same basename (confirmed for
  20sb: `snap.40_0.h5part` exists both under `.../archive/` -- an old, incomplete copy
  missing the `Bin Label` dataset -- and under `.../snap.40/` -- a later re-generated,
  complete copy with `Bin Label`). Previously both stayed in the candidate list with a
  tied sort key, so `_locate_hdf5_path`-style single-match lookups picked whichever
  file `glob()` happened to traverse first -- for 20sb this silently selected the
  incomplete `archive/` copy for `TTOT=1.0`, dropping all 35481 real hard binaries at
  that snapshot into the `-9`-sentinel "unknown" fallback. Now dedupes by
  filename-index, keeping the larger file (a cheap, reliable proxy for "more complete"
  without opening every HDF5 file).
- Documented, not fixable in code: 20sb's raw archived `snap.40_N.h5part` files
  genuinely lack the `Bin Label` dataset for roughly `TTOT` in `(1, ~600-1000)` --
  confirmed by direct inspection of `snap.40_9.h5part`/`snap.40_99.h5part`/
  `snap.40_500.h5part` (no `Bin Label` in any `Step#` group) vs. `snap.40_1000.h5part`
  onward (present). Per user: `Bin Label` was added to the NBODY6++GPU output code
  partway through this run, so its absence in the early archived output is expected,
  not a bug -- the `-9`-sentinel fallback (all binaries classified non-hard) is
  correct behaviour for that window, not something to work around.

### Added
- `nbody_pipeline.analysis.bin_label_backfill` (`python -m nbody_pipeline
  backfill-bin-label`): reconstructs `snapshot_binaries.bin_label` for the ~1258
  cached parquet rows still carrying the `-9` ("unknown") sentinel because their
  source HDF5 predates the `"176 Bin Label"`/`"176 Bin cm Name"` dataset (NBODY6++GPU
  commit `1381398e`, corrected `b9ac5a0e`). The label is fully derivable from fields
  present in every source version (>= Aug2021): KS pairs' cm name is `NZERO` +
  first-member `NAME` at pairing time (`ksinit.F:30`); the wide branch hard-codes
  `GAMMA=0`/`cm_kw=-1`; the merger branch negates the cm name (`merge.f:263`); the
  remainder falls to the `GALIMIT` perturbation threshold (`custom_output.F:415`) for
  hierarchical/outer binaries. `reconstruct_bin_label` is a pure, vectorized function
  applied only to `-9` rows (real labels are never touched); `resolve_nzero` infers
  NZERO from the cached `snapshot_scalars` row at `ttot == 0` (or takes an explicit
  `--nzero SIMU=N` override) and refuses to guess if the cache's earliest row isn't
  really `t=0`; `backfill_parts` rewrites only the affected Parquet parts in place via
  the existing `write_part` (same deterministic part filename, so neither the
  manifest nor `schema_hash` changes -- no full-rebuild trigger) and is idempotent (a
  part with nothing left to change is not rewritten). `--validate` cross-tabulates the
  algorithm against a simulation that already has real labels before it's trusted on
  data with no ground truth. A separate contamination audit (all 1249
  `job-*.out` + per-file HDF5 dataset-name checks) found 202 archived files
  mislabeled by a short-lived buggy build (between the two commits above) that
  already carry a *real* value under the same `"176"` slot (just named `"Bin cm
  Name"` instead of `"Bin Label"`) -- `particle_lake.py`'s raw reader already maps
  that fallback name to the same logical column, so those files are correctly
  excluded from backfill by construction (only genuine `-9` rows are ever touched).

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
