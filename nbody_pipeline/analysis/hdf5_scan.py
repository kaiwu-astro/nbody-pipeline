"""
Unified analysis/data-reduction pipeline for HDF5 simulation snapshots.

This module is the one place that implements looping over HDF5 files to
extract analysis output, regardless of whether a task emits one scalar per
snapshot, many per-object rows per snapshot, or (in the future) event
records. Output cardinality is not a module boundary; see
docs/analysis_architecture.md for the full architecture, the task
output-type taxonomy (snapshot_scalar / object_rows / events / plot), and
caching layers. Per-file plotting (one HDF5 file -> one figure) remains the
responsibility of SimulationPlotter.plot_hdf5_file.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, replace as dataclasses_replace
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
from rich.progress import Progress

from nbody_pipeline.io import HDF5FileProcessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HDF5ScanOptions:
    """Options controlling HDF5 file enumeration and table loading."""

    sample_every_nb_time: float | None = 1.0
    wait_age_hour: int | float = 24
    use_hdf5_cache: bool = True
    parallel: bool = False
    exclude_bad_dirname: bool = True
    force: bool = False
    incremental_from_cache_tail: bool = True
    checkpoint_every_files: int | None = 100
    skip_unreadable_files: bool = False


@dataclass(frozen=True)
class HDF5ScanJob:
    """One scan task queued for one simulation with one option set."""

    simu_name: str
    task: HDF5ScanTask
    options: HDF5ScanOptions


def hdf5_scan_options_from_config(
    config: Any, *, force: bool = False, overrides: Mapping[str, Any] | None = None
) -> HDF5ScanOptions:
    """Build scan options from the global ``hdf5`` configuration section.

    ``overrides`` is applied last via ``dataclasses.replace``, for features
    (e.g. ``particle_lake.scan``) that need to override a handful of fields
    (``sample_every_nb_time``, ``use_hdf5_cache``, ``checkpoint_every_files``,
    ...) on top of the global defaults without duplicating the whole section.
    """
    hdf5_config = getattr(config, "hdf5", {}) or {}
    if not isinstance(hdf5_config, dict):
        hdf5_config = {}
    file_selection = hdf5_config.get("file_selection", {})
    table_cache = hdf5_config.get("table_cache", {})
    scan = hdf5_config.get("scan", {})
    options = HDF5ScanOptions(
        sample_every_nb_time=file_selection.get("sample_every_nb_time", 1.0),
        wait_age_hour=file_selection.get("wait_age_hour", 24),
        exclude_bad_dirname=file_selection.get("exclude_bad_dirname", True),
        use_hdf5_cache=table_cache.get("use_hdf5_cache", True),
        parallel=scan.get("parallel", False),
        incremental_from_cache_tail=scan.get("incremental_from_cache_tail", True),
        checkpoint_every_files=scan.get("checkpoint_every_files", 100),
        force=force,
    )
    if overrides:
        options = dataclasses_replace(options, **overrides)
    return options


def ttot_matches_sample(ttot: float, sample_every_nb_time: float | None) -> bool:
    """Return whether ``ttot`` lies on the global NB-time sample grid."""
    if sample_every_nb_time is None or sample_every_nb_time <= 0:
        return True
    ratio = float(ttot) / float(sample_every_nb_time)
    return bool(np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-9))


def ttot_sample_mask(
    values: Sequence[float] | np.ndarray | pd.Series | pd.Index,
    sample_every_nb_time: float | None,
) -> np.ndarray:
    """Return a vectorized mask for values lying on the NB-time sample grid."""
    values_array = np.asarray(values, dtype=float)
    if sample_every_nb_time is None or sample_every_nb_time <= 0:
        return np.ones(values_array.shape, dtype=bool)
    ratio = values_array / float(sample_every_nb_time)
    return np.isclose(ratio, np.rint(ratio), rtol=0.0, atol=1e-9)


class HDF5ScanTask(Protocol):
    """Protocol implemented by small analysis reductions over HDF5 files.

    Tasks may optionally define ``prepare_full_rebuild() -> None``. When
    present, the runner calls it right before discarding cached state for a
    forced/options-changed/schema-changed full rebuild. This is not part of
    the Protocol's required signature (existing feather-backed tasks need
    nothing here, since a full overwrite already handles it); it exists for
    cache backends such as ``ParquetDatasetCacheMixin`` that must
    synchronously clear on-disk state broader than one file, e.g. a
    directory of Parquet parts.

    Tasks may optionally define a class attribute ``hdf5_reader_kind``
    (``"processed"`` (default) or ``"raw"``). ``"raw"`` routes the runner's
    per-file table read through ``HDF5FileProcessor.read_raw_tables`` instead
    of ``read_tables`` -- untouched source dtypes, no NS/BH display clipping,
    never reads/writes the L1 feather cache. All tasks sharing one file read
    (one ``HDF5ScanRunner.run()`` call) must declare the same
    ``hdf5_reader_kind``; mixing raises.

    ``write_cache_and_meta``'s ``prune_orphans`` is ``True`` by default (the
    old, unconditional behavior) but the runner calls it with ``False`` for
    every *mid-run* checkpoint (periodic or crash-flush): under
    ``parallel=True``, other workers can still be mid-``write_part`` (creating
    a fresh, not-yet-referenced tmp/part file) while a checkpoint runs in the
    main process, so pruning "orphans" there can delete a live, in-progress
    write out from under its worker. Only the final post-scan call (after
    every worker's result has been consumed, so none can still be writing)
    passes the default ``True``. Cache backends without an orphan-part concept
    (e.g. ``FeatherMetaCacheMixin``, ``ParquetTableCacheMixin``) simply accept
    and ignore the flag.
    """

    name: str
    required_tables: Sequence[str]
    columns_by_table: Mapping[str, Sequence[str] | None]

    def read_cache(self) -> pd.DataFrame: ...

    def read_meta(self) -> Dict[str, Any]: ...

    def is_file_fresh(
        self, hdf5_path: str, meta: Dict[str, Any], cache_df: pd.DataFrame
    ) -> bool: ...

    def process_file(
        self,
        hdf5_path: str,
        df_dict: Dict[str, pd.DataFrame],
        meta: Dict[str, Any],
        cache_df: pd.DataFrame,
    ) -> Dict[str, Any]: ...

    def merge_file_result(
        self, cache_df: pd.DataFrame, hdf5_path: str, result: Dict[str, Any]
    ) -> pd.DataFrame: ...

    def write_cache_and_meta(
        self,
        cache_df: pd.DataFrame,
        processed_files: Dict[str, Dict[str, Any]],
        options: HDF5ScanOptions,
        *,
        prune_orphans: bool = True,
    ) -> None: ...

    def finalize_cache(self, cache_df: pd.DataFrame) -> pd.DataFrame: ...


class HDF5ScanRunner:
    """Run one or more HDF5 scan tasks while sharing file reads."""

    def __init__(self, config_manager: Any, hdf5_file_processor: HDF5FileProcessor | None = None):
        self.config = config_manager
        self.hdf5_file_processor = hdf5_file_processor or HDF5FileProcessor(config_manager)

    def run(
        self,
        simu_name: str,
        tasks: Sequence[HDF5ScanTask],
        options: HDF5ScanOptions,
    ) -> Dict[str, pd.DataFrame]:
        """Run tasks over stale HDF5 files and return each task's cache DataFrame."""
        if not tasks:
            return {}
        task_names = [task.name for task in tasks]
        duplicate_task_names = sorted({name for name in task_names if task_names.count(name) > 1})
        if duplicate_task_names:
            raise ValueError(
                "HDF5 scan tasks in one run must have unique names: "
                + ", ".join(duplicate_task_names)
            )

        states: Dict[str, Dict[str, Any]] = {}
        process_states: Dict[str, Dict[str, Any]] = {}
        stale_tasks_by_file: Dict[str, list[HDF5ScanTask]] = {}
        hdf5_paths = self.hdf5_file_processor.get_all_hdf5_paths(
            simu_name,
            wait_age_hour=options.wait_age_hour,
            sample_every_nb_time=options.sample_every_nb_time,
            exclude_bad_dirname=options.exclude_bad_dirname,
        )

        for task in tasks:
            meta = task.read_meta()
            meta_options = meta.get("scan_options") if isinstance(meta, dict) else None
            meta_options = normalize_persisted_scan_options(meta_options)
            options_changed = bool(meta) and meta_options != persistent_scan_options(options)
            schema_version = getattr(task, "schema_version", None)
            schema_changed = (
                bool(meta)
                and schema_version is not None
                and meta.get("schema_version") != schema_version
            )
            if options.force or options_changed or schema_changed:
                cache_df = pd.DataFrame()
                meta: Dict[str, Any] = {}
                prepare_full_rebuild = getattr(task, "prepare_full_rebuild", None)
                if prepare_full_rebuild is not None:
                    prepare_full_rebuild()
            else:
                cache_df = task.finalize_cache(task.read_cache())
            states[task.name] = {
                "cache_df": cache_df,
                "meta": meta,
                "processed_files": dict(meta.get("processed_files", {})),
            }
            process_states[task.name] = {"cache_df": cache_df, "meta": meta}
            for hdf5_path in self._paths_to_check_for_task(task, hdf5_paths, meta, options):
                if options.force or not task.is_file_fresh(hdf5_path, meta, cache_df):
                    stale_tasks_by_file.setdefault(hdf5_path, []).append(task)

        work_items = [
            (hdf5_path, tuple(stale_tasks_by_file[hdf5_path]))
            for hdf5_path in hdf5_paths
            if hdf5_path in stale_tasks_by_file
        ]

        progress_description = f"{simu_name} HDF5 scan"
        if options.parallel and len(work_items) > 1:
            file_results = self._run_parallel(simu_name, work_items, process_states, options)
        else:
            raw_file_results = (
                self._run_file_tasks(simu_name, hdf5_path, file_tasks, process_states, options)
                for hdf5_path, file_tasks in work_items
            )
            file_results = _iter_with_progress(
                raw_file_results,
                total=len(work_items),
                description=progress_description,
            )

        task_by_name = {task.name: task for task in tasks}
        dirty_task_names: set[str] = set()
        files_since_checkpoint = 0
        checkpoint_every_files = _normalized_checkpoint_every_files(options.checkpoint_every_files)

        try:
            for file_result in file_results:
                hdf5_path = file_result["hdf5_path"]
                for task_name, task_result in file_result["task_results"].items():
                    task = task_by_name[task_name]
                    state = states[task_name]
                    state["cache_df"] = task.merge_file_result(
                        state["cache_df"], hdf5_path, task_result
                    )
                    state["processed_files"][hdf5_path] = task_result["file_meta"]
                    dirty_task_names.add(task_name)
                files_since_checkpoint += 1
                if (
                    checkpoint_every_files is not None
                    and files_since_checkpoint >= checkpoint_every_files
                ):
                    self._write_task_checkpoints(tasks, states, options)
                    dirty_task_names.clear()
                    files_since_checkpoint = 0
        except (Exception, KeyboardInterrupt):
            if dirty_task_names:
                self._write_task_checkpoints(tasks, states, options)
            raise

        output: Dict[str, pd.DataFrame] = {}
        for task in tasks:
            state = states[task.name]
            cache_df = task.finalize_cache(state["cache_df"])
            task.write_cache_and_meta(cache_df, state["processed_files"], options)
            output[task.name] = cache_df
        return output

    def _write_task_checkpoints(
        self,
        tasks: Sequence[HDF5ScanTask],
        states: Mapping[str, Dict[str, Any]],
        options: HDF5ScanOptions,
    ) -> None:
        # prune_orphans=False: other workers may still be mid-write_part() while
        # this mid-run checkpoint runs (see HDF5ScanTask's docstring).
        for task in tasks:
            state = states[task.name]
            cache_df = task.finalize_cache(state["cache_df"])
            task.write_cache_and_meta(
                cache_df, state["processed_files"], options, prune_orphans=False
            )

    def _paths_to_check_for_task(
        self,
        task: HDF5ScanTask,
        hdf5_paths: Sequence[str],
        meta: Mapping[str, Any],
        options: HDF5ScanOptions,
    ) -> Sequence[str]:
        if options.force or not options.incremental_from_cache_tail:
            return hdf5_paths

        processed_files = meta.get("processed_files", {})
        if not processed_files:
            return hdf5_paths

        processed_indices = [
            index for index, hdf5_path in enumerate(hdf5_paths) if hdf5_path in processed_files
        ]
        if not processed_indices:
            return hdf5_paths
        tail_start = max(processed_indices)
        unprocessed_before_tail = [
            hdf5_path for hdf5_path in hdf5_paths[:tail_start] if hdf5_path not in processed_files
        ]
        return [*unprocessed_before_tail, *hdf5_paths[tail_start:]]

    def _run_parallel(
        self,
        simu_name: str,
        work_items: Sequence[tuple[str, Sequence[HDF5ScanTask]]],
        states: Mapping[str, Dict[str, Any]],
        options: HDF5ScanOptions,
    ) -> Iterable[Dict[str, Any]]:
        processes = getattr(self.config, "processes_count", None)
        maxtasksperchild = getattr(self.config, "tasks_per_child", None)
        ctx = multiprocessing.get_context("forkserver")
        args = [
            (self.config, simu_name, hdf5_path, file_tasks, states, options)
            for hdf5_path, file_tasks in work_items
        ]
        with ctx.Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
            yield from _iter_with_progress(
                pool.imap(_run_file_tasks_worker, args),
                total=len(work_items),
                description=f"{simu_name} HDF5 scan",
            )

    def _run_file_tasks(
        self,
        simu_name: str,
        hdf5_path: str,
        file_tasks: Sequence[HDF5ScanTask],
        states: Mapping[str, Dict[str, Any]],
        options: HDF5ScanOptions,
    ) -> Dict[str, Any]:
        required_tables = sorted({table for task in file_tasks for table in task.required_tables})
        columns_by_table = _merge_columns_by_table(file_tasks)
        reader_kind = _shared_hdf5_reader_kind(file_tasks, hdf5_path)

        def _read() -> Dict[str, pd.DataFrame]:
            if reader_kind == "raw":
                return self.hdf5_file_processor.read_raw_tables(
                    hdf5_path,
                    tables=required_tables,
                    columns_by_table=columns_by_table,
                )
            return self.hdf5_file_processor.read_tables(
                hdf5_path,
                simu_name,
                tables=required_tables,
                columns_by_table=columns_by_table,
                use_cache=options.use_hdf5_cache,
            )

        if options.skip_unreadable_files:
            try:
                df_dict = _read()
            except Exception as exc:
                # A real archive spanning years of restarted jobs will contain a few
                # unreadable files (truncated writes, on-disk corruption -- e.g. h5py
                # RuntimeError("Unable to get group info (wrong B-tree signature)")).
                # One bad file must not abort a multi-hour/multi-terabyte scan: treat
                # it as contributing zero rows (every task's empty-tables path
                # already produces a valid, schema-typed empty result) and mark it
                # processed so it isn't retried every run. Exceptions raised by our
                # own processing logic (task.process_file below) are NOT caught here
                # and still fail loudly. Opt-in via HDF5ScanOptions.skip_unreadable_files
                # (only the particle lake enables this by default) -- other features keep
                # the strict fail-fast-and-checkpoint behavior.
                logger.warning(
                    "Skipping unreadable HDF5 file %s: %s: %s",
                    hdf5_path,
                    type(exc).__name__,
                    exc,
                )
                df_dict = {}
        else:
            df_dict = _read()
        df_dict = _filter_df_dict_by_sample(df_dict, options.sample_every_nb_time)
        task_results = {}
        for task in file_tasks:
            state = states[task.name]
            task_results[task.name] = task.process_file(
                hdf5_path,
                df_dict,
                state["meta"],
                state["cache_df"],
            )
        return {"hdf5_path": hdf5_path, "task_results": task_results}


class ScanBackedAnalysisBase:
    """Shared wrapper logic for analysis classes backed by one HDF5 scan job."""

    def __init__(
        self, config_manager: Any, hdf5_file_processor: HDF5FileProcessor | None = None
    ) -> None:
        self.config = config_manager
        self.hdf5_file_processor = hdf5_file_processor or HDF5FileProcessor(config_manager)

    def _run_scan_job(self, job: HDF5ScanJob) -> pd.DataFrame:
        runner = HDF5ScanRunner(self.config, self.hdf5_file_processor)
        return runner.run(job.simu_name, [job.task], job.options)[job.task.name]

    def _load_or_update_scan_job(self, job: HDF5ScanJob, *, update: bool) -> pd.DataFrame:
        if not update:
            return job.task.finalize_cache(job.task.read_cache())
        return self._run_scan_job(job)

    def _scan_options(
        self, *, force: bool = False, overrides: Mapping[str, Any] | None = None
    ) -> HDF5ScanOptions:
        """Return global HDF5 scan options for this analysis."""
        return hdf5_scan_options_from_config(self.config, force=force, overrides=overrides)


def _run_file_tasks_worker(
    args: tuple[Any, str, str, Sequence[HDF5ScanTask], Any, HDF5ScanOptions],
):
    config, simu_name, hdf5_path, file_tasks, states, options = args
    runner = HDF5ScanRunner(config)
    return runner._run_file_tasks(simu_name, hdf5_path, file_tasks, states, options)


def _iter_with_progress(
    iterator: Iterable[Dict[str, Any]],
    *,
    total: int,
    description: str,
) -> Iterable[Dict[str, Any]]:
    if total <= 0:
        yield from iterator
        return

    with Progress() as progress:
        task_id = progress.add_task(description, total=total)
        for result in iterator:
            yield result
            progress.advance(task_id)


def _merge_columns_by_table(
    tasks: Iterable[HDF5ScanTask],
) -> Dict[str, Sequence[str] | None]:
    columns: Dict[str, set[str] | None] = {}
    for task in tasks:
        for table in task.required_tables:
            task_columns = task.columns_by_table.get(table)
            if task_columns is None:
                columns[table] = None
            elif columns.get(table) is not None:
                columns.setdefault(table, set()).update(task_columns)
    return {
        table: None if table_columns is None else sorted(table_columns)
        for table, table_columns in columns.items()
    }


def _shared_hdf5_reader_kind(file_tasks: Sequence[HDF5ScanTask], hdf5_path: str) -> str:
    """Return the one ``hdf5_reader_kind`` shared by every task reading this file."""
    kinds = {getattr(task, "hdf5_reader_kind", "processed") for task in file_tasks}
    if len(kinds) > 1:
        raise ValueError(
            f"HDF5 scan tasks sharing one file read must use the same hdf5_reader_kind, "
            f"got {sorted(kinds)} for {hdf5_path} (tasks: {[task.name for task in file_tasks]})"
        )
    return next(iter(kinds), "processed")


def _filter_df_dict_by_sample(
    df_dict: Dict[str, pd.DataFrame], sample_every_nb_time: float | None
) -> Dict[str, pd.DataFrame]:
    if sample_every_nb_time is None or sample_every_nb_time <= 0:
        return df_dict
    filtered: Dict[str, pd.DataFrame] = {}
    for table_name, df in df_dict.items():
        if df.empty:
            filtered[table_name] = df
            continue
        if "TTOT" in df.columns:
            mask = ttot_sample_mask(df["TTOT"].to_numpy(dtype=float), sample_every_nb_time)
            filtered[table_name] = df if bool(mask.all()) else df.loc[mask].copy()
        elif table_name == "scalars":
            mask = ttot_sample_mask(pd.Index(df.index).to_numpy(dtype=float), sample_every_nb_time)
            filtered[table_name] = df if bool(mask.all()) else df.loc[mask].copy()
        else:
            filtered[table_name] = df
    return filtered


class HDF5ScanSession:
    """Queue HDF5 scan jobs and batch compatible jobs into shared file reads."""

    def __init__(self, config_manager: Any, hdf5_file_processor: HDF5FileProcessor | None = None):
        self.config = config_manager
        self.hdf5_file_processor = hdf5_file_processor or HDF5FileProcessor(config_manager)
        self.jobs: list[HDF5ScanJob] = []

    def add_job(self, job: HDF5ScanJob) -> HDF5ScanJob:
        """Queue one scan job and return it for caller-side bookkeeping."""
        self.jobs.append(job)
        return job

    def add_task(
        self,
        simu_name: str,
        task: HDF5ScanTask,
        options: HDF5ScanOptions | None = None,
    ) -> HDF5ScanJob:
        """Create and queue one scan job."""
        return self.add_job(HDF5ScanJob(simu_name, task, options or HDF5ScanOptions()))

    def run(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Run queued jobs grouped by simulation and options, then clear the queue."""
        groups: dict[tuple[str, HDF5ScanOptions], list[HDF5ScanTask]] = defaultdict(list)
        for job in self.jobs:
            groups[(job.simu_name, job.options)].append(job.task)

        runner = HDF5ScanRunner(self.config, self.hdf5_file_processor)
        output: Dict[str, Dict[str, pd.DataFrame]] = {}
        completed = False
        try:
            for (simu_name, options), tasks in groups.items():
                output.setdefault(simu_name, {}).update(runner.run(simu_name, tasks, options))
            completed = True
            return output
        finally:
            if completed:
                self.jobs.clear()


class FeatherMetaCacheMixin:
    """Small helper for tasks that persist one feather cache and one JSON sidecar."""

    schema_version: int

    @property
    def cache_path(self) -> Path:
        raise NotImplementedError

    @property
    def meta_path(self) -> Path:
        return self.cache_path.with_name(self.cache_path.stem + ".meta.json")

    def read_cache(self) -> pd.DataFrame:
        if not self.cache_path.exists():
            return pd.DataFrame()
        return pd.read_feather(self.cache_path)

    def read_meta(self) -> Dict[str, Any]:
        if not self.meta_path.exists():
            return {}
        try:
            return json.loads(self.meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read scan metadata %s: %r", self.meta_path, exc)
            return {}

    def write_cache_and_meta(
        self,
        cache_df: pd.DataFrame,
        processed_files: Dict[str, Dict[str, Any]],
        options: HDF5ScanOptions,
        *,
        prune_orphans: bool = True,
    ) -> None:
        del prune_orphans  # no orphan-part concept for a single cache file
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_cache_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp_meta_path = self.meta_path.with_suffix(self.meta_path.suffix + ".tmp")
        cache_df.to_feather(tmp_cache_path)
        os.replace(tmp_cache_path, self.cache_path)
        meta = self.build_meta(cache_df, processed_files, options)
        tmp_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        os.replace(tmp_meta_path, self.meta_path)

    def build_meta(
        self,
        cache_df: pd.DataFrame,
        processed_files: Dict[str, Dict[str, Any]],
        options: HDF5ScanOptions,
    ) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scan_options": persistent_scan_options(options),
            "last_ttot": _last_ttot(cache_df),
            "processed_files": processed_files,
        }

    def finalize_cache(self, cache_df: pd.DataFrame) -> pd.DataFrame:
        return cache_df.reset_index(drop=True)


def file_mtime(hdf5_path: str) -> float:
    """Return file mtime as float, or NaN if the file is unavailable."""
    try:
        return float(os.path.getmtime(hdf5_path))
    except OSError:
        return np.nan


_NON_PERSISTED_SCAN_OPTION_KEYS = (
    "force",
    "checkpoint_every_files",
    "parallel",
    "skip_unreadable_files",
)


def persistent_scan_options(options: HDF5ScanOptions) -> Dict[str, Any]:
    """Options persisted in metadata for cache compatibility checks.

    ``parallel`` is excluded: it must never be able to trigger a full
    rebuild just because a pilot run on the login node (``parallel=False``)
    was followed by an sbatch run (``parallel=True``) for the same
    simulation/feature -- that mismatch previously meant
    ``prepare_full_rebuild()`` silently deleting a multi-TB Parquet dataset.
    See docs/analysis_architecture.md Risks.
    """
    values = asdict(options)
    for key in _NON_PERSISTED_SCAN_OPTION_KEYS:
        values.pop(key, None)
    return values


def normalize_persisted_scan_options(meta_options: Any) -> Any:
    """Strip non-persisted keys from an on-disk ``scan_options`` value before comparing.

    Manifests written before ``parallel`` was excluded from
    ``persistent_scan_options`` still have it on disk; without this
    normalization, reading such a manifest would look like an options change
    and force a one-time full rebuild of otherwise-fresh Parquet output.
    """
    if not isinstance(meta_options, dict):
        return meta_options
    return {k: v for k, v in meta_options.items() if k not in _NON_PERSISTED_SCAN_OPTION_KEYS}


def _normalized_checkpoint_every_files(value: int | None) -> int | None:
    if value is None:
        return None
    if value <= 0:
        return None
    return int(value)


def file_times_from_scalars(df_dict: Mapping[str, pd.DataFrame]) -> list[float]:
    """Return sorted TTOT values from a scan df_dict."""
    scalars = df_dict.get("scalars", pd.DataFrame())
    if "TTOT" in scalars.columns:
        values = scalars["TTOT"]
    else:
        values = pd.Series(scalars.index)
    return sorted(float(t) for t in values.dropna().unique())


def default_file_meta(hdf5_path: str, df_dict: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    """Metadata shared by scan tasks for one processed HDF5 file."""
    return {"mtime": file_mtime(hdf5_path), "ttot": file_times_from_scalars(df_dict)}


def file_is_fresh(
    hdf5_path: str,
    meta: Mapping[str, Any],
    cached_times: set[float] | None = None,
) -> bool:
    """Return whether metadata says this HDF5 file is fresh in the cache."""
    file_meta = meta.get("processed_files", {}).get(hdf5_path)
    if not file_meta:
        return False
    current_mtime = file_mtime(hdf5_path)
    if np.isnan(current_mtime):
        return False
    if not np.isclose(float(file_meta.get("mtime", np.nan)), current_mtime, rtol=0.0, atol=1e-9):
        return False
    if cached_times is None:
        return True
    return set(float(t) for t in file_meta.get("ttot", [])).issubset(cached_times)


def replace_ttot_rows(
    cache_df: pd.DataFrame,
    new_df: pd.DataFrame,
    ttot_column: str,
) -> pd.DataFrame:
    """Replace cached rows matching new TTOT values, then append new rows."""
    if new_df.empty:
        return cache_df
    if not cache_df.empty and ttot_column in cache_df.columns and ttot_column in new_df.columns:
        cache_df = cache_df[~cache_df[ttot_column].astype(float).isin(new_df[ttot_column])]
    return pd.concat([cache_df, new_df], ignore_index=True, sort=False)


def _last_ttot(cache_df: pd.DataFrame) -> float | None:
    for column in ("TTOT", "Time[NB]"):
        if column in cache_df.columns and not cache_df.empty:
            return float(cache_df[column].max())
    return None
