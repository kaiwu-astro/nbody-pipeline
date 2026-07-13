"""
Parquet cache mixins for HDF5ScanTask implementations.

These mirror FeatherMetaCacheMixin's role in hdf5_scan.py but persist
Parquet (validated against a nbody_pipeline.schemas.TableSchema) instead
of feather, for tasks whose output is meant to live in the Parquet/DuckDB
analysis layer. See docs/analysis_architecture.md for the caching-layer
design this implements.

Two shapes are provided:

- ``ParquetDatasetCacheMixin`` for object_rows/events tasks, where the full
  table would be too large (or too awkward) to hold in memory across a scan.
  Each source HDF5 file gets its own Parquet "part" file, written directly
  in ``merge_file_result`` (pass-through: the in-memory ``cache_df`` threaded
  through the runner stays a schema-typed empty sentinel for the whole run).
- ``ParquetTableCacheMixin`` for snapshot_scalar tasks, which mirrors
  FeatherMetaCacheMixin's classic in-memory-merge model (one DataFrame held
  in memory, merged via the task's own ``merge_file_result``, written as one
  file) but as a single validated Parquet file.

Both share one manifest.json format keyed by ``schema_hash`` (a hash over
column name/dtype pairs from the table's schema YAML): if a developer edits
a table's columns without bumping ``schema_version``, the stale manifest is
silently treated as empty on read, forcing every source file to be
reprocessed and its part/table rewritten under the new schema.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from nbody_pipeline.schemas import TableSchema

from .hdf5_scan import HDF5ScanOptions, file_is_fresh, persistent_scan_options

logger = logging.getLogger(__name__)


def _part_filename(hdf5_path: str) -> str:
    """Deterministic per-source-file Parquet part name."""
    stem = Path(hdf5_path).stem
    digest = hashlib.sha1(os.path.abspath(hdf5_path).encode("utf-8")).hexdigest()[:8]
    return f"part-{stem}-{digest}.parquet"


def _replace_with_retry(src: Path, dst: Path, attempts: int = 5, base_delay: float = 0.5) -> None:
    """``os.replace`` with retry on ``FileNotFoundError``.

    During the real full-archive lake build, ``os.replace(tmp_path, part_path)``
    occasionally raised ``FileNotFoundError`` for ``tmp_path`` immediately after
    this same process had just written it -- reproduced only under the real
    ~32-way-concurrent-writes-into-one-directory production load on this shared
    filesystem, never in an isolated synthetic stress test with the same
    concurrency (ruling out a same-name race; ``src`` is already
    per-call-unique). Likely a transient directory-entry visibility hiccup on
    the underlying network filesystem under heavy concurrent metadata load.
    Retrying (the source file's *content* was already fully written and
    fsync'd by ``to_parquet`` before this call) is safe and cheap relative to
    losing an entire multi-hour scan.
    """
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except FileNotFoundError:
            if attempt == attempts - 1:
                raise
            logger.warning(
                "os.replace(%s, %s) found no such tmp file (attempt %d/%d), retrying",
                src,
                dst,
                attempt + 1,
                attempts,
            )
            time.sleep(base_delay * (2**attempt))


class ParquetDatasetCacheMixin:
    """Cache mixin for object_rows/events tasks: one Parquet part per source file."""

    schema_version: int

    @property
    def table_schema(self) -> TableSchema:
        raise NotImplementedError

    @property
    def cache_dir(self) -> Path:
        raise NotImplementedError

    @property
    def data_dir(self) -> Path:
        return self.cache_dir / "data"

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    def read_cache(self) -> pd.DataFrame:
        """Return a schema-typed empty sentinel; rows live in Parquet parts, not memory."""
        return self.table_schema.empty_dataframe()

    def read_meta(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read manifest %s: %r", self.manifest_path, exc)
            return {}
        if manifest.get("schema_hash") != self.table_schema.schema_hash():
            return {}
        return manifest

    def is_file_fresh(self, hdf5_path: str, meta: Dict[str, Any], cache_df: pd.DataFrame) -> bool:
        if not file_is_fresh(hdf5_path, meta):
            return False
        file_meta = meta.get("processed_files", {}).get(hdf5_path, {})
        part_name = file_meta.get("part")
        if not part_name:
            return False
        return (self.data_dir / part_name).exists()

    @property
    def parquet_write_options(self) -> Dict[str, Any]:
        """Extra kwargs passed to ``DataFrame.to_parquet`` when writing a part.

        Override per-task, e.g. for ``compression="zstd"`` /
        ``use_byte_stream_split`` / ``row_group_size`` on the large lake
        tables. Empty by default, matching the previous unconditional
        ``to_parquet(tmp_path, index=False)`` call.
        """
        return {}

    def write_part(self, hdf5_path: str, rows: pd.DataFrame) -> Dict[str, Any]:
        """Validate, atomically write ``rows`` as this file's Parquet part, return its file_meta.

        Safe to call directly from inside a worker process's ``process_file``
        (the escape hatch documented in docs/analysis_architecture.md Risks
        #2): part paths are unique per source file
        (``_part_filename(hdf5_path)``), so concurrent workers writing
        different files' parts never collide. A task whose per-file row count
        is too large to pickle back to the main process economically should
        call this itself and return the ``{"part", "row_count", "ttot_min",
        "ttot_max"}`` shape from ``process_file`` (see ``merge_file_result``'s
        ``"part"`` branch) instead of returning the full ``rows`` DataFrame.
        """
        self.table_schema.validate_dataframe(rows)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        part_name = _part_filename(hdf5_path)
        part_path = self.data_dir / part_name
        # Per-call-unique tmp name (pid + random suffix): rules out any two calls
        # ever targeting the same tmp path, though this alone did not fully explain
        # the flake below (a synthetic 32-way concurrent stress test with unique
        # names never reproduced it).
        tmp_path = part_path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        rows.to_parquet(tmp_path, index=False, **self.parquet_write_options)
        _replace_with_retry(tmp_path, part_path)

        ttot_min = float(rows["ttot"].min()) if len(rows) and "ttot" in rows.columns else None
        ttot_max = float(rows["ttot"].max()) if len(rows) and "ttot" in rows.columns else None
        return {
            "part": part_name,
            "row_count": int(len(rows)),
            "ttot_min": ttot_min,
            "ttot_max": ttot_max,
        }

    def merge_file_result(
        self, cache_df: pd.DataFrame, hdf5_path: str, result: Dict[str, Any]
    ) -> pd.DataFrame:
        """Fold this file's part metadata into ``file_meta``; cache_df passes through unchanged.

        Two shapes are accepted, dispatched on which key ``result`` has:

        - ``"rows"``: the in-process shape (e.g. ``CompactObjectHistoryTask``)
          -- ``process_file`` returned the full DataFrame, and this method
          calls ``write_part`` on it here (in the main process).
        - ``"part"``: the worker-direct-write shape -- ``process_file`` already
          called ``write_part`` itself (typically inside a worker process) and
          returned only the resulting part metadata; there is nothing left to
          write here.
        """
        if "part" in result:
            part_meta = {
                "part": result["part"],
                "row_count": result.get("row_count"),
                "ttot_min": result.get("ttot_min"),
                "ttot_max": result.get("ttot_max"),
            }
        else:
            part_meta = self.write_part(hdf5_path, result["rows"])
        result["file_meta"] = {**result.get("file_meta", {}), **part_meta}
        return cache_df

    def write_cache_and_meta(
        self,
        cache_df: pd.DataFrame,
        processed_files: Dict[str, Dict[str, Any]],
        options: HDF5ScanOptions,
        *,
        prune_orphans: bool = True,
    ) -> None:
        # prune_orphans=False during mid-run checkpoints: other workers can still be
        # mid-write_part() (a fresh tmp/part file not yet in processed_files is not
        # necessarily an orphan -- see HDF5ScanTask's docstring). Confirmed by a real
        # crash: a live worker's brand-new tmp file got deleted out from under it by
        # a concurrent checkpoint's unconditional prune, later manifesting as
        # os.replace() raising FileNotFoundError for a file this same process had
        # just finished writing.
        if prune_orphans:
            self._prune_orphan_parts(processed_files)
        manifest = {
            "schema_version": self.schema_version,
            "schema_hash": self.table_schema.schema_hash(),
            "table_name": self.table_schema.table,
            "scan_options": persistent_scan_options(options),
            "processed_files": processed_files,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_manifest_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        tmp_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(tmp_manifest_path, self.manifest_path)

    def _prune_orphan_parts(self, processed_files: Dict[str, Dict[str, Any]]) -> None:
        if not self.data_dir.exists():
            return
        referenced = {
            file_meta["part"] for file_meta in processed_files.values() if file_meta.get("part")
        }
        for entry in self.data_dir.iterdir():
            if entry.suffix == ".tmp":
                entry.unlink()
            elif entry.suffix == ".parquet" and entry.name not in referenced:
                entry.unlink()

    def finalize_cache(self, cache_df: pd.DataFrame) -> pd.DataFrame:
        return cache_df

    def prepare_full_rebuild(self) -> None:
        """Clear all on-disk state so no reader can see a mix of old/new schema parts."""
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        if self.manifest_path.exists():
            self.manifest_path.unlink()


class ParquetTableCacheMixin:
    """Cache mixin for snapshot_scalar tasks: one merged, schema-validated Parquet table."""

    schema_version: int

    @property
    def table_schema(self) -> TableSchema:
        raise NotImplementedError

    @property
    def cache_path(self) -> Path:
        raise NotImplementedError

    @property
    def manifest_path(self) -> Path:
        return self.cache_path.with_name(self.cache_path.stem + ".manifest.json")

    def read_cache(self) -> pd.DataFrame:
        if not self.cache_path.exists():
            return self.table_schema.empty_dataframe()
        return pd.read_parquet(self.cache_path)

    def read_meta(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        try:
            manifest = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read manifest %s: %r", self.manifest_path, exc)
            return {}
        if manifest.get("schema_hash") != self.table_schema.schema_hash():
            return {}
        return manifest

    def write_cache_and_meta(
        self,
        cache_df: pd.DataFrame,
        processed_files: Dict[str, Dict[str, Any]],
        options: HDF5ScanOptions,
        *,
        prune_orphans: bool = True,
    ) -> None:
        del prune_orphans  # no orphan-part concept: one merged file, no per-file parts
        self.table_schema.validate_dataframe(cache_df)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_cache_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        cache_df.to_parquet(tmp_cache_path, index=False)
        os.replace(tmp_cache_path, self.cache_path)

        manifest = {
            "schema_version": self.schema_version,
            "schema_hash": self.table_schema.schema_hash(),
            "table_name": self.table_schema.table,
            "scan_options": persistent_scan_options(options),
            "processed_files": processed_files,
        }
        tmp_manifest_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        tmp_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(tmp_manifest_path, self.manifest_path)

    def finalize_cache(self, cache_df: pd.DataFrame) -> pd.DataFrame:
        return cache_df.reset_index(drop=True)

    def prepare_full_rebuild(self) -> None:
        if self.cache_path.exists():
            self.cache_path.unlink()
        if self.manifest_path.exists():
            self.manifest_path.unlink()
