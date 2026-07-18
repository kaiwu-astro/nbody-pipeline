"""Full snapshot particle lake: singles/binaries/mergers/scalars (VO-safe, Parquet-backed).

Unlike the pilot ``compact_object_history``/``snapshot_summary`` tables (a
handful of derived rows), this reduces every raw HDF5 column worth keeping
for singles/binaries/mergers/scalars into four VO-safe schema-registered
tables, one row per object per snapshot (or one row per snapshot for
scalars). See docs/analysis_architecture.md Roadmap #5.

All four tasks read via ``hdf5_reader_kind = "source"``
(``HDF5FileProcessor.read_raw_tables`` with ``simu_name=None``): untouched
source dtypes, no NS/BH display clipping, and -- critically -- never reads
back from the lake it is building (unlike ``"raw"``, which is lake-first with
an HDF5 fallback; see ``nbody_pipeline.analysis.hdf5_scan.HDF5ScanTask``'s
docstring for the three-way ``hdf5_reader_kind`` split). The three per-object
tasks write their Parquet part directly inside
``process_file`` via ``ParquetDatasetCacheMixin.write_part`` (the worker
"direct write" escape hatch documented in docs/analysis_architecture.md Risks
#2) instead of returning the full DataFrame to the main process, since a
single source file's singles table alone can be several GB.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from nbody_pipeline.analysis.cache_paths import (
    SNAPSHOT_BINARIES_FEATURE,
    SNAPSHOT_MERGERS_FEATURE,
    SNAPSHOT_SCALARS_FEATURE,
    SNAPSHOT_SINGLES_FEATURE,
    analysis_cache_dir,
)
from nbody_pipeline.analysis.hdf5_scan import (
    HDF5ScanJob,
    HDF5ScanOptions,
    HDF5ScanRunner,
    ScanBackedAnalysisBase,
    default_file_meta,
    file_is_fresh,
    replace_ttot_rows,
)
from nbody_pipeline.analysis.parquet_cache import ParquetDatasetCacheMixin, ParquetTableCacheMixin
from nbody_pipeline.io.text_parsers import get_scale_dict_from_hdf5_df, read_step_times
from nbody_pipeline.schemas import TableSchema, load_table_schema

# zstd + BYTE_STREAM_SPLIT (float-column-friendly bit-splitting, pyarrow >= 8)
# + ~1M-row row groups for DuckDB row-group pruning on (ttot, id) predicates.
LAKE_PARQUET_WRITE_OPTIONS: Dict[str, Any] = {
    "compression": "zstd",
    "use_byte_stream_split": True,
    "row_group_size": 1_000_000,
}

# Sentinel for snapshot_binaries.bin_label when the source HDF5 file predates
# the "176 Bin Label"/"176 Bin cm Name" dataset (see SnapshotBinariesTask._build_rows).
_BIN_LABEL_UNKNOWN = -9

_SCALE_SCALAR_COLUMNS = [
    "TTOT",
    "RBAR",
    "ZMBAR",
    "VSTAR",
    "TSCALE",
    "RDENS(1)",
    "RDENS(2)",
    "RDENS(3)",
]


def _as_int64(values: Any) -> np.ndarray:
    """Round-trip integer-valued columns (int32 IDs, or float32 counts) to int64."""
    return np.rint(np.asarray(values, dtype="float64")).astype("int64")


def _as_int32(values: Any) -> np.ndarray:
    return np.rint(np.asarray(values, dtype="float64")).astype("int32")


def _as_float32(values: Any) -> np.ndarray:
    return np.asarray(values, dtype="float32")


def _scale_and_rdens_pc(scalars: pd.DataFrame) -> tuple[Dict[str, float], pd.DataFrame]:
    """Scale-factor dict and a (TTOT -> RDENS*RBAR in pc) frame for this file's steps.

    Mirrors nbody_pipeline.io.hdf5_reader.HDF5FileProcessor.read_file: RBAR/VSTAR/
    ZMBAR/TSCALE are simulation-wide constants (read from the first step), while
    RDENS moves every snapshot and must be looked up per TTOT.
    """
    scale = get_scale_dict_from_hdf5_df(scalars)
    rdens_nb = scalars.set_index("TTOT", drop=False)[["RDENS(1)", "RDENS(2)", "RDENS(3)"]]
    rdens_pc = rdens_nb * scale["r"]
    return scale, rdens_pc


def _rdens_corrected_pc(
    raw_values: Any, ttot: np.ndarray, rdens_pc_column: pd.Series
) -> np.ndarray:
    """Density-center-corrected position in pc, float32 (matches X [pc]/Bin cm X [pc])."""
    offsets = pd.Series(ttot).map(rdens_pc_column).to_numpy(dtype="float64")
    return (np.asarray(raw_values, dtype="float64") - offsets).astype("float32")


def _distance_pc(x_pc: np.ndarray, y_pc: np.ndarray, z_pc: np.ndarray) -> np.ndarray:
    return np.sqrt(
        x_pc.astype("float64") ** 2 + y_pc.astype("float64") ** 2 + z_pc.astype("float64") ** 2
    ).astype("float32")


def _drop_excluded_ttot(
    df: pd.DataFrame, hdf5_path: str, excluded_ttot_by_path: Mapping[str, set[float]]
) -> pd.DataFrame:
    """Drop rows whose TTOT lost the cross-file dedup tie-break for this source file."""
    excluded = excluded_ttot_by_path.get(hdf5_path)
    if not excluded or df.empty or "TTOT" not in df.columns:
        return df
    return df.loc[~df["TTOT"].isin(excluded)]


def _file_list_hash(hdf5_paths: Sequence[str]) -> str:
    payload = "\n".join(sorted(hdf5_paths))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def compute_ttot_dedup_exclusions(
    hdf5_paths: Sequence[str], read_step_times_fn: Any = read_step_times
) -> Dict[str, set[float]]:
    """Resolve cross-file TTOT duplicates: restart-boundary checkpoints that two
    different run directories both wrote (see scripts/lake_preflight.py's
    directory_range_overlap/duplicate_filename_time candidates -- this is the
    write-time fix for those).

    For every TTOT written by more than one file, the file with the latest
    mtime (ties broken by path) is authoritative. Returns ``{path: {ttot to
    drop from that file}}`` for every other contributing file; a file that
    never loses a TTOT to another file has no entry. ``read_step_times_fn``
    reads only Step# group attrs, never a full dataset -- cheap relative to
    the actual particle-data read that follows in ``process_file``. Defaults
    to the free function but callers normally pass
    ``hdf5_file_processor.read_step_times`` so tests can substitute a fake.
    """
    winner_of_ttot: Dict[float, tuple[str, float]] = {}
    contributors_of_ttot: Dict[float, set[str]] = defaultdict(set)
    for path in hdf5_paths:
        mtime = os.path.getmtime(path)
        for ttot in read_step_times_fn(path):
            contributors_of_ttot[ttot].add(path)
            current = winner_of_ttot.get(ttot)
            if current is None or (mtime, path) > (current[1], current[0]):
                winner_of_ttot[ttot] = (path, mtime)

    excluded_by_path: Dict[str, set[float]] = defaultdict(set)
    for ttot, paths in contributors_of_ttot.items():
        if len(paths) < 2:
            continue
        winner_path = winner_of_ttot[ttot][0]
        for path in paths:
            if path != winner_path:
                excluded_by_path[path].add(ttot)
    return dict(excluded_by_path)


class ParticleLakeProcessor(ScanBackedAnalysisBase):
    """Build and cache the full particle lake (4 tables) for one simulation."""

    def build_scan_jobs(self, simu_name: str, *, force: bool = False) -> List[HDF5ScanJob]:
        """One job per lake table, all sharing one HDF5ScanOptions.

        Same options means ``HDF5ScanSession``/``HDF5ScanRunner`` batch all
        four tasks into a single pass over this simulation's HDF5 files.
        """
        options = self._lake_scan_options(force=force)
        hdf5_paths = self.hdf5_file_processor.get_all_hdf5_paths(
            simu_name,
            wait_age_hour=options.wait_age_hour,
            sample_every_nb_time=options.sample_every_nb_time,
            exclude_bad_dirname=options.exclude_bad_dirname,
        )
        excluded_ttot_by_path = self._load_or_build_ttot_dedup_map(
            simu_name, hdf5_paths, force=force
        )
        tasks = [
            SnapshotSinglesTask(self.config, simu_name, excluded_ttot_by_path),
            SnapshotBinariesTask(self.config, simu_name, excluded_ttot_by_path),
            SnapshotMergersTask(self.config, simu_name, excluded_ttot_by_path),
            SnapshotScalarsTask(self.config, simu_name, excluded_ttot_by_path),
        ]
        return [HDF5ScanJob(simu_name, task, options) for task in tasks]

    def _lake_scan_options(self, *, force: bool = False) -> HDF5ScanOptions:
        lake_config = getattr(self.config, "particle_lake", {}) or {}
        overrides = dict(lake_config.get("scan", {})) if isinstance(lake_config, dict) else {}
        # A full-archive lake build spans years of restarted jobs and will hit the
        # occasional corrupted/truncated file (see docs/analysis_architecture.md
        # Roadmap #5) -- default to skipping those rather than aborting a
        # multi-hour scan; config can still force strict fail-fast via
        # particle_lake.scan.skip_unreadable_files: false.
        overrides.setdefault("skip_unreadable_files", True)
        return self._scan_options(force=force, overrides=overrides)

    def _ttot_dedup_cache_path(self, simu_name: str) -> Path:
        # A simulation-level sidecar (not per-feature): the same dedup map applies to
        # all three object_rows lake tables built from this simulation's file list.
        return analysis_cache_dir(self.config, simu_name, SNAPSHOT_SINGLES_FEATURE).parent / (
            "ttot_dedup_map.json"
        )

    def _load_or_build_ttot_dedup_map(
        self, simu_name: str, hdf5_paths: Sequence[str], *, force: bool
    ) -> Dict[str, set[float]]:
        """Cache compute_ttot_dedup_exclusions on disk, keyed by the exact file list.

        Recomputing requires reopening every file for a cheap attrs-only read, but
        that is still real wall-clock time at full-simulation scale (~10^4 files);
        skip it whenever the file list is unchanged from the last run.
        """
        cache_path = self._ttot_dedup_cache_path(simu_name)
        file_hash = _file_list_hash(hdf5_paths)
        if not force and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
            except (OSError, json.JSONDecodeError):
                cached = None
            if cached is not None and cached.get("file_list_hash") == file_hash:
                return {
                    path: set(ttots)
                    for path, ttots in cached.get("excluded_ttot_by_path", {}).items()
                }

        excluded_ttot_by_path = compute_ttot_dedup_exclusions(
            hdf5_paths, self.hdf5_file_processor.read_step_times
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "file_list_hash": file_hash,
                    "excluded_ttot_by_path": {
                        path: sorted(ttots) for path, ttots in excluded_ttot_by_path.items()
                    },
                },
                indent=2,
            )
        )
        os.replace(tmp_path, cache_path)
        return excluded_ttot_by_path

    def update(self, simu_name: str, *, force: bool = False) -> Dict[str, pd.DataFrame]:
        """Run all four lake tasks for one simulation, sharing one file read per source file."""
        jobs = self.build_scan_jobs(simu_name, force=force)
        runner = HDF5ScanRunner(self.config, self.hdf5_file_processor)
        return runner.run(simu_name, [job.task for job in jobs], jobs[0].options)


class SnapshotSinglesTask(ParquetDatasetCacheMixin):
    """Raw-value per-snapshot rows for every single star."""

    schema_version = 1
    name = "snapshot_singles"
    hdf5_reader_kind = "source"
    required_tables: Sequence[str] = ("scalars", "singles")
    columns_by_table: Mapping[str, Sequence[str] | None] = {
        "scalars": list(_SCALE_SCALAR_COLUMNS),
        "singles": [
            "Name",
            "KW",
            "Type",
            "M",
            "X1",
            "X2",
            "X3",
            "V1",
            "V2",
            "V3",
            "POT",
            "R*",
            "L*",
            "Teff*",
            "RC*",
            "MC*",
            "ASPN",
            "EPOCH",
        ],
    }

    def __init__(
        self,
        config_manager: Any,
        simu_name: str,
        excluded_ttot_by_path: Mapping[str, set[float]] | None = None,
    ) -> None:
        self.config = config_manager
        self.simu_name = simu_name
        self.excluded_ttot_by_path = excluded_ttot_by_path or {}

    @property
    def table_schema(self) -> TableSchema:
        return load_table_schema("snapshot_singles")

    @property
    def cache_dir(self) -> Path:
        return analysis_cache_dir(self.config, self.simu_name, SNAPSHOT_SINGLES_FEATURE)

    @property
    def parquet_write_options(self) -> Dict[str, Any]:
        return LAKE_PARQUET_WRITE_OPTIONS

    def process_file(
        self,
        hdf5_path: str,
        df_dict: Dict[str, pd.DataFrame],
        meta: Dict[str, Any],
        cache_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        rows = self._build_rows(hdf5_path, df_dict)
        part_meta = self.write_part(hdf5_path, rows)
        return {**part_meta, "file_meta": default_file_meta(hdf5_path, df_dict)}

    def _build_rows(self, hdf5_path: str, df_dict: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        scalars = df_dict.get("scalars", pd.DataFrame())
        singles = df_dict.get("singles", pd.DataFrame())
        singles = _drop_excluded_ttot(singles, hdf5_path, self.excluded_ttot_by_path)
        if scalars.empty or singles.empty:
            return self.table_schema.empty_dataframe()

        scale, rdens_pc = _scale_and_rdens_pc(scalars)
        ttot = singles["TTOT"].to_numpy(dtype="float64")
        x_pc = _rdens_corrected_pc(singles["X1"], ttot, rdens_pc["RDENS(1)"])
        y_pc = _rdens_corrected_pc(singles["X2"], ttot, rdens_pc["RDENS(2)"])
        z_pc = _rdens_corrected_pc(singles["X3"], ttot, rdens_pc["RDENS(3)"])

        rows = pd.DataFrame(
            {
                "simulation_id": self.simu_name,
                "ttot": ttot,
                "time_myr": ttot * scale["t"],
                "object_id": _as_int64(singles["Name"]),
                "kw": _as_int32(singles["KW"]),
                "type_code": _as_int32(singles["Type"]),
                "mass_msun": _as_float32(singles["M"]),
                "x_pc": x_pc,
                "y_pc": y_pc,
                "z_pc": z_pc,
                "vx_kms": _as_float32(singles["V1"]),
                "vy_kms": _as_float32(singles["V2"]),
                "vz_kms": _as_float32(singles["V3"]),
                "r_pc": _distance_pc(x_pc, y_pc, z_pc),
                "pot_nb": _as_float32(singles["POT"]),
                "radius_rsun": _as_float32(singles["R*"]),
                "luminosity_lsun": _as_float32(singles["L*"]),
                "teff_k": _as_float32(singles["Teff*"]),
                "core_radius_rsun": _as_float32(singles["RC*"]),
                "core_mass_msun": _as_float32(singles["MC*"]),
                "spin_aspn": _as_float32(singles["ASPN"]),
                "epoch_myr": _as_float32(singles["EPOCH"]),
                "source_hdf5_path": hdf5_path,
            }
        )
        rows = rows.sort_values(["ttot", "object_id"], kind="stable").reset_index(drop=True)
        return rows[list(self.table_schema.column_names())]


class SnapshotBinariesTask(ParquetDatasetCacheMixin):
    """Raw-value per-snapshot rows for every binary."""

    schema_version = 1
    name = "snapshot_binaries"
    hdf5_reader_kind = "source"
    required_tables: Sequence[str] = ("scalars", "binaries")
    columns_by_table: Mapping[str, Sequence[str] | None] = {
        "scalars": list(_SCALE_SCALAR_COLUMNS),
        "binaries": [
            "Bin Name1",
            "Bin Name2",
            "Bin cm Name",
            "Bin KW1",
            "Bin KW2",
            "Bin cm KW",
            "Bin Label",
            "Bin M1*",
            "Bin M2*",
            "Bin cm X1",
            "Bin cm X2",
            "Bin cm X3",
            "Bin cm V1",
            "Bin cm V2",
            "Bin cm V3",
            "Bin rel X1",
            "Bin rel X2",
            "Bin rel X3",
            "Bin rel V1",
            "Bin rel V2",
            "Bin rel V3",
            "Bin POT",
            "Bin A[au]",
            "Bin ECC",
            "Bin P[d]",
            "Bin G",
            "Bin RS1*",
            "Bin RS2*",
            "Bin L1*",
            "Bin L2*",
            "Bin Teff1*",
            "Bin Teff2*",
            "Bin RC1*",
            "Bin RC2*",
            "Bin MC1*",
            "Bin MC2*",
            "ASPN1",
            "ASPN2",
            "EPOCH1",
            "EPOCH2",
        ],
    }

    def __init__(
        self,
        config_manager: Any,
        simu_name: str,
        excluded_ttot_by_path: Mapping[str, set[float]] | None = None,
    ) -> None:
        self.config = config_manager
        self.simu_name = simu_name
        self.excluded_ttot_by_path = excluded_ttot_by_path or {}

    @property
    def table_schema(self) -> TableSchema:
        return load_table_schema("snapshot_binaries")

    @property
    def cache_dir(self) -> Path:
        return analysis_cache_dir(self.config, self.simu_name, SNAPSHOT_BINARIES_FEATURE)

    @property
    def parquet_write_options(self) -> Dict[str, Any]:
        return LAKE_PARQUET_WRITE_OPTIONS

    def process_file(
        self,
        hdf5_path: str,
        df_dict: Dict[str, pd.DataFrame],
        meta: Dict[str, Any],
        cache_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        rows = self._build_rows(hdf5_path, df_dict)
        part_meta = self.write_part(hdf5_path, rows)
        return {**part_meta, "file_meta": default_file_meta(hdf5_path, df_dict)}

    def _build_rows(self, hdf5_path: str, df_dict: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        scalars = df_dict.get("scalars", pd.DataFrame())
        binaries = df_dict.get("binaries", pd.DataFrame())
        binaries = _drop_excluded_ttot(binaries, hdf5_path, self.excluded_ttot_by_path)
        if scalars.empty or binaries.empty:
            return self.table_schema.empty_dataframe()

        scale, rdens_pc = _scale_and_rdens_pc(scalars)
        ttot = binaries["TTOT"].to_numpy(dtype="float64")
        cm_x_pc = _rdens_corrected_pc(binaries["Bin cm X1"], ttot, rdens_pc["RDENS(1)"])
        cm_y_pc = _rdens_corrected_pc(binaries["Bin cm X2"], ttot, rdens_pc["RDENS(2)"])
        cm_z_pc = _rdens_corrected_pc(binaries["Bin cm X3"], ttot, rdens_pc["RDENS(3)"])
        # Some archived files predate the "176 Bin Label"/"176 Bin cm Name" dataset
        # (confirmed against real data: old_run_archive/snap.40/*.h5part has every
        # other Bin* column but neither Bin Label name) -- fall back to the
        # documented "unknown" sentinel instead of KeyError-ing the whole file.
        if "Bin Label" in binaries.columns:
            bin_label = _as_int32(binaries["Bin Label"])
        else:
            bin_label = np.full(len(binaries), _BIN_LABEL_UNKNOWN, dtype="int32")

        rows = pd.DataFrame(
            {
                "simulation_id": self.simu_name,
                "ttot": ttot,
                "time_myr": ttot * scale["t"],
                "object_id_1": _as_int64(binaries["Bin Name1"]),
                "object_id_2": _as_int64(binaries["Bin Name2"]),
                "cm_id": _as_int64(binaries["Bin cm Name"]),
                "kw_1": _as_int32(binaries["Bin KW1"]),
                "kw_2": _as_int32(binaries["Bin KW2"]),
                "cm_kw": _as_int32(binaries["Bin cm KW"]),
                "bin_label": bin_label,
                "mass_1_msun": _as_float32(binaries["Bin M1*"]),
                "mass_2_msun": _as_float32(binaries["Bin M2*"]),
                "cm_x_pc": cm_x_pc,
                "cm_y_pc": cm_y_pc,
                "cm_z_pc": cm_z_pc,
                "cm_vx_kms": _as_float32(binaries["Bin cm V1"]),
                "cm_vy_kms": _as_float32(binaries["Bin cm V2"]),
                "cm_vz_kms": _as_float32(binaries["Bin cm V3"]),
                "r_pc": _distance_pc(cm_x_pc, cm_y_pc, cm_z_pc),
                "rel_x_pc": _as_float32(binaries["Bin rel X1"]),
                "rel_y_pc": _as_float32(binaries["Bin rel X2"]),
                "rel_z_pc": _as_float32(binaries["Bin rel X3"]),
                "rel_vx_kms": _as_float32(binaries["Bin rel V1"]),
                "rel_vy_kms": _as_float32(binaries["Bin rel V2"]),
                "rel_vz_kms": _as_float32(binaries["Bin rel V3"]),
                "cm_pot_nb": _as_float32(binaries["Bin POT"]),
                "semi_major_axis_au": _as_float32(binaries["Bin A[au]"]),
                "eccentricity": _as_float32(binaries["Bin ECC"]),
                "period_days": _as_float32(binaries["Bin P[d]"]),
                "pert_gamma": _as_float32(binaries["Bin G"]),
                "radius_1_rsun": _as_float32(binaries["Bin RS1*"]),
                "radius_2_rsun": _as_float32(binaries["Bin RS2*"]),
                "luminosity_1_lsun": _as_float32(binaries["Bin L1*"]),
                "luminosity_2_lsun": _as_float32(binaries["Bin L2*"]),
                "teff_1_k": _as_float32(binaries["Bin Teff1*"]),
                "teff_2_k": _as_float32(binaries["Bin Teff2*"]),
                "core_radius_1_rsun": _as_float32(binaries["Bin RC1*"]),
                "core_radius_2_rsun": _as_float32(binaries["Bin RC2*"]),
                "core_mass_1_msun": _as_float32(binaries["Bin MC1*"]),
                "core_mass_2_msun": _as_float32(binaries["Bin MC2*"]),
                "spin_aspn_1": _as_float32(binaries["ASPN1"]),
                "spin_aspn_2": _as_float32(binaries["ASPN2"]),
                "epoch_1_myr": _as_float32(binaries["EPOCH1"]),
                "epoch_2_myr": _as_float32(binaries["EPOCH2"]),
                "source_hdf5_path": hdf5_path,
            }
        )
        # cm_id is not a reliable per-snapshot unique key (see its schema description);
        # sort by the confirmed-unique (object_id_1, object_id_2) pair instead.
        rows = rows.sort_values(["ttot", "object_id_1", "object_id_2"], kind="stable").reset_index(
            drop=True
        )
        return rows[list(self.table_schema.column_names())]


class SnapshotMergersTask(ParquetDatasetCacheMixin):
    """Raw-value per-snapshot rows for every merger/collision event."""

    schema_version = 1
    name = "snapshot_mergers"
    hdf5_reader_kind = "source"
    required_tables: Sequence[str] = ("scalars", "mergers")
    columns_by_table: Mapping[str, Sequence[str] | None] = {
        "scalars": list(_SCALE_SCALAR_COLUMNS),
        "mergers": [
            "Mer NAM1",
            "Mer NAM2",
            "Mer NAM3",
            "Mer NAMC",
            "Mer KW1",
            "Mer KW2",
            "Mer KW3",
            "Mer KWC",
            "Mer M1",
            "Mer M2",
            "Mer M3",
            "Mer XC1",
            "Mer XC2",
            "Mer XC3",
            "Mer VC1",
            "Mer VC2",
            "Mer VC3",
            "Mer XR01",
            "Mer XR02",
            "Mer XR03",
            "Mer VR01",
            "Mer VR02",
            "Mer VR03",
            "Mer XR11",
            "Mer XR12",
            "Mer XR13",
            "Mer VR11",
            "Mer VR12",
            "Mer VR13",
            "Mer POT",
            "Mer RS1",
            "Mer RS2",
            "Mer RS3",
            "Mer L1",
            "Mer L2",
            "Mer L3",
            "Mer TE1",
            "Mer TE2",
            "Mer TE3",
            "Mer RC1",
            "Mer RC2",
            "Mer RC3",
            "Mer MC1",
            "Mer MC2",
            "Mer MC3",
            "Mer A0[au]",
            "Mer ECC0",
            "Mer P0[d]",
            "Mer A1[au]",
            "Mer ECC1",
            "Mer P1[d]",
        ],
    }

    def __init__(
        self,
        config_manager: Any,
        simu_name: str,
        excluded_ttot_by_path: Mapping[str, set[float]] | None = None,
    ) -> None:
        self.config = config_manager
        self.simu_name = simu_name
        self.excluded_ttot_by_path = excluded_ttot_by_path or {}

    @property
    def table_schema(self) -> TableSchema:
        return load_table_schema("snapshot_mergers")

    @property
    def cache_dir(self) -> Path:
        return analysis_cache_dir(self.config, self.simu_name, SNAPSHOT_MERGERS_FEATURE)

    @property
    def parquet_write_options(self) -> Dict[str, Any]:
        return LAKE_PARQUET_WRITE_OPTIONS

    def process_file(
        self,
        hdf5_path: str,
        df_dict: Dict[str, pd.DataFrame],
        meta: Dict[str, Any],
        cache_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        rows = self._build_rows(hdf5_path, df_dict)
        part_meta = self.write_part(hdf5_path, rows)
        return {**part_meta, "file_meta": default_file_meta(hdf5_path, df_dict)}

    def _build_rows(self, hdf5_path: str, df_dict: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        scalars = df_dict.get("scalars", pd.DataFrame())
        mergers = df_dict.get("mergers", pd.DataFrame())
        mergers = _drop_excluded_ttot(mergers, hdf5_path, self.excluded_ttot_by_path)
        if scalars.empty or mergers.empty:
            return self.table_schema.empty_dataframe()

        scale, rdens_pc = _scale_and_rdens_pc(scalars)
        ttot = mergers["TTOT"].to_numpy(dtype="float64")
        cm_x_pc = _rdens_corrected_pc(mergers["Mer XC1"], ttot, rdens_pc["RDENS(1)"])
        cm_y_pc = _rdens_corrected_pc(mergers["Mer XC2"], ttot, rdens_pc["RDENS(2)"])
        cm_z_pc = _rdens_corrected_pc(mergers["Mer XC3"], ttot, rdens_pc["RDENS(3)"])

        rows = pd.DataFrame(
            {
                "simulation_id": self.simu_name,
                "ttot": ttot,
                "time_myr": ttot * scale["t"],
                "object_id_1": _as_int64(mergers["Mer NAM1"]),
                "object_id_2": _as_int64(mergers["Mer NAM2"]),
                "object_id_3": _as_int64(mergers["Mer NAM3"]),
                "cm_id": _as_int64(mergers["Mer NAMC"]),
                "kw_1": _as_int32(mergers["Mer KW1"]),
                "kw_2": _as_int32(mergers["Mer KW2"]),
                "kw_3": _as_int32(mergers["Mer KW3"]),
                "cm_kw": _as_int32(mergers["Mer KWC"]),
                "mass_1_msun": _as_float32(mergers["Mer M1"]),
                "mass_2_msun": _as_float32(mergers["Mer M2"]),
                "mass_3_msun": _as_float32(mergers["Mer M3"]),
                "cm_x_pc": cm_x_pc,
                "cm_y_pc": cm_y_pc,
                "cm_z_pc": cm_z_pc,
                "cm_vx_kms": _as_float32(mergers["Mer VC1"]),
                "cm_vy_kms": _as_float32(mergers["Mer VC2"]),
                "cm_vz_kms": _as_float32(mergers["Mer VC3"]),
                "rel0_x_pc": _as_float32(mergers["Mer XR01"]),
                "rel0_y_pc": _as_float32(mergers["Mer XR02"]),
                "rel0_z_pc": _as_float32(mergers["Mer XR03"]),
                "rel0_vx_kms": _as_float32(mergers["Mer VR01"]),
                "rel0_vy_kms": _as_float32(mergers["Mer VR02"]),
                "rel0_vz_kms": _as_float32(mergers["Mer VR03"]),
                "rel1_x_pc": _as_float32(mergers["Mer XR11"]),
                "rel1_y_pc": _as_float32(mergers["Mer XR12"]),
                "rel1_z_pc": _as_float32(mergers["Mer XR13"]),
                "rel1_vx_kms": _as_float32(mergers["Mer VR11"]),
                "rel1_vy_kms": _as_float32(mergers["Mer VR12"]),
                "rel1_vz_kms": _as_float32(mergers["Mer VR13"]),
                "cm_pot_nb": _as_float32(mergers["Mer POT"]),
                "radius_1_rsun": _as_float32(mergers["Mer RS1"]),
                "radius_2_rsun": _as_float32(mergers["Mer RS2"]),
                "radius_3_rsun": _as_float32(mergers["Mer RS3"]),
                "luminosity_1_lsun": _as_float32(mergers["Mer L1"]),
                "luminosity_2_lsun": _as_float32(mergers["Mer L2"]),
                "luminosity_3_lsun": _as_float32(mergers["Mer L3"]),
                "teff_1_k": _as_float32(mergers["Mer TE1"]),
                "teff_2_k": _as_float32(mergers["Mer TE2"]),
                "teff_3_k": _as_float32(mergers["Mer TE3"]),
                "core_radius_1_rsun": _as_float32(mergers["Mer RC1"]),
                "core_radius_2_rsun": _as_float32(mergers["Mer RC2"]),
                "core_radius_3_rsun": _as_float32(mergers["Mer RC3"]),
                "core_mass_1_msun": _as_float32(mergers["Mer MC1"]),
                "core_mass_2_msun": _as_float32(mergers["Mer MC2"]),
                "core_mass_3_msun": _as_float32(mergers["Mer MC3"]),
                "semi_major_axis_0_au": _as_float32(mergers["Mer A0[au]"]),
                "eccentricity_0": _as_float32(mergers["Mer ECC0"]),
                "period_0_days": _as_float32(mergers["Mer P0[d]"]),
                "semi_major_axis_1_au": _as_float32(mergers["Mer A1[au]"]),
                "eccentricity_1": _as_float32(mergers["Mer ECC1"]),
                "period_1_days": _as_float32(mergers["Mer P1[d]"]),
                "source_hdf5_path": hdf5_path,
            }
        )
        # cm_id's uniqueness is unconfirmed (see its schema description); sort by the
        # parent-member ids instead, consistent with snapshot_binaries.
        rows = rows.sort_values(
            ["ttot", "object_id_1", "object_id_2", "object_id_3"], kind="stable"
        ).reset_index(drop=True)
        return rows[list(self.table_schema.column_names())]


class SnapshotScalarsTask(ParquetTableCacheMixin):
    """One row per snapshot (TTOT) covering every valid slot of the raw scalars array."""

    # v2: apply the same cross-file TTOT dedup winner (excluded_ttot_by_path, latest
    # mtime) as snapshot_singles/binaries/mergers. v1 relied solely on
    # replace_ttot_rows's "last file processed wins" (file-processing order, i.e.
    # filename-derived time) for its own dedup, which can disagree with the other
    # three tables' winner for a restart-boundary TTOT with near-but-not-identical
    # contributing files -- confirmed on real data (0sb/20sb/60sb) as a small
    # (<1e-4%) row-count mismatch between scalars.n_single/n_binary/n_merger and the
    # actual per-object row counts. Bumping schema_version forces a full rebuild of
    # just this table (cheap: one row per TTOT) to apply the fix retroactively.
    schema_version = 2
    name = "snapshot_scalars"
    hdf5_reader_kind = "source"
    required_tables: Sequence[str] = ("scalars",)
    columns_by_table: Mapping[str, Sequence[str] | None] = {"scalars": None}

    def __init__(
        self,
        config_manager: Any,
        simu_name: str,
        excluded_ttot_by_path: Mapping[str, set[float]] | None = None,
    ) -> None:
        self.config = config_manager
        self.simu_name = simu_name
        self.excluded_ttot_by_path = excluded_ttot_by_path or {}

    @property
    def table_schema(self) -> TableSchema:
        return load_table_schema("snapshot_scalars")

    @property
    def cache_path(self) -> Path:
        return (
            analysis_cache_dir(self.config, self.simu_name, SNAPSHOT_SCALARS_FEATURE)
            / "snapshot_scalars.parquet"
        )

    def is_file_fresh(self, hdf5_path: str, meta: Dict[str, Any], cache_df: pd.DataFrame) -> bool:
        cached_times = set()
        if "ttot" in cache_df.columns:
            cached_times.update(float(ttot) for ttot in cache_df["ttot"].dropna().unique())
        return file_is_fresh(hdf5_path, meta, cached_times or None)

    def process_file(
        self,
        hdf5_path: str,
        df_dict: Dict[str, pd.DataFrame],
        meta: Dict[str, Any],
        cache_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        scalars = df_dict.get("scalars", pd.DataFrame())
        scalars = _drop_excluded_ttot(scalars, hdf5_path, self.excluded_ttot_by_path)
        rows = self._build_rows(hdf5_path, scalars)
        return {"rows": rows, "file_meta": default_file_meta(hdf5_path, df_dict)}

    def merge_file_result(
        self, cache_df: pd.DataFrame, hdf5_path: str, result: Dict[str, Any]
    ) -> pd.DataFrame:
        new_df = result.get("rows", self.table_schema.empty_dataframe())
        return replace_ttot_rows(cache_df, new_df, "ttot")

    def finalize_cache(self, cache_df: pd.DataFrame) -> pd.DataFrame:
        cache_df = super().finalize_cache(cache_df)
        if cache_df.empty:
            return cache_df
        return cache_df.sort_values("ttot").reset_index(drop=True)

    def _build_rows(self, hdf5_path: str, scalars: pd.DataFrame) -> pd.DataFrame:
        if scalars.empty:
            return self.table_schema.empty_dataframe()

        def f64(col: str) -> np.ndarray:
            return scalars[col].to_numpy(dtype="float64")

        n = len(scalars)
        ttot = f64("TTOT")
        rows = pd.DataFrame(
            {
                "simulation_id": self.simu_name,
                "ttot": ttot,
                "time_myr": ttot * f64("TSCALE"),
                "npairs": _as_int64(scalars["NPAIRS"]),
                "rbar_pc": f64("RBAR"),
                "zmbar_msun": f64("ZMBAR"),
                "n": _as_int64(scalars["N"]),
                "tstar_myr": f64("TSTAR"),
                "rdens_x_nb": f64("RDENS(1)"),
                "rdens_y_nb": f64("RDENS(2)"),
                "rdens_z_nb": f64("RDENS(3)"),
                "ttot_over_tcr0": f64("TTOT/TCR0"),
                "tscale_myr": f64("TSCALE"),
                "vstar_kms": f64("VSTAR"),
                "rc_nb": f64("RC"),
                "nc": _as_int64(scalars["NC"]),
                "vc_nb": f64("VC"),
                "rhom_nb": f64("RHOM"),
                "cmax": f64("CMAX"),
                "rscale_nb": f64("RSCALE"),
                "rsmin_nb": f64("RSMIN"),
                "dmin1_nb": f64("DMIN1"),
                "rg_x_pc": f64("RG(1)"),
                "rg_y_pc": f64("RG(2)"),
                "rg_z_pc": f64("RG(3)"),
                "vg_x_kmps": f64("VG(1)"),
                "vg_y_kmps": f64("VG(2)"),
                "vg_z_kmps": f64("VG(3)"),
                "tidal_1": f64("TIDAL(1)"),
                "tidal_2": f64("TIDAL(2)"),
                "tidal_3": f64("TIDAL(3)"),
                "tidal_4": f64("TIDAL(4)"),
                "gmg": f64("GMG"),
                "omega": f64("OMEGA"),
                "disk": f64("DISK"),
                "disk_a": f64("A"),
                "disk_b": f64("B"),
                "zmet": f64("ZMET"),
                **{f"zpars_{i}": f64(f"ZPARS({i})") for i in range(1, 21)},
                "etai": f64("ETAI"),
                "etar": f64("ETAR"),
                "etau": f64("ETAU"),
                "eclose_nb": f64("ECLOSE"),
                "dtmin_nb": f64("DTMIN"),
                "rmin_nb": f64("RMIN"),
                "gmin": f64("GMIN"),
                "gmax": f64("GMAX"),
                "smax": f64("SMAX"),
                "nnbopt": _as_int64(scalars["NNBOPT"]),
                "epoch0_myr": f64("EPOCH0"),
                "n_single": _as_int64(scalars["N_SINGLE"]),
                "n_binary": _as_int64(scalars["N_BINARY"]),
                "n_merger": _as_int64(scalars["N_MERGER"]),
                "source_hdf5_path": hdf5_path,
            }
        )
        assert len(rows) == n
        return rows[list(self.table_schema.column_names())]


__all__ = [
    "LAKE_PARQUET_WRITE_OPTIONS",
    "ParticleLakeProcessor",
    "SnapshotSinglesTask",
    "SnapshotBinariesTask",
    "SnapshotMergersTask",
    "SnapshotScalarsTask",
    "compute_ttot_dedup_exclusions",
]
