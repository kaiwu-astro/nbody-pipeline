"""Per-snapshot compact-object history table (VO-safe, Parquet-backed).

Pilot ``object_rows`` scan task: one row per compact object (KW 10-14) per
snapshot, covering both single stars and binary members, in the VO-safe
schema registered as ``compact_object_history`` (see
nbody_pipeline.schemas). See docs/analysis_architecture.md for the
caching-layer design this implements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from nbody_pipeline.analysis.cache_paths import (
    COMPACT_OBJECT_HISTORY_FEATURE,
    analysis_cache_dir,
)
from nbody_pipeline.analysis.hdf5_scan import (
    HDF5ScanJob,
    ScanBackedAnalysisBase,
    default_file_meta,
)
from nbody_pipeline.analysis.parquet_cache import ParquetDatasetCacheMixin
from nbody_pipeline.schemas import TableSchema, load_table_schema

BH_KW = 14
NS_KW = 13
WD_KW = (10, 11, 12)


class CompactObjectHistoryProcessor(ScanBackedAnalysisBase):
    """Build and cache per-snapshot compact-object rows."""

    def build_scan_job(self, simu_name: str, *, force: bool = False) -> HDF5ScanJob:
        """Build the compact-object-history scan job."""
        task = CompactObjectHistoryTask(self.config, simu_name)
        return HDF5ScanJob(simu_name, task, self._scan_options(force=force))

    def update(self, simu_name: str, *, force: bool = False) -> pd.DataFrame:
        """Update the cached compact-object-history table for one simulation."""
        return self._run_scan_job(self.build_scan_job(simu_name, force=force))

    def load(self, simu_name: str, *, update: bool = True, force: bool = False) -> pd.DataFrame:
        """Return the cached (or freshly updated) compact-object-history table."""
        job = self.build_scan_job(simu_name, force=force)
        return self._load_or_update_scan_job(job, update=update)


class CompactObjectHistoryTask(ParquetDatasetCacheMixin):
    """Scan HDF5 files for compact-object (KW 10-14) rows, VO-safe schema."""

    schema_version = 1
    name = "compact_object_history"
    required_tables: Sequence[str] = ("scalars", "singles", "binaries")
    columns_by_table: Mapping[str, Sequence[str] | None] = {
        "scalars": ["TTOT", "Time[Myr]"],
        "singles": [
            "TTOT",
            "Time[Myr]",
            "Name",
            "KW",
            "M",
            "X [pc]",
            "Y [pc]",
            "Z [pc]",
            "V1",
            "V2",
            "V3",
            "Distance_to_cluster_center[pc]",
        ],
        "binaries": [
            "TTOT",
            "Time[Myr]",
            "Bin Name1",
            "Bin Name2",
            "Bin cm Name",
            "Bin KW1",
            "Bin KW2",
            "Bin M1*",
            "Bin M2*",
            "Bin cm X [pc]",
            "Bin cm Y [pc]",
            "Bin cm Z [pc]",
            "Bin cm V1",
            "Bin cm V2",
            "Bin cm V3",
            "Distance_to_cluster_center[pc]",
            "Bin A[au]",
            "Bin ECC",
        ],
    }

    def __init__(self, config_manager: Any, simu_name: str) -> None:
        self.config = config_manager
        self.simu_name = simu_name

    @property
    def table_schema(self) -> TableSchema:
        return load_table_schema("compact_object_history")

    @property
    def cache_dir(self) -> Path:
        return analysis_cache_dir(self.config, self.simu_name, COMPACT_OBJECT_HISTORY_FEATURE)

    def process_file(
        self,
        hdf5_path: str,
        df_dict: Dict[str, pd.DataFrame],
        meta: Dict[str, Any],
        cache_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        rows = self._build_rows(hdf5_path, df_dict)
        return {"rows": rows, "file_meta": default_file_meta(hdf5_path, df_dict)}

    def _build_rows(self, hdf5_path: str, df_dict: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        compact_kw = np.asarray(getattr(self.config, "compact_object_KW", [10, 11, 12, 13, 14]))
        blocks: list[pd.DataFrame] = []

        singles = df_dict.get("singles", pd.DataFrame())
        if not singles.empty and "KW" in singles.columns:
            candidates = singles.loc[singles["KW"].isin(compact_kw)]
            if not candidates.empty:
                blocks.append(self._single_block(hdf5_path, candidates))

        binaries = df_dict.get("binaries", pd.DataFrame())
        if not binaries.empty and {"Bin KW1", "Bin KW2"}.issubset(binaries.columns):
            member1 = binaries.loc[binaries["Bin KW1"].isin(compact_kw)]
            if not member1.empty:
                blocks.append(self._binary_member_block(hdf5_path, member1, component=1))
            member2 = binaries.loc[binaries["Bin KW2"].isin(compact_kw)]
            if not member2.empty:
                blocks.append(self._binary_member_block(hdf5_path, member2, component=2))

        column_names = list(self.table_schema.column_names())
        if not blocks:
            return self.table_schema.empty_dataframe()
        return pd.concat(blocks, ignore_index=True, sort=False)[column_names]

    def _single_block(self, hdf5_path: str, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        return pd.DataFrame(
            {
                "simulation_id": self.simu_name,
                "ttot": df["TTOT"].to_numpy(dtype="float64"),
                "time_myr": df["Time[Myr]"].to_numpy(dtype="float64"),
                "object_id": _int64_ids(df["Name"]),
                "kw": _int64_ids(df["KW"]),
                "object_type": _object_type(df["KW"]),
                "mass_msun": df["M"].to_numpy(dtype="float64"),
                "x_pc": df["X [pc]"].to_numpy(dtype="float64"),
                "y_pc": df["Y [pc]"].to_numpy(dtype="float64"),
                "z_pc": df["Z [pc]"].to_numpy(dtype="float64"),
                "vx_kms": df["V1"].to_numpy(dtype="float64"),
                "vy_kms": df["V2"].to_numpy(dtype="float64"),
                "vz_kms": df["V3"].to_numpy(dtype="float64"),
                "r_pc": df["Distance_to_cluster_center[pc]"].to_numpy(dtype="float64"),
                "is_in_binary": np.zeros(n, dtype="bool"),
                "companion_id": _nullable_int64_ids(pd.Series([np.nan] * n)),
                "companion_kw": _nullable_int64_ids(pd.Series([np.nan] * n)),
                "binary_cm_id": _nullable_int64_ids(pd.Series([np.nan] * n)),
                "companion_mass_msun": np.full(n, np.nan),
                "semi_major_axis_au": np.full(n, np.nan),
                "eccentricity": np.full(n, np.nan),
                "source_hdf5_path": hdf5_path,
            }
        )

    def _binary_member_block(
        self, hdf5_path: str, df: pd.DataFrame, *, component: int
    ) -> pd.DataFrame:
        other = 2 if component == 1 else 1
        n = len(df)
        name_col = f"Bin Name{component}"
        other_name_col = f"Bin Name{other}"
        kw_col = f"Bin KW{component}"
        other_kw_col = f"Bin KW{other}"
        mass_col = f"Bin M{component}*"
        other_mass_col = f"Bin M{other}*"
        binary_cm_ids = (
            _nullable_int64_ids(df["Bin cm Name"])
            if "Bin cm Name" in df.columns
            else _nullable_int64_ids(pd.Series([np.nan] * n))
        )
        semi_major_axis_au = (
            df["Bin A[au]"].to_numpy(dtype="float64")
            if "Bin A[au]" in df.columns
            else np.full(n, np.nan)
        )
        eccentricity = (
            df["Bin ECC"].to_numpy(dtype="float64")
            if "Bin ECC" in df.columns
            else np.full(n, np.nan)
        )
        return pd.DataFrame(
            {
                "simulation_id": self.simu_name,
                "ttot": df["TTOT"].to_numpy(dtype="float64"),
                "time_myr": df["Time[Myr]"].to_numpy(dtype="float64"),
                "object_id": _int64_ids(df[name_col]),
                "kw": _int64_ids(df[kw_col]),
                "object_type": _object_type(df[kw_col]),
                "mass_msun": df[mass_col].to_numpy(dtype="float64"),
                "x_pc": df["Bin cm X [pc]"].to_numpy(dtype="float64"),
                "y_pc": df["Bin cm Y [pc]"].to_numpy(dtype="float64"),
                "z_pc": df["Bin cm Z [pc]"].to_numpy(dtype="float64"),
                "vx_kms": df["Bin cm V1"].to_numpy(dtype="float64"),
                "vy_kms": df["Bin cm V2"].to_numpy(dtype="float64"),
                "vz_kms": df["Bin cm V3"].to_numpy(dtype="float64"),
                "r_pc": df["Distance_to_cluster_center[pc]"].to_numpy(dtype="float64"),
                "is_in_binary": np.ones(n, dtype="bool"),
                "companion_id": _nullable_int64_ids(df[other_name_col]),
                "companion_kw": _nullable_int64_ids(df[other_kw_col]),
                "binary_cm_id": binary_cm_ids,
                "companion_mass_msun": df[other_mass_col].to_numpy(dtype="float64"),
                "semi_major_axis_au": semi_major_axis_au,
                "eccentricity": eccentricity,
                "source_hdf5_path": hdf5_path,
            }
        )


def _object_type(kw: pd.Series) -> np.ndarray:
    kw_values = kw.to_numpy(dtype="float64")
    conditions = [kw_values == BH_KW, kw_values == NS_KW, np.isin(kw_values, WD_KW)]
    choices = ["BH", "NS", "WD"]
    return np.select(conditions, choices, default="unknown")


def _int64_ids(series: pd.Series) -> np.ndarray:
    """Round-trip whole-valued identifiers/codes stored as floats to int64."""
    return np.rint(series.to_numpy(dtype="float64")).astype("int64")


def _nullable_int64_ids(series: pd.Series) -> pd.array:
    """Round-trip identifiers/codes to nullable Int64, preserving NaN as NA."""
    return pd.array(np.rint(series.to_numpy(dtype="float64")), dtype="Int64")
