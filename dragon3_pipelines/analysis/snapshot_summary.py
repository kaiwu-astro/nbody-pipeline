"""One-row-per-snapshot cluster summary table (VO-safe, Parquet-backed).

Pilot ``snapshot_scalar`` scan task: one row per TTOT, summarizing
cluster-wide scalar quantities (population counts, total mass, core/half-mass
radius, galactocentric position/velocity), in the VO-safe schema registered
as ``snapshot_summary`` (see dragon3_pipelines.schemas). See
docs/analysis_architecture.md for the caching-layer design this implements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from dragon3_pipelines.analysis.cache_paths import (
    SNAPSHOT_SUMMARY_FEATURE,
    analysis_cache_dir,
)
from dragon3_pipelines.analysis.hdf5_scan import (
    HDF5ScanJob,
    ScanBackedAnalysisBase,
    default_file_meta,
    file_is_fresh,
    replace_ttot_rows,
)
from dragon3_pipelines.analysis.parquet_cache import ParquetTableCacheMixin
from dragon3_pipelines.schemas import TableSchema, load_table_schema

BH_KW = 14
NS_KW = 13
WD_KW = (10, 11, 12)

_SCALAR_COLUMNS = [
    "TTOT",
    "Time[Myr]",
    "N_SINGLE",
    "N_BINARY",
    "RC",
    "RBAR",
    "RG(1)",
    "RG(2)",
    "RG(3)",
    "VG(1)",
    "VG(2)",
    "VG(3)",
]


class SnapshotSummaryProcessor(ScanBackedAnalysisBase):
    """Build and cache one-row-per-snapshot cluster summaries."""

    def build_scan_job(self, simu_name: str, *, force: bool = False) -> HDF5ScanJob:
        """Build the snapshot-summary scan job."""
        task = SnapshotSummaryTask(self.config, simu_name)
        return HDF5ScanJob(simu_name, task, self._scan_options(force=force))

    def update(self, simu_name: str, *, force: bool = False) -> pd.DataFrame:
        """Update the cached snapshot-summary table for one simulation."""
        return self._run_scan_job(self.build_scan_job(simu_name, force=force))

    def load(self, simu_name: str, *, update: bool = True, force: bool = False) -> pd.DataFrame:
        """Return the cached (or freshly updated) snapshot-summary table."""
        job = self.build_scan_job(simu_name, force=force)
        return self._load_or_update_scan_job(job, update=update)


class SnapshotSummaryTask(ParquetTableCacheMixin):
    """Scan HDF5 files and build one cluster-summary row per TTOT."""

    schema_version = 1
    name = "snapshot_summary"
    required_tables: Sequence[str] = ("scalars", "singles", "binaries")
    columns_by_table: Mapping[str, Sequence[str] | None] = {
        "scalars": list(_SCALAR_COLUMNS),
        "singles": ["TTOT", "KW", "M", "Distance_to_cluster_center[pc]"],
        "binaries": [
            "TTOT",
            "Bin KW1",
            "Bin KW2",
            "Bin M1*",
            "Bin M2*",
            "Distance_to_cluster_center[pc]",
        ],
    }

    def __init__(self, config_manager: Any, simu_name: str) -> None:
        self.config = config_manager
        self.simu_name = simu_name

    @property
    def table_schema(self) -> TableSchema:
        return load_table_schema("snapshot_summary")

    @property
    def cache_path(self) -> Path:
        return (
            analysis_cache_dir(self.config, self.simu_name, SNAPSHOT_SUMMARY_FEATURE)
            / "snapshot_summary.parquet"
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
        rows = self._build_rows(
            df_dict.get("scalars", pd.DataFrame()),
            df_dict.get("singles", pd.DataFrame()),
            df_dict.get("binaries", pd.DataFrame()),
        )
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

    def _build_rows(
        self, scalars: pd.DataFrame, singles: pd.DataFrame, binaries: pd.DataFrame
    ) -> pd.DataFrame:
        if scalars.empty:
            return self.table_schema.empty_dataframe()

        base = scalars.reset_index(drop=True)
        missing = set(_SCALAR_COLUMNS).difference(base.columns)
        if missing:
            raise ValueError(
                "scalars table missing required columns for snapshot_summary: "
                + ", ".join(sorted(missing))
            )

        objects = _object_rows(singles, binaries)
        stats = _snapshot_stats(objects)

        ttot = base["TTOT"].to_numpy(dtype="float64")
        stats = stats.reindex(ttot)
        n = len(base)
        rows = pd.DataFrame(
            {
                "simulation_id": self.simu_name,
                "ttot": ttot,
                "time_myr": base["Time[Myr]"].to_numpy(dtype="float64"),
                "n_singles": _round_to_int64(base["N_SINGLE"]),
                "n_binaries": _round_to_int64(base["N_BINARY"]),
                "n_bh": np.nan_to_num(stats["n_bh"].to_numpy(dtype="float64")).astype("int64"),
                "n_ns": np.nan_to_num(stats["n_ns"].to_numpy(dtype="float64")).astype("int64"),
                "n_wd": np.nan_to_num(stats["n_wd"].to_numpy(dtype="float64")).astype("int64"),
                "total_mass_msun": np.nan_to_num(
                    stats["total_mass_msun"].to_numpy(dtype="float64")
                ),
                "core_radius_pc": (
                    base["RC"].to_numpy(dtype="float64") * base["RBAR"].to_numpy(dtype="float64")
                ),
                "half_mass_radius_pc": np.nan_to_num(
                    stats["half_mass_radius_pc"].to_numpy(dtype="float64")
                ),
                "rg_x_pc": base["RG(1)"].to_numpy(dtype="float64"),
                "rg_y_pc": base["RG(2)"].to_numpy(dtype="float64"),
                "rg_z_pc": base["RG(3)"].to_numpy(dtype="float64"),
                "vg_x_kmps": base["VG(1)"].to_numpy(dtype="float64"),
                "vg_y_kmps": base["VG(2)"].to_numpy(dtype="float64"),
                "vg_z_kmps": base["VG(3)"].to_numpy(dtype="float64"),
            }
        )
        assert len(rows) == n
        return rows[list(self.table_schema.column_names())]


def _object_rows(singles: pd.DataFrame, binaries: pd.DataFrame) -> pd.DataFrame:
    """Column-aligned (ttot, kw, mass, r_pc) rows for singles + both binary members."""
    parts: list[pd.DataFrame] = []
    if not singles.empty:
        parts.append(
            pd.DataFrame(
                {
                    "ttot": singles["TTOT"].to_numpy(dtype="float64"),
                    "kw": singles["KW"].to_numpy(dtype="float64"),
                    "mass": singles["M"].to_numpy(dtype="float64"),
                    "r_pc": singles["Distance_to_cluster_center[pc]"].to_numpy(dtype="float64"),
                }
            )
        )
    if not binaries.empty:
        for component in (1, 2):
            parts.append(
                pd.DataFrame(
                    {
                        "ttot": binaries["TTOT"].to_numpy(dtype="float64"),
                        "kw": binaries[f"Bin KW{component}"].to_numpy(dtype="float64"),
                        "mass": binaries[f"Bin M{component}*"].to_numpy(dtype="float64"),
                        "r_pc": binaries["Distance_to_cluster_center[pc]"].to_numpy(
                            dtype="float64"
                        ),
                    }
                )
            )
    if not parts:
        return pd.DataFrame(columns=["ttot", "kw", "mass", "r_pc"])
    return pd.concat(parts, ignore_index=True)


def _snapshot_stats(objects: pd.DataFrame) -> pd.DataFrame:
    """Per-TTOT n_bh/n_ns/n_wd/total_mass_msun/half_mass_radius_pc, indexed by ttot."""
    columns = ["n_bh", "n_ns", "n_wd", "total_mass_msun", "half_mass_radius_pc"]
    if objects.empty:
        return pd.DataFrame(columns=columns).astype("float64")

    counts = pd.DataFrame(
        {
            "ttot": objects["ttot"],
            "mass": objects["mass"],
            "n_bh": (objects["kw"] == BH_KW).astype("float64"),
            "n_ns": (objects["kw"] == NS_KW).astype("float64"),
            "n_wd": objects["kw"].isin(WD_KW).astype("float64"),
        }
    )
    grouped = counts.groupby("ttot", sort=False)
    stats = (
        grouped[["n_bh", "n_ns", "n_wd", "mass"]].sum().rename(columns={"mass": "total_mass_msun"})
    )
    stats["half_mass_radius_pc"] = objects.groupby("ttot", sort=False).apply(
        _half_mass_radius_pc, include_groups=False
    )
    return stats[columns]


def _half_mass_radius_pc(group: pd.DataFrame) -> float:
    """Radius enclosing half the total mass: sort by r_pc, cumsum, take the median crossing."""
    sorted_group = group.sort_values("r_pc")
    cumulative_mass = sorted_group["mass"].to_numpy(dtype="float64").cumsum()
    total_mass = float(cumulative_mass[-1]) if len(cumulative_mass) else 0.0
    if total_mass <= 0:
        return np.nan
    index = int(np.searchsorted(cumulative_mass, 0.5 * total_mass))
    index = min(index, len(sorted_group) - 1)
    return float(sorted_group["r_pc"].to_numpy(dtype="float64")[index])


def _round_to_int64(series: pd.Series) -> np.ndarray:
    return np.rint(series.to_numpy(dtype="float64")).astype("int64")
