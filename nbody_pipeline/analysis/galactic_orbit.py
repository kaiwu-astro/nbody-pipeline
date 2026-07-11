"""Cluster galactic-orbit table construction from scalar HDF5 snapshots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from nbody_pipeline.analysis.cache_paths import GALACTIC_ORBIT_FEATURE, analysis_cache_dir
from nbody_pipeline.analysis.hdf5_scan import (
    FeatherMetaCacheMixin,
    HDF5ScanJob,
    ScanBackedAnalysisBase,
    default_file_meta,
    file_is_fresh,
)

logger = logging.getLogger(__name__)

GALACTIC_ORBIT_COLUMNS = [
    "TTOT",
    "Time[Myr]",
    "RG(1)",
    "RG(2)",
    "RG(3)",
    "VG(1)",
    "VG(2)",
    "VG(3)",
    "source_hdf5_path",
    "source_file_time",
    "source_row_index",
]


class GalacticOrbitProcessor(ScanBackedAnalysisBase):
    """Build and cache cluster galactic-orbit points from scalar snapshots."""

    def _galactic_orbit_config(self) -> Dict[str, Any]:
        defaults = {
            "enabled": True,
            "cache_filename": "galactic_orbit.feather",
            "time_color_max_myr": 500.0,
        }
        user_config = getattr(self.config, "galactic_orbit", {}) or {}
        return {**defaults, **user_config}

    def build_scan_job(self, simu_name: str, *, force: bool = False) -> HDF5ScanJob:
        """Build a scan job for batched execution by ``HDF5ScanSession``."""
        options = self._scan_options(force=force)
        task = GalacticOrbitTask(self, simu_name)
        return HDF5ScanJob(simu_name, task, options)

    def update(self, simu_name: str, *, force: bool = False) -> pd.DataFrame:
        """Update the cached galactic-orbit table for one simulation."""
        return self._run_scan_job(self.build_scan_job(simu_name, force=force))

    def load_plot_data(self, simu_name: str, update: bool = True) -> pd.DataFrame:
        """Return stable, de-duplicated orbit points for plotting."""
        job = self.build_scan_job(simu_name)
        cache_df = self._load_or_update_scan_job(job, update=update)
        return deduplicate_orbit_points(cache_df)


class GalacticOrbitTask(FeatherMetaCacheMixin):
    """Scan scalar snapshots and store cluster galactic positions and velocities."""

    schema_version = 1
    required_tables: Sequence[str] = ("scalars",)
    columns_by_table: Mapping[str, Sequence[str] | None] = {
        "scalars": [
            "TTOT",
            "Time[Myr]",
            "RG(1)",
            "RG(2)",
            "RG(3)",
            "VG(1)",
            "VG(2)",
            "VG(3)",
        ]
    }

    def __init__(self, processor: GalacticOrbitProcessor, simu_name: str) -> None:
        self.processor = processor
        self.config = processor.config
        self.simu_name = simu_name
        self.name = "galactic_orbit"

    @property
    def cache_path(self) -> Path:
        orbit_config = self.processor._galactic_orbit_config()
        return (
            analysis_cache_dir(self.config, self.simu_name, GALACTIC_ORBIT_FEATURE)
            / orbit_config["cache_filename"]
        )

    def read_cache(self) -> pd.DataFrame:
        cache_df = super().read_cache()
        if cache_df.empty:
            return pd.DataFrame(columns=GALACTIC_ORBIT_COLUMNS)
        return cache_df

    def is_file_fresh(self, hdf5_path: str, meta: Dict[str, Any], cache_df: pd.DataFrame) -> bool:
        return file_is_fresh(hdf5_path, meta)

    def process_file(
        self,
        hdf5_path: str,
        df_dict: Dict[str, pd.DataFrame],
        meta: Dict[str, Any],
        cache_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        scalars = df_dict.get("scalars", pd.DataFrame())
        missing = set(self.columns_by_table["scalars"] or ()).difference(scalars.columns)
        if missing:
            raise ValueError(
                "Scalar table missing required columns for galactic orbit: "
                + ", ".join(sorted(missing))
            )

        rows = scalars.reset_index(drop=True).copy()
        rows = rows[list(self.columns_by_table["scalars"])]
        rows["source_hdf5_path"] = hdf5_path
        rows["source_file_time"] = (
            self.processor.hdf5_file_processor.get_hdf5_file_time_from_filename(hdf5_path)
        )
        rows["source_row_index"] = rows.index.astype(int)
        rows = rows[GALACTIC_ORBIT_COLUMNS]
        return {"rows": rows, "file_meta": default_file_meta(hdf5_path, df_dict)}

    def merge_file_result(
        self, cache_df: pd.DataFrame, hdf5_path: str, result: Dict[str, Any]
    ) -> pd.DataFrame:
        if "source_hdf5_path" in cache_df.columns:
            cache_df = cache_df[cache_df["source_hdf5_path"] != hdf5_path]
        new_df = result.get("rows", pd.DataFrame(columns=GALACTIC_ORBIT_COLUMNS))
        if cache_df.empty:
            return new_df.reset_index(drop=True)
        if new_df.empty:
            return cache_df.reset_index(drop=True)
        return pd.concat([cache_df, new_df], ignore_index=True, sort=False)

    def finalize_cache(self, cache_df: pd.DataFrame) -> pd.DataFrame:
        if cache_df.empty:
            return pd.DataFrame(columns=GALACTIC_ORBIT_COLUMNS)
        sort_columns = ["source_file_time", "source_hdf5_path", "source_row_index"]
        cache_df = cache_df.sort_values(sort_columns, kind="mergesort")
        return cache_df.reset_index(drop=True)[GALACTIC_ORBIT_COLUMNS]


def deduplicate_orbit_points(cache_df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate TTOT values after stable source ordering, warning if any are found."""
    if cache_df.empty:
        return pd.DataFrame(columns=GALACTIC_ORBIT_COLUMNS)
    sort_columns = ["source_file_time", "source_hdf5_path", "source_row_index"]
    plot_df = cache_df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    duplicate_mask = plot_df["TTOT"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_values = sorted(
            float(value) for value in plot_df.loc[duplicate_mask, "TTOT"].unique()
        )
        logger.warning(
            "Duplicate galactic-orbit TTOT values detected; keeping first source row for: %s",
            duplicate_values,
        )
        plot_df = plot_df.drop_duplicates(subset=["TTOT"], keep="first")
    return plot_df.reset_index(drop=True)
