"""Current-mass Lagrangian table construction from HDF5 snapshots."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from nbody_pipeline.analysis.cache_paths import CURRENT_LAGRANGIAN_FEATURE, analysis_cache_dir
from nbody_pipeline.analysis.hdf5_scan import (
    FeatherMetaCacheMixin,
    HDF5ScanJob,
    HDF5ScanOptions,
    ScanBackedAnalysisBase,
    default_file_meta,
    file_is_fresh,
    replace_ttot_rows,
)
from nbody_pipeline.io.text_parsers import make_l7header, transform_l7df_to_sns_friendly

logger = logging.getLogger(__name__)


class CurrentMassLagrangianProcessor(ScanBackedAnalysisBase):
    """Build and cache current-mass Lagrangian profiles from HDF5 snapshots."""

    SCHEMA_VERSION = 1
    METRICS = [
        "rlagr",
        "avmass",
        "nshell",
        "vx",
        "vy",
        "vz",
        "v",
        "vr",
        "vt",
        "sigma2",
        "sigma_r2",
        "sigma_t2",
        "vrot",
    ]

    @property
    def percentages(self) -> List[str]:
        """Return lagr.7 total-population percentage suffixes."""
        return [
            col.removeprefix("rlagr")
            for col in make_l7header()
            if col.startswith("rlagr") and not col.startswith(("rlagr_s", "rlagr_b"))
        ]

    def _current_lagrangian_config(self) -> Dict[str, Any]:
        defaults = {
            "enabled": True,
            "cache_filename": "current_mass_lagr.feather",
        }
        user_config = getattr(self.config, "current_lagrangian", {}) or {}
        return {**defaults, **user_config}

    def _cache_dir(self, simu_name: str) -> Path:
        return analysis_cache_dir(self.config, simu_name, CURRENT_LAGRANGIAN_FEATURE)

    def _cache_path(self, simu_name: str) -> Path:
        return self._cache_dir(simu_name) / self._current_lagrangian_config()["cache_filename"]

    def _meta_path(self, simu_name: str) -> Path:
        cache_path = self._cache_path(simu_name)
        return cache_path.with_name(cache_path.stem + ".meta.json")

    def _read_cache(self, simu_name: str) -> pd.DataFrame:
        cache_path = self._cache_path(simu_name)
        if not cache_path.exists():
            return pd.DataFrame()
        return pd.read_feather(cache_path)

    def _read_meta(self, simu_name: str) -> Dict[str, Any]:
        meta_path = self._meta_path(simu_name)
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read current Lagrangian metadata %s: %r", meta_path, exc)
            return {}

    def _write_cache_and_meta(
        self, simu_name: str, df: pd.DataFrame, processed_files: Dict[str, Dict[str, Any]]
    ) -> None:
        cache_dir = self._cache_dir(simu_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(simu_name)
        meta_path = self._meta_path(simu_name)

        tmp_cache_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")

        df.to_feather(tmp_cache_path)
        os.replace(tmp_cache_path, cache_path)

        meta = {
            "schema_version": self.SCHEMA_VERSION,
            "statistics": "current-mass weighted, singles table only",
            "percentages": self.percentages,
            "processed_files": processed_files,
        }
        tmp_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
        os.replace(tmp_meta_path, meta_path)

    def _is_file_fresh_in_meta(
        self, hdf5_path: str, meta: Dict[str, Any], cached_times: set[float]
    ) -> bool:
        file_meta = meta.get("processed_files", {}).get(hdf5_path)
        if not file_meta:
            return False
        try:
            current_mtime = os.path.getmtime(hdf5_path)
        except OSError:
            return False
        if not np.isclose(
            float(file_meta.get("mtime", np.nan)), current_mtime, rtol=0.0, atol=1e-9
        ):
            return False
        return set(file_meta.get("ttot", [])).issubset(cached_times)

    def update(self, simu_name: str, *, force: bool = False) -> pd.DataFrame:
        """Update the cached current-mass Lagrangian table for one simulation."""
        job = self.build_scan_job(simu_name, force=force)
        return self._run_scan_job(job)

    def build_scan_job(self, simu_name: str, *, force: bool = False) -> HDF5ScanJob:
        """Build a scan job for batched execution by ``HDF5ScanSession``."""
        options = self._scan_options(force=force)
        task = CurrentMassLagrangianTask(self, simu_name)
        return HDF5ScanJob(simu_name, task, options)

    def load_sns_friendly_data(self, simu_name: str, update: bool = True) -> pd.DataFrame:
        """Return a seaborn-friendly long table compatible with ``LagrVisualizer``."""
        df = self.update(simu_name) if update else self._read_cache(simu_name)
        if df.empty:
            return pd.DataFrame(columns=["Time[Myr]", "Percentage", "Metric", "Value", "%"])

        plot_df = df.drop(columns=["Time[NB]"], errors="ignore")
        l7df_sns = transform_l7df_to_sns_friendly(plot_df)
        return self._append_sigma_rows(l7df_sns)

    def compute_snapshot(
        self, single_df_at_t: pd.DataFrame, scalar_row: pd.Series
    ) -> Dict[str, float]:
        """Compute one wide-table row for a single snapshot."""
        row: Dict[str, float] = {
            "Time[NB]": float(scalar_row["TTOT"]),
            "Time[Myr]": float(scalar_row["Time[Myr]"]),
        }

        snapshot_df = single_df_at_t.copy()
        if snapshot_df.empty:
            for metric in self.METRICS:
                for suffix in self.percentages:
                    row[f"{metric}{suffix}"] = np.nan
            return row

        mass = snapshot_df["M"].to_numpy(dtype=float)
        total_mass = mass.sum()
        if total_mass <= 0:
            raise ValueError("Current-mass Lagrangian snapshot has non-positive total mass.")

        velocity = snapshot_df[["V1", "V2", "V3"]].to_numpy(dtype=float)
        center_of_mass_velocity = np.average(velocity, axis=0, weights=mass)
        velocity = velocity - center_of_mass_velocity

        radius = snapshot_df["Distance_to_cluster_center[pc]"].to_numpy(dtype=float)
        position = self._position_array(snapshot_df, radius)
        order = np.argsort(radius, kind="mergesort")
        sorted_mass = mass[order]
        sorted_radius = radius[order]
        cumulative_mass = np.cumsum(sorted_mass)
        prefix_stats = self._build_prefix_stats(
            sorted_mass,
            sorted_radius,
            position[order],
            velocity[order],
        )

        for suffix in self.percentages:
            if suffix == "<RC":
                rc_pc = float(scalar_row["RC"]) * float(scalar_row["RBAR"])
                count = int(np.searchsorted(sorted_radius, rc_pc, side="right"))
                stats = self._stats_at_prefix(prefix_stats, count - 1)
            else:
                target_mass = total_mass * float(suffix)
                idx = int(np.searchsorted(cumulative_mass, target_mass, side="left"))
                idx = min(idx, len(order) - 1)
                stats = self._stats_at_prefix(prefix_stats, idx)

            for metric, value in stats.items():
                row[f"{metric}{suffix}"] = value

        return row

    def _position_array(self, snapshot_df: pd.DataFrame, radius: np.ndarray) -> np.ndarray:
        if {"X [pc]", "Y [pc]", "Z [pc]"}.issubset(snapshot_df.columns):
            return snapshot_df[["X [pc]", "Y [pc]", "Z [pc]"]].to_numpy(dtype=float)
        position_cols = ["X1", "X2", "X3"]
        if set(position_cols).issubset(snapshot_df.columns):
            return snapshot_df[position_cols].to_numpy(dtype=float)
        position = np.zeros((len(radius), 3), dtype=float)
        position[:, 0] = radius
        return position

    def _empty_region_stats(self) -> Dict[str, float]:
        return {
            "rlagr": np.nan,
            "avmass": np.nan,
            "nshell": 0,
            "vx": np.nan,
            "vy": np.nan,
            "vz": np.nan,
            "v": np.nan,
            "vr": np.nan,
            "vt": np.nan,
            "sigma2": np.nan,
            "sigma_r2": np.nan,
            "sigma_t2": np.nan,
            "vrot": np.nan,
        }

    def _build_prefix_stats(
        self,
        sorted_mass: np.ndarray,
        sorted_radius: np.ndarray,
        sorted_position: np.ndarray,
        sorted_velocity: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        speed = np.linalg.norm(sorted_velocity, axis=1)
        radial_unit = np.divide(
            sorted_position,
            sorted_radius[:, None],
            out=np.zeros_like(sorted_position, dtype=float),
            where=sorted_radius[:, None] > 0,
        )
        radial_velocity = np.sum(sorted_velocity * radial_unit, axis=1)
        tangential_velocity = np.sqrt(np.maximum(speed**2 - radial_velocity**2, 0.0))
        velocity2 = np.sum(sorted_velocity**2, axis=1)
        angular_momentum_speed = np.linalg.norm(np.cross(sorted_position, sorted_velocity), axis=1)
        vrot = np.divide(
            angular_momentum_speed,
            sorted_radius,
            out=np.zeros_like(angular_momentum_speed, dtype=float),
            where=sorted_radius > 0,
        )

        weighted_velocity = sorted_velocity * sorted_mass[:, None]
        return {
            "radius": sorted_radius,
            "mass": np.cumsum(sorted_mass),
            "count": np.arange(1, len(sorted_mass) + 1, dtype=float),
            "velocity": np.cumsum(weighted_velocity, axis=0),
            "speed": np.cumsum(sorted_mass * speed),
            "radial_velocity": np.cumsum(sorted_mass * radial_velocity),
            "tangential_velocity": np.cumsum(sorted_mass * tangential_velocity),
            "velocity2": np.cumsum(sorted_mass * velocity2),
            "radial_velocity2": np.cumsum(sorted_mass * radial_velocity**2),
            "vrot": np.cumsum(sorted_mass * vrot),
        }

    def _stats_at_prefix(self, prefix_stats: Dict[str, np.ndarray], idx: int) -> Dict[str, float]:
        if idx < 0:
            return self._empty_region_stats()

        mass_sum = float(prefix_stats["mass"][idx])
        count = int(prefix_stats["count"][idx])
        mean_velocity = prefix_stats["velocity"][idx] / mass_sum
        mean_speed = float(prefix_stats["speed"][idx] / mass_sum)
        mean_radial_velocity = float(prefix_stats["radial_velocity"][idx] / mass_sum)
        mean_tangential_velocity = float(prefix_stats["tangential_velocity"][idx] / mass_sum)

        sigma2 = float(
            prefix_stats["velocity2"][idx] / mass_sum - np.dot(mean_velocity, mean_velocity)
        )
        sigma2 = max(sigma2, 0.0)
        sigma_r2 = float(prefix_stats["radial_velocity2"][idx] / mass_sum - mean_radial_velocity**2)
        sigma_r2 = max(sigma_r2, 0.0)
        sigma_t2 = max(sigma2 - sigma_r2, 0.0)

        return {
            "rlagr": float(prefix_stats["radius"][idx]),
            "avmass": float(mass_sum / count),
            "nshell": count,
            "vx": float(mean_velocity[0]),
            "vy": float(mean_velocity[1]),
            "vz": float(mean_velocity[2]),
            "v": mean_speed,
            "vr": mean_radial_velocity,
            "vt": mean_tangential_velocity,
            "sigma2": sigma2,
            "sigma_r2": sigma_r2,
            "sigma_t2": sigma_t2,
            "vrot": float(prefix_stats["vrot"][idx] / mass_sum),
        }

    def _compute_region_stats(
        self,
        mass: np.ndarray,
        radius: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        mask: Iterable[bool],
    ) -> Dict[str, float]:
        mask = np.asarray(mask, dtype=bool)
        n_stars = int(mask.sum())
        if n_stars == 0:
            return self._empty_region_stats()

        region_mass = mass[mask]
        region_position = position[mask]
        region_velocity = velocity[mask]
        region_radius = radius[mask]
        region_mass_sum = region_mass.sum()

        mean_velocity = np.average(region_velocity, axis=0, weights=region_mass)
        speed = np.linalg.norm(region_velocity, axis=1)

        radial_unit = np.divide(
            region_position,
            region_radius[:, None],
            out=np.zeros_like(region_position, dtype=float),
            where=region_radius[:, None] > 0,
        )
        radial_velocity = np.sum(region_velocity * radial_unit, axis=1)
        tangential_velocity2 = np.maximum(speed**2 - radial_velocity**2, 0.0)
        tangential_velocity = np.sqrt(tangential_velocity2)

        mean_speed = float(np.average(speed, weights=region_mass))
        mean_radial_velocity = float(np.average(radial_velocity, weights=region_mass))
        mean_tangential_velocity = float(np.average(tangential_velocity, weights=region_mass))

        velocity_delta = region_velocity - mean_velocity
        sigma2 = float(np.average(np.sum(velocity_delta**2, axis=1), weights=region_mass))
        sigma_r2 = float(
            np.average((radial_velocity - mean_radial_velocity) ** 2, weights=region_mass)
        )
        sigma_t2 = max(sigma2 - sigma_r2, 0.0)

        angular_momentum_speed = np.linalg.norm(np.cross(region_position, region_velocity), axis=1)
        vrot = np.divide(
            angular_momentum_speed,
            region_radius,
            out=np.zeros_like(angular_momentum_speed, dtype=float),
            where=region_radius > 0,
        )

        return {
            "rlagr": float(np.max(region_radius)),
            "avmass": float(region_mass_sum / n_stars),
            "nshell": n_stars,
            "vx": float(mean_velocity[0]),
            "vy": float(mean_velocity[1]),
            "vz": float(mean_velocity[2]),
            "v": mean_speed,
            "vr": mean_radial_velocity,
            "vt": mean_tangential_velocity,
            "sigma2": sigma2,
            "sigma_r2": sigma_r2,
            "sigma_t2": sigma_t2,
            "vrot": float(np.average(vrot, weights=region_mass)),
        }

    def _append_sigma_rows(self, l7df_sns: pd.DataFrame) -> pd.DataFrame:
        new_rows = []
        for metric_old in ["sigma2", "sigma_r2", "sigma_t2"]:
            df_subset = l7df_sns[l7df_sns["Metric"] == metric_old].copy()
            if not df_subset.empty:
                df_subset["Value"] = np.sqrt(df_subset["Value"])
                df_subset["Metric"] = metric_old[:-1]
                new_rows.append(df_subset)
        if not new_rows:
            return l7df_sns
        return pd.concat([l7df_sns] + new_rows, ignore_index=True)


class CurrentMassLagrangianTask(FeatherMetaCacheMixin):
    """Scan task backing CurrentMassLagrangianProcessor.update()."""

    schema_version = CurrentMassLagrangianProcessor.SCHEMA_VERSION
    required_tables = ("scalars", "singles")
    columns_by_table = {"scalars": None, "singles": None}

    def __init__(self, processor: CurrentMassLagrangianProcessor, simu_name: str) -> None:
        self.processor = processor
        self.config = processor.config
        self.simu_name = simu_name
        self.name = "current_mass_lagrangian"

    @property
    def cache_path(self) -> Path:
        return self.processor._cache_path(self.simu_name)

    def is_file_fresh(self, hdf5_path: str, meta: Dict[str, Any], cache_df: pd.DataFrame) -> bool:
        cached_times = (
            set(cache_df["Time[NB]"].astype(float).tolist())
            if "Time[NB]" in cache_df.columns and not cache_df.empty
            else set()
        )
        return file_is_fresh(hdf5_path, meta, cached_times)

    def process_file(
        self,
        hdf5_path: str,
        df_dict: Dict[str, pd.DataFrame],
        meta: Dict[str, Any],
        cache_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        file_meta = default_file_meta(hdf5_path, df_dict)
        cached_times = (
            set(cache_df["Time[NB]"].astype(float).tolist())
            if "Time[NB]" in cache_df.columns and not cache_df.empty
            else set()
        )
        old_file_meta = meta.get("processed_files", {}).get(hdf5_path)
        current_mtime = file_meta["mtime"]
        if old_file_meta and not np.isclose(
            float(old_file_meta.get("mtime", np.nan)), current_mtime, rtol=0.0, atol=1e-9
        ):
            times_to_compute = file_meta["ttot"]
        else:
            times_to_compute = [ttot for ttot in file_meta["ttot"] if ttot not in cached_times]

        file_rows = []
        for ttot in times_to_compute:
            single_df_at_t, _, is_valid = self.processor.hdf5_file_processor.get_snapshot_at_t(
                {
                    "scalars": df_dict["scalars"],
                    "singles": df_dict["singles"],
                    "binaries": pd.DataFrame({"TTOT": []}),
                },
                ttot,
            )
            if not is_valid or single_df_at_t is None:
                logger.warning(
                    "Skipping invalid current Lagrangian snapshot %s TTOT=%s",
                    hdf5_path,
                    ttot,
                )
                continue
            scalar_row = df_dict["scalars"].loc[ttot]
            file_rows.append(self.processor.compute_snapshot(single_df_at_t, scalar_row))

        return {"rows": pd.DataFrame(file_rows), "file_meta": file_meta}

    def merge_file_result(
        self, cache_df: pd.DataFrame, hdf5_path: str, result: Dict[str, Any]
    ) -> pd.DataFrame:
        new_df = result.get("rows", pd.DataFrame())
        if "Time[NB]" in cache_df.columns:
            ttot_values = result.get("file_meta", {}).get("ttot", [])
            cache_df = cache_df[~cache_df["Time[NB]"].astype(float).isin(ttot_values)]
        return replace_ttot_rows(cache_df, new_df, "Time[NB]")

    def finalize_cache(self, cache_df: pd.DataFrame) -> pd.DataFrame:
        if cache_df.empty:
            return cache_df.reset_index(drop=True)
        return (
            cache_df.sort_values("Time[NB]")
            .drop_duplicates(subset=["Time[NB]"], keep="last")
            .reset_index(drop=True)
        )

    def build_meta(
        self,
        cache_df: pd.DataFrame,
        processed_files: Dict[str, Dict[str, Any]],
        options: HDF5ScanOptions,
    ) -> Dict[str, Any]:
        meta = super().build_meta(cache_df, processed_files, options)
        meta.update(
            {
                "statistics": "current-mass weighted, singles table only",
                "percentages": self.processor.percentages,
            }
        )
        return meta
