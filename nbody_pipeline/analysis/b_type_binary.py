"""Extract processed binary rows containing B-type main-sequence members."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from nbody_pipeline.analysis.cache_paths import B_TYPE_BINARY_FEATURE, analysis_cache_dir
from nbody_pipeline.analysis.hdf5_scan import (
    FeatherMetaCacheMixin,
    HDF5ScanJob,
    HDF5ScanOptions,
    ScanBackedAnalysisBase,
    default_file_meta,
    file_is_fresh,
)
from nbody_pipeline.analysis.primordial_binary import PrimordialBinaryIdentifier

logger = logging.getLogger(__name__)


class BTypeBinaryExtractor(ScanBackedAnalysisBase):
    """Load processed binary rows where either component matches the B-type criteria."""

    def load_b_type_binaries(
        self,
        simu_name: str,
        *,
        update: bool = True,
        force: bool = False,
    ) -> pd.DataFrame:
        """Return complete processed binary rows containing B-type main-sequence members."""
        job = self.build_scan_job(
            simu_name,
            update=update,
            force=force,
        )
        return self._load_or_update_scan_job(job, update=update)

    def build_scan_job(
        self,
        simu_name: str,
        *,
        update: bool = True,
        force: bool = False,
    ) -> HDF5ScanJob:
        """Build a scan job for batched execution by ``HDF5ScanSession``."""
        options = self._scan_options(force=force)

        primordial = self._load_primordial_binaries(
            simu_name,
            update=update,
            wait_age_hour=options.wait_age_hour,
            use_hdf5_cache=options.use_hdf5_cache,
        )
        task = BTypeBinaryTask(
            self.config,
            simu_name,
            primordial_pair_keys=self._primordial_pair_keys(primordial),
            primordial_signature=self._primordial_signature(simu_name),
        )
        return HDF5ScanJob(simu_name, task, options)

    def _load_primordial_binaries(
        self,
        simu_name: str,
        *,
        update: bool,
        wait_age_hour: int | float,
        use_hdf5_cache: bool,
    ) -> pd.DataFrame:
        identifier = PrimordialBinaryIdentifier(self.config)
        identifier.hdf5_file_processor = self.hdf5_file_processor
        return identifier.load_primordial_binaries(
            simu_name,
            update=update,
            wait_age_hour=wait_age_hour,
            use_hdf5_cache=use_hdf5_cache,
        )

    def _primordial_signature(self, simu_name: str) -> Dict[str, Any]:
        identifier = PrimordialBinaryIdentifier(self.config)
        meta_path = identifier._meta_path(simu_name)
        cache_path = identifier._cache_path(simu_name)
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read primordial binary metadata %s: %r", meta_path, exc)
        return {
            "cache_path": str(cache_path),
            "cache_mtime": cache_path.stat().st_mtime if cache_path.exists() else None,
            "meta_path": str(meta_path),
            "meta_mtime": meta_path.stat().st_mtime if meta_path.exists() else None,
            "meta": meta,
        }

    def _primordial_pair_keys(self, primordial: pd.DataFrame) -> set[str]:
        if "primordial_pair_key" in primordial.columns:
            return set(primordial["primordial_pair_key"].dropna().astype(str))
        if {"Bin Name1", "Bin Name2"}.issubset(primordial.columns):
            return set(_pair_keys(primordial["Bin Name1"], primordial["Bin Name2"]))
        return set()

    def _extraction_config(self) -> Dict[str, Any]:
        defaults = {}
        user_config = getattr(self.config, "binary_stellar_type_extraction", {}) or {}
        return {**defaults, **user_config}


class BTypeBinaryTask(FeatherMetaCacheMixin):
    """Scan task extracting binary rows with B-type main-sequence members."""

    schema_version = 1
    required_tables: Sequence[str] = ("scalars", "binaries")
    columns_by_table: Mapping[str, Sequence[str] | None] = {"scalars": ["TTOT"], "binaries": None}

    def __init__(
        self,
        config_manager: Any,
        simu_name: str,
        *,
        primordial_pair_keys: set[str],
        primordial_signature: Dict[str, Any],
    ) -> None:
        self.config = config_manager
        self.simu_name = simu_name
        self.primordial_pair_keys = primordial_pair_keys
        self.primordial_signature = primordial_signature
        self.name = "b_type_binary"

    @property
    def cache_path(self) -> Path:
        if hasattr(self, "_active_cache_path"):
            return self._active_cache_path
        existing = self._existing_cache_paths()
        if existing:
            return existing[-1]
        return self._cache_path_for_last_ttot(0.0)

    def is_file_fresh(self, hdf5_path: str, meta: Dict[str, Any], cache_df: pd.DataFrame) -> bool:
        return meta.get("primordial_signature") == self.primordial_signature and file_is_fresh(
            hdf5_path, meta
        )

    def process_file(
        self,
        hdf5_path: str,
        df_dict: Dict[str, pd.DataFrame],
        meta: Dict[str, Any],
        cache_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        binaries = df_dict.get("binaries", pd.DataFrame())
        if binaries.empty:
            rows = pd.DataFrame(columns=binaries.columns)
        else:
            missing = _required_binary_columns().difference(binaries.columns)
            if missing:
                raise ValueError(
                    "Binary table missing required columns for B-type binary extraction: "
                    + ", ".join(sorted(missing))
                )
            rows = self._matching_rows(binaries)
        return {"rows": rows, "file_meta": default_file_meta(hdf5_path, df_dict)}

    def merge_file_result(
        self, cache_df: pd.DataFrame, hdf5_path: str, result: Dict[str, Any]
    ) -> pd.DataFrame:
        new_df = result.get("rows", pd.DataFrame())
        ttot_values = result.get("file_meta", {}).get("ttot", [])
        if "TTOT" in cache_df.columns and ttot_values:
            cache_df = cache_df[~cache_df["TTOT"].astype(float).isin(ttot_values)]
        if new_df.empty:
            return cache_df.reset_index(drop=True)
        return pd.concat([cache_df, new_df], ignore_index=True, sort=False)

    def finalize_cache(self, cache_df: pd.DataFrame) -> pd.DataFrame:
        if cache_df.empty:
            return cache_df.reset_index(drop=True)
        sort_columns = [
            col
            for col in ["TTOT", "Time[Myr]", "Bin Name1", "Bin Name2"]
            if col in cache_df.columns
        ]
        if sort_columns:
            cache_df = cache_df.sort_values(sort_columns)
        return cache_df.reset_index(drop=True)

    def write_cache_and_meta(
        self,
        cache_df: pd.DataFrame,
        processed_files: Dict[str, Dict[str, Any]],
        options: HDF5ScanOptions,
    ) -> None:
        last_ttot = self._last_processed_ttot(cache_df, processed_files)
        self._active_cache_path = self._cache_path_for_last_ttot(last_ttot)
        super().write_cache_and_meta(cache_df, processed_files, options)
        for old_path in self._existing_cache_paths():
            if old_path != self._active_cache_path:
                old_meta = old_path.with_name(old_path.stem + ".meta.json")
                old_path.unlink(missing_ok=True)
                old_meta.unlink(missing_ok=True)

    def build_meta(
        self,
        cache_df: pd.DataFrame,
        processed_files: Dict[str, Dict[str, Any]],
        options: HDF5ScanOptions,
    ) -> Dict[str, Any]:
        meta = super().build_meta(cache_df, processed_files, options)
        meta.update(
            {
                "b_type_criteria": {
                    "kw": 1,
                    "teff_min": 10500,
                    "teff_max": 31500,
                    "mass_min": 2.75,
                    "mass_max": 17.7,
                },
                "primordial_signature": self.primordial_signature,
            }
        )
        return meta

    def _matching_rows(self, binaries: pd.DataFrame) -> pd.DataFrame:
        member1 = _is_b_type_member(binaries, "1")
        member2 = _is_b_type_member(binaries, "2")
        matching = member1 | member2
        rows = binaries.loc[matching].copy()
        if rows.empty:
            return rows

        rows["b_type_member1"] = member1.loc[matching].to_numpy(dtype=bool)
        rows["b_type_member2"] = member2.loc[matching].to_numpy(dtype=bool)
        rows["b_type_member_count"] = rows["b_type_member1"].astype(int) + rows[
            "b_type_member2"
        ].astype(int)
        rows["b_type_pair_key"] = _pair_keys(rows["Bin Name1"], rows["Bin Name2"])
        rows["is_primordial_binary"] = rows["b_type_pair_key"].isin(self.primordial_pair_keys)
        return rows

    def _cache_dir(self) -> Path:
        return analysis_cache_dir(self.config, self.simu_name, B_TYPE_BINARY_FEATURE)

    def _cache_path_for_last_ttot(self, last_ttot: float) -> Path:
        return self._cache_dir() / f"b_type_binaries_until_{last_ttot:.6f}.feather"

    def _existing_cache_paths(self) -> list[Path]:
        return sorted(
            self._cache_dir().glob("b_type_binaries_until_*.feather"),
            key=self._cache_sort_key,
        )

    def _cache_sort_key(self, path: Path) -> float:
        try:
            return float(path.stem.rsplit("_until_", 1)[1])
        except (IndexError, ValueError):
            return -1.0

    def _last_processed_ttot(
        self, cache_df: pd.DataFrame, processed_files: Dict[str, Dict[str, Any]]
    ) -> float:
        last_values = []
        if "TTOT" in cache_df.columns and not cache_df.empty:
            last_values.append(float(cache_df["TTOT"].max()))
        for file_meta in processed_files.values():
            last_values.extend(float(ttot) for ttot in file_meta.get("ttot", []))
        return max(last_values) if last_values else 0.0


def _required_binary_columns() -> set[str]:
    return {
        "Bin KW1",
        "Bin KW2",
        "Bin Teff1*",
        "Bin Teff2*",
        "Bin M1*",
        "Bin M2*",
        "Bin Name1",
        "Bin Name2",
    }


def _is_b_type_member(binaries: pd.DataFrame, suffix: str) -> pd.Series:
    return (
        (binaries[f"Bin KW{suffix}"].astype(int) == 1)
        & binaries[f"Bin Teff{suffix}*"].astype(float).between(10500, 31500, inclusive="both")
        & binaries[f"Bin M{suffix}*"].astype(float).between(2.75, 17.7, inclusive="both")
    )


def _pair_keys(name1: pd.Series, name2: pd.Series) -> list[str]:
    left = name1.astype(int)
    right = name2.astype(int)
    name_min = pd.concat([left, right], axis=1).min(axis=1).astype(int)
    name_max = pd.concat([left, right], axis=1).max(axis=1).astype(int)
    return [f"{min_id}-{max_id}" for min_id, max_id in zip(name_min, name_max)]
