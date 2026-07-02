"""Extract processed binary rows containing a target stellar type."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from dragon3_pipelines.analysis.cache_paths import BINARY_STELLAR_TYPE_FEATURE, analysis_cache_dir
from dragon3_pipelines.analysis.hdf5_scan import (
    FeatherMetaCacheMixin,
    HDF5ScanJob,
    HDF5ScanOptions,
    ScanBackedAnalysisBase,
    default_file_meta,
    file_is_fresh,
    replace_ttot_rows,
)

logger = logging.getLogger(__name__)


class BinaryStellarTypeExtractor(ScanBackedAnalysisBase):
    """Load processed binary rows where either component has a target KW code."""

    def load_binaries_with_stellar_type(
        self,
        simu_name: str,
        *,
        stellar_type: str | None = None,
        kw: int | str | None = None,
        update: bool = True,
        force: bool = False,
    ) -> pd.DataFrame:
        """Return complete processed binary rows matching one stellar type or KW code."""
        job = self.build_scan_job(
            simu_name,
            stellar_type=stellar_type,
            kw=kw,
            force=force,
        )
        return self._load_or_update_scan_job(job, update=update)

    def build_scan_job(
        self,
        simu_name: str,
        *,
        stellar_type: str | None = None,
        kw: int | str | None = None,
        force: bool = False,
    ) -> HDF5ScanJob:
        """Build a scan job for batched execution by ``HDF5ScanSession``."""
        target_kw, normalized_stellar_type = self.resolve_target(stellar_type=stellar_type, kw=kw)
        task = BinaryStellarTypeTask(
            self.config,
            simu_name,
            target_kw=target_kw,
            stellar_type=normalized_stellar_type,
        )
        options = self._scan_options(force=force)
        return HDF5ScanJob(simu_name, task, options)

    def resolve_target(
        self, stellar_type: str | None = None, kw: int | str | None = None
    ) -> tuple[int, str | None]:
        """Resolve exactly one of stellar_type or kw into a KW code."""
        if (stellar_type is None) == (kw is None):
            raise ValueError("Specify exactly one of stellar_type or kw.")
        if kw is not None:
            try:
                target_kw = int(kw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid KW code: {kw!r}") from exc
            if target_kw not in self.config.kw_to_stellar_type:
                raise ValueError(f"Unknown KW code: {target_kw}")
            return target_kw, self.config.kw_to_stellar_type.get(target_kw)

        normalized = str(stellar_type).strip().upper()
        type_to_kw = {
            str(key).upper(): int(value) for key, value in self.config.stellar_type_to_kw.items()
        }
        if normalized not in type_to_kw:
            raise ValueError(f"Unknown stellar_type: {stellar_type!r}")
        return type_to_kw[normalized], self.config.kw_to_stellar_type.get(type_to_kw[normalized])

    def _extraction_config(self) -> Dict[str, Any]:
        defaults = {
            "cache_filename_template": "binaries_with_{target}_until_{last_ttot:.6f}.feather",
        }
        user_config = getattr(self.config, "binary_stellar_type_extraction", {}) or {}
        return {**defaults, **user_config}


class BinaryStellarTypeTask(FeatherMetaCacheMixin):
    """Scan task extracting binary rows with either component matching target KW."""

    schema_version = 1
    required_tables: Sequence[str] = ("scalars", "binaries")
    columns_by_table: Mapping[str, Sequence[str] | None] = {"scalars": ["TTOT"], "binaries": None}

    def __init__(
        self,
        config_manager: Any,
        simu_name: str,
        *,
        target_kw: int,
        stellar_type: str | None,
    ) -> None:
        self.config = config_manager
        self.simu_name = simu_name
        self.target_kw = int(target_kw)
        self.stellar_type = stellar_type
        self.name = f"binary_stellar_type_{self.target_kw}"

    @property
    def cache_path(self) -> Path:
        if hasattr(self, "_active_cache_path"):
            return self._active_cache_path
        existing = self._existing_cache_paths()
        if existing:
            return existing[-1]
        return self._cache_path_for_last_ttot(0.0)

    def is_file_fresh(self, hdf5_path: str, meta: Dict[str, Any], cache_df: pd.DataFrame) -> bool:
        return file_is_fresh(hdf5_path, meta)

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
            required = {"Bin KW1", "Bin KW2"}
            missing = required.difference(binaries.columns)
            if missing:
                raise ValueError(
                    "Binary table missing required columns for stellar type extraction: "
                    + ", ".join(sorted(missing))
                )
            mask = (binaries["Bin KW1"].astype(int) == self.target_kw) | (
                binaries["Bin KW2"].astype(int) == self.target_kw
            )
            rows = binaries.loc[mask].copy()
        return {"rows": rows, "file_meta": default_file_meta(hdf5_path, df_dict)}

    def merge_file_result(
        self, cache_df: pd.DataFrame, hdf5_path: str, result: Dict[str, Any]
    ) -> pd.DataFrame:
        new_df = result.get("rows", pd.DataFrame())
        ttot_values = result.get("file_meta", {}).get("ttot", [])
        if "TTOT" in cache_df.columns and ttot_values:
            cache_df = cache_df[~cache_df["TTOT"].astype(float).isin(ttot_values)]
        return replace_ttot_rows(cache_df, new_df, "TTOT")

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
                "target_kw": self.target_kw,
                "target_stellar_type": self.stellar_type,
                "cache_filename_template": self._config()["cache_filename_template"],
            }
        )
        return meta

    def _config(self) -> Dict[str, Any]:
        defaults = {
            "cache_filename_template": "binaries_with_{target}_until_{last_ttot:.6f}.feather"
        }
        user_config = getattr(self.config, "binary_stellar_type_extraction", {}) or {}
        return {**defaults, **user_config}

    def _cache_dir(self) -> Path:
        return analysis_cache_dir(self.config, self.simu_name, BINARY_STELLAR_TYPE_FEATURE)

    def _cache_path_for_last_ttot(self, last_ttot: float) -> Path:
        pattern = self._config()["cache_filename_template"]
        return self._cache_dir() / pattern.format(
            target=self._target_label(),
            kw=self.target_kw,
            stellar_type=self.stellar_type or "KW",
            last_ttot=last_ttot,
        )

    def _existing_cache_paths(self) -> list[Path]:
        zero_name = self._cache_path_for_last_ttot(0.0).name
        glob_pattern = zero_name.replace("0.000000", "*")
        return sorted(self._cache_dir().glob(glob_pattern), key=self._cache_sort_key)

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

    def _target_label(self) -> str:
        if self.stellar_type:
            return f"{self.target_kw}_{self.stellar_type}"
        return str(self.target_kw)
