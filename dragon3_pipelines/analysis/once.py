"""Helpers for one-shot analysis caches."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from dragon3_pipelines.analysis.cache_paths import AnalysisCacheFeature, analysis_cache_dir

logger = logging.getLogger(__name__)


class SimulationOnceAnalysisBase:
    """Base class for per-simulation analysis results that do not auto-refresh."""

    def __init__(
        self,
        config_manager: Any,
        *,
        feature: AnalysisCacheFeature,
        cache_filename: str,
        meta_filename: str,
    ) -> None:
        self.config = config_manager
        self.feature = feature
        self.cache_filename = cache_filename
        self.meta_filename = meta_filename

    def load_or_compute(
        self,
        simu_name: str,
        *,
        update: bool,
        force: bool,
        meta: Mapping[str, Any] | None = None,
        compute: Callable[[], pd.DataFrame | tuple[pd.DataFrame, Mapping[str, Any]]],
    ) -> pd.DataFrame:
        """Return cached data unless recomputation is required."""
        if not update:
            return self._read_cache(simu_name)
        if self._cache_path(simu_name).exists() and not force:
            return self._read_cache(simu_name)

        result = compute()
        if isinstance(result, tuple):
            df, computed_meta = result
            cache_meta = dict(computed_meta)
        else:
            df = result
            cache_meta = dict(meta or {})
        self._write_cache_and_meta(simu_name, df, cache_meta)
        return df

    def _cache_dir(self, simu_name: str) -> Path:
        return analysis_cache_dir(self.config, simu_name, self.feature)

    def _cache_path(self, simu_name: str) -> Path:
        return self._cache_dir(simu_name) / self.cache_filename

    def _meta_path(self, simu_name: str) -> Path:
        return self._cache_dir(simu_name) / self.meta_filename

    def _read_cache(self, simu_name: str) -> pd.DataFrame:
        cache_path = self._cache_path(simu_name)
        if not cache_path.exists():
            return pd.DataFrame()
        return pd.read_feather(cache_path)

    def _read_meta(self, simu_name: str) -> dict[str, Any]:
        meta_path = self._meta_path(simu_name)
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read analysis metadata %s: %r", meta_path, exc)
            return {}

    def _write_cache_and_meta(
        self, simu_name: str, df: pd.DataFrame, meta: Mapping[str, Any]
    ) -> None:
        cache_dir = self._cache_dir(simu_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(simu_name)
        meta_path = self._meta_path(simu_name)
        tmp_cache_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_meta_path = meta_path.with_suffix(meta_path.suffix + ".tmp")

        df.to_feather(tmp_cache_path)
        os.replace(tmp_cache_path, cache_path)
        tmp_meta_path.write_text(json.dumps(dict(meta), indent=2, sort_keys=True))
        os.replace(tmp_meta_path, meta_path)
