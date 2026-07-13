"""Tests for analysis_cache_dir's lake-vs-analysis-cache routing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from nbody_pipeline.analysis.cache_paths import (
    COMPACT_OBJECT_HISTORY_FEATURE,
    LAKE_FEATURES,
    SNAPSHOT_SCALARS_FEATURE,
    SNAPSHOT_SINGLES_FEATURE,
    analysis_cache_dir,
)


def test_lake_features_contains_exactly_the_four_lake_tables() -> None:
    assert LAKE_FEATURES == {
        "snapshot_singles",
        "snapshot_binaries",
        "snapshot_mergers",
        "snapshot_scalars",
    }


def test_lake_feature_routes_to_lake_dir_when_configured(tmp_path: Path) -> None:
    config = Mock()
    config.analysis_cache_dir_of = {"sim": str(tmp_path / "cache" / "sim")}
    config.lake_dir_of = {"sim": str(tmp_path / "lake" / "sim")}

    path = analysis_cache_dir(config, "sim", SNAPSHOT_SINGLES_FEATURE)

    assert path == tmp_path / "lake" / "sim" / SNAPSHOT_SINGLES_FEATURE


def test_lake_feature_falls_back_to_analysis_cache_dir_when_lake_dir_unset(
    tmp_path: Path,
) -> None:
    config = Mock()
    config.analysis_cache_dir_of = {"sim": str(tmp_path / "cache" / "sim")}
    config.lake_dir_of = {}

    path = analysis_cache_dir(config, "sim", SNAPSHOT_SCALARS_FEATURE)

    assert path == tmp_path / "cache" / "sim" / SNAPSHOT_SCALARS_FEATURE


def test_non_lake_feature_never_routes_to_lake_dir(tmp_path: Path) -> None:
    config = Mock()
    config.analysis_cache_dir_of = {"sim": str(tmp_path / "cache" / "sim")}
    config.lake_dir_of = {"sim": str(tmp_path / "lake" / "sim")}

    path = analysis_cache_dir(config, "sim", COMPACT_OBJECT_HISTORY_FEATURE)

    assert path == tmp_path / "cache" / "sim" / COMPACT_OBJECT_HISTORY_FEATURE


def test_lake_feature_with_missing_lake_dir_of_attribute_falls_back(tmp_path: Path) -> None:
    """Legacy/lightweight configs (e.g. Mock() without lake_dir_of set) must still work."""
    config = Mock(spec=["analysis_cache_dir_of"])
    config.analysis_cache_dir_of = {"sim": str(tmp_path / "cache" / "sim")}

    path = analysis_cache_dir(config, "sim", SNAPSHOT_SINGLES_FEATURE)

    assert path == tmp_path / "cache" / "sim" / SNAPSHOT_SINGLES_FEATURE
