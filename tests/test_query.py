"""Tests for the DuckDB query entry point over the Parquet feature store."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from dragon3_pipelines.query import (
    PARQUET_FEATURES,
    duckdb_connect,
    feature_dataset_glob,
    load_feature,
)
from dragon3_pipelines.schemas import TableSchema, load_table_schema


def _row_dataframe(schema: TableSchema, values: dict) -> pd.DataFrame:
    data = {
        column.name: pd.array([values.get(column.name)], dtype=column.dtype)
        for column in schema.columns
    }
    return pd.DataFrame(data, columns=schema.column_names())


def _compact_object_row(simulation_id: str, object_id: int, mass_msun: float, ttot: float) -> dict:
    return {
        "simulation_id": simulation_id,
        "ttot": ttot,
        "time_myr": ttot * 0.1,
        "object_id": object_id,
        "kw": 14,
        "object_type": "single",
        "mass_msun": mass_msun,
        "x_pc": 1.0,
        "y_pc": 2.0,
        "z_pc": 3.0,
        "vx_kms": 0.1,
        "vy_kms": 0.2,
        "vz_kms": 0.3,
        "r_pc": 3.7,
        "is_in_binary": False,
        "source_hdf5_path": f"/fake/{simulation_id}.h5part",
    }


def _write_dataset_feature(config: Mock, simu_name: str, feature: str, rows: list[dict]) -> None:
    """Write a dataset-style (per-file part) feature, mirroring ParquetDatasetCacheMixin."""
    schema = load_table_schema(feature)
    feature_dir = Path(config.analysis_cache_dir_of[simu_name]) / feature
    data_dir = feature_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        _row_dataframe(schema, row).to_parquet(data_dir / f"part-{index}.parquet", index=False)


def _write_table_feature(config: Mock, simu_name: str, feature: str, rows: list[dict]) -> None:
    """Write a table-style (single merged file) feature, mirroring ParquetTableCacheMixin."""
    schema = load_table_schema(feature)
    frames = [_row_dataframe(schema, row) for row in rows]
    df = pd.concat(frames, ignore_index=True) if frames else schema.empty_dataframe()
    feature_dir = Path(config.analysis_cache_dir_of[simu_name]) / feature
    feature_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feature_dir / f"{feature}.parquet", index=False)


def _make_config(tmp_path: Path, simu_names: list[str]) -> Mock:
    config = Mock()
    config.pathof = {name: str(tmp_path / name) for name in simu_names}
    config.analysis_cache_dir_of = {name: str(tmp_path / "cache" / name) for name in simu_names}
    return config


def test_parquet_features_lists_both_pilots() -> None:
    assert set(PARQUET_FEATURES) == {"compact_object_history", "snapshot_summary"}


def test_feature_dataset_glob_dataset_style(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["simA"])
    _write_dataset_feature(
        config, "simA", "compact_object_history", [_compact_object_row("simA", 1, 20.0, 1.0)]
    )
    glob = feature_dataset_glob(config, "simA", "compact_object_history")
    assert glob.endswith("data/*.parquet")


def test_feature_dataset_glob_table_style(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["simA"])
    _write_table_feature(
        config,
        "simA",
        "snapshot_summary",
        [
            {
                "simulation_id": "simA",
                "ttot": 1.0,
                "time_myr": 0.1,
                "n_singles": 1,
                "n_binaries": 0,
                "n_bh": 0,
                "n_ns": 0,
                "n_wd": 0,
                "total_mass_msun": 1.0,
                "core_radius_pc": 0.1,
                "half_mass_radius_pc": 0.2,
                "rg_x_pc": 0.0,
                "rg_y_pc": 0.0,
                "rg_z_pc": 0.0,
                "vg_x_kmps": 0.0,
                "vg_y_kmps": 0.0,
                "vg_z_kmps": 0.0,
            }
        ],
    )
    glob = feature_dataset_glob(config, "simA", "snapshot_summary")
    assert glob.endswith("snapshot_summary.parquet")
    assert "/data/" not in glob


def test_feature_dataset_glob_missing_raises_friendly_error(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["simA"])
    with pytest.raises(FileNotFoundError, match="analyze"):
        feature_dataset_glob(config, "simA", "compact_object_history")


def test_load_feature_full_read(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["simA"])
    _write_dataset_feature(
        config,
        "simA",
        "compact_object_history",
        [
            _compact_object_row("simA", 1, 20.0, 1.0),
            _compact_object_row("simA", 2, 30.0, 2.0),
        ],
    )
    df = load_feature(config, "simA", "compact_object_history")
    assert sorted(df["object_id"].tolist()) == [1, 2]


def test_load_feature_column_projection(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["simA"])
    _write_dataset_feature(
        config, "simA", "compact_object_history", [_compact_object_row("simA", 1, 20.0, 1.0)]
    )
    df = load_feature(config, "simA", "compact_object_history", columns=["object_id", "mass_msun"])
    assert list(df.columns) == ["object_id", "mass_msun"]


def test_load_feature_unknown_column_raises(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["simA"])
    _write_dataset_feature(
        config, "simA", "compact_object_history", [_compact_object_row("simA", 1, 20.0, 1.0)]
    )
    with pytest.raises(ValueError, match="Unknown column"):
        load_feature(config, "simA", "compact_object_history", columns=["not_a_column"])


def test_load_feature_where_params(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["simA"])
    _write_dataset_feature(
        config,
        "simA",
        "compact_object_history",
        [
            _compact_object_row("simA", 1, 20.0, 1.0),
            _compact_object_row("simA", 2, 30.0, 2.0),
        ],
    )
    df = load_feature(
        config, "simA", "compact_object_history", where="mass_msun > ?", params=[25.0]
    )
    assert df["object_id"].tolist() == [2]


def test_duckdb_connect_unions_across_simulations(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["simA", "simB"])
    _write_dataset_feature(
        config, "simA", "compact_object_history", [_compact_object_row("simA", 1, 20.0, 1.0)]
    )
    _write_dataset_feature(
        config, "simB", "compact_object_history", [_compact_object_row("simB", 2, 40.0, 1.0)]
    )
    connection = duckdb_connect(config, features=["compact_object_history"])
    try:
        df = connection.execute(
            "SELECT simulation_id, object_id FROM compact_object_history ORDER BY simulation_id"
        ).df()
    finally:
        connection.close()
    assert df["simulation_id"].tolist() == ["simA", "simB"]
    assert df["object_id"].tolist() == [1, 2]


def test_duckdb_connect_skips_missing_feature(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["simA", "simB"])
    _write_dataset_feature(
        config, "simA", "compact_object_history", [_compact_object_row("simA", 1, 20.0, 1.0)]
    )
    # simB has no compact_object_history data on disk yet.
    connection = duckdb_connect(config, features=["compact_object_history"])
    try:
        df = connection.execute("SELECT simulation_id FROM compact_object_history").df()
    finally:
        connection.close()
    assert df["simulation_id"].tolist() == ["simA"]


def test_duckdb_connect_skips_feature_with_no_data_anywhere(tmp_path: Path) -> None:
    config = _make_config(tmp_path, ["simA"])
    connection = duckdb_connect(config, features=["compact_object_history", "snapshot_summary"])
    try:
        tables = connection.execute("SHOW TABLES").df()["name"].tolist()
    finally:
        connection.close()
    assert tables == []
