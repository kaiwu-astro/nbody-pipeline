"""Tests for the Parquet dataset/table cache mixins."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from nbody_pipeline.analysis.hdf5_scan import (
    HDF5ScanOptions,
    HDF5ScanRunner,
    default_file_meta,
    replace_ttot_rows,
)
from nbody_pipeline.analysis.parquet_cache import (
    ParquetDatasetCacheMixin,
    ParquetTableCacheMixin,
)
from nbody_pipeline.schemas import ColumnSchema, SchemaValidationError, TableSchema
from tests.test_hdf5_scan import FailingProcessor, FakeProcessor, FileBackedTask, make_config

TEST_SCHEMA = TableSchema(
    table="fake_object_rows",
    version=1,
    description="Synthetic schema for parquet cache mixin tests.",
    columns=(
        ColumnSchema(
            name="path",
            dtype="string",
            description="source hdf5 path",
            unit=None,
            ucd=None,
            public=False,
            nullable=False,
        ),
        ColumnSchema(
            name="ttot",
            dtype="float64",
            description="NB time",
            unit=None,
            ucd=None,
            public=True,
            nullable=False,
        ),
        ColumnSchema(
            name="value",
            dtype="float64",
            description="synthetic value",
            unit=None,
            ucd=None,
            public=True,
            nullable=False,
        ),
    ),
)


def _rows(hdf5_path: str, ttots: list[float], value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "path": pd.array([hdf5_path] * len(ttots), dtype="string"),
            "ttot": pd.array([float(t) for t in ttots], dtype="float64"),
            "value": pd.array([float(value)] * len(ttots), dtype="float64"),
        }
    )


class FakeDatasetTask(ParquetDatasetCacheMixin):
    name = "fake_dataset"
    required_tables = ("scalars",)
    columns_by_table = {"scalars": ["TTOT"]}
    schema_version = 1

    def __init__(self, cache_dir: Path):
        self._cache_dir = cache_dir

    @property
    def table_schema(self) -> TableSchema:
        return TEST_SCHEMA

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def process_file(self, hdf5_path, df_dict, meta, cache_df):
        ttots = df_dict["scalars"]["TTOT"].tolist()
        rows = _rows(hdf5_path, ttots, float(len(hdf5_path)))
        return {"rows": rows, "file_meta": default_file_meta(hdf5_path, df_dict)}


class ZeroRowDatasetTask(FakeDatasetTask):
    def process_file(self, hdf5_path, df_dict, meta, cache_df):
        return {
            "rows": TEST_SCHEMA.empty_dataframe(),
            "file_meta": default_file_meta(hdf5_path, df_dict),
        }


class FakeTableTask(ParquetTableCacheMixin):
    name = "fake_table"
    required_tables = ("scalars",)
    columns_by_table = {"scalars": ["TTOT"]}
    schema_version = 1

    def __init__(self, cache_path: Path):
        self._cache_path = cache_path

    @property
    def table_schema(self) -> TableSchema:
        return TEST_SCHEMA

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    def is_file_fresh(self, hdf5_path, meta, cache_df):
        from nbody_pipeline.analysis.hdf5_scan import file_is_fresh

        return file_is_fresh(hdf5_path, meta)

    def process_file(self, hdf5_path, df_dict, meta, cache_df):
        ttots = df_dict["scalars"]["TTOT"].tolist()
        rows = _rows(hdf5_path, ttots, float(len(hdf5_path)))
        return {"rows": rows, "file_meta": default_file_meta(hdf5_path, df_dict)}

    def merge_file_result(self, cache_df, hdf5_path, result):
        return replace_ttot_rows(cache_df, result["rows"], "ttot")


def _make_files(tmp_path: Path, count: int) -> tuple[list[str], dict]:
    paths = [str(tmp_path / f"snap.40_{idx}.h5part") for idx in range(count)]
    for path in paths:
        Path(path).write_text("fake")
    tables = {
        path: {"scalars": pd.DataFrame({"TTOT": [float(idx)]})} for idx, path in enumerate(paths)
    }
    return paths, tables


def test_dataset_cache_writes_one_part_per_file_and_manifest(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    paths, tables = _make_files(tmp_path, 3)
    processor = FakeProcessor(paths, tables)
    task = FakeDatasetTask(tmp_path / "cache" / "feature")
    runner = HDF5ScanRunner(config, processor)

    runner.run("sim", [task], HDF5ScanOptions(wait_age_hour=0))

    data_dir = task.data_dir
    parts = sorted(data_dir.glob("*.parquet"))
    assert len(parts) == 3
    combined = pd.read_parquet(data_dir)
    assert sorted(combined["ttot"].tolist()) == [0.0, 1.0, 2.0]
    manifest = json.loads(task.manifest_path.read_text())
    assert manifest["table_name"] == TEST_SCHEMA.table
    assert manifest["schema_hash"] == TEST_SCHEMA.schema_hash()
    assert set(manifest["processed_files"]) == set(paths)


def test_dataset_cache_incremental_rewrites_only_touched_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    paths, tables = _make_files(tmp_path, 3)
    cache_dir = tmp_path / "cache" / "feature"

    HDF5ScanRunner(config, FakeProcessor(paths, tables)).run(
        "sim", [FakeDatasetTask(cache_dir)], HDF5ScanOptions(wait_age_hour=0)
    )

    unchanged_processor = FakeProcessor(paths, tables)
    HDF5ScanRunner(config, unchanged_processor).run(
        "sim", [FakeDatasetTask(cache_dir)], HDF5ScanOptions(wait_age_hour=0)
    )
    assert unchanged_processor.read_count == 0

    touched_path = paths[-1]
    tables[touched_path] = {"scalars": pd.DataFrame({"TTOT": [99.0]})}
    os.utime(touched_path, None)
    touched_processor = FakeProcessor(paths, tables)
    HDF5ScanRunner(config, touched_processor).run(
        "sim", [FakeDatasetTask(cache_dir)], HDF5ScanOptions(wait_age_hour=0)
    )
    assert touched_processor.read_paths == [touched_path]

    combined = pd.read_parquet(cache_dir / "data")
    touched_rows = combined[combined["path"] == touched_path]
    assert touched_rows["ttot"].tolist() == [99.0]
    other_rows = combined[combined["path"] != touched_path]
    assert sorted(other_rows["ttot"].tolist()) == [0.0, 1.0]


def test_dataset_cache_treats_missing_part_as_stale(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    paths, tables = _make_files(tmp_path, 1)
    cache_dir = tmp_path / "cache" / "feature"
    HDF5ScanRunner(config, FakeProcessor(paths, tables)).run(
        "sim", [FakeDatasetTask(cache_dir)], HDF5ScanOptions(wait_age_hour=0)
    )
    for part in (cache_dir / "data").glob("*.parquet"):
        part.unlink()

    processor = FakeProcessor(paths, tables)
    HDF5ScanRunner(config, processor).run(
        "sim", [FakeDatasetTask(cache_dir)], HDF5ScanOptions(wait_age_hour=0)
    )

    assert processor.read_count == 1
    assert list((cache_dir / "data").glob("*.parquet"))


def test_dataset_cache_schema_version_bump_triggers_full_rebuild(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    paths, tables = _make_files(tmp_path, 2)
    cache_dir = tmp_path / "cache" / "feature"
    HDF5ScanRunner(config, FakeProcessor(paths, tables)).run(
        "sim", [FakeDatasetTask(cache_dir)], HDF5ScanOptions(wait_age_hour=0)
    )

    rebuild_calls = []

    class FakeDatasetTaskV2(FakeDatasetTask):
        schema_version = 2

        def prepare_full_rebuild(self):
            rebuild_calls.append(True)
            super().prepare_full_rebuild()

    processor = FakeProcessor(paths, tables)
    task_v2 = FakeDatasetTaskV2(cache_dir)
    HDF5ScanRunner(config, processor).run("sim", [task_v2], HDF5ScanOptions(wait_age_hour=0))

    assert rebuild_calls == [True]
    assert processor.read_count == 2
    manifest = json.loads(task_v2.manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert set(manifest["processed_files"]) == set(paths)


def test_dataset_cache_prunes_orphan_parts_and_tmp_files(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    paths, tables = _make_files(tmp_path, 1)
    cache_dir = tmp_path / "cache" / "feature"
    task = FakeDatasetTask(cache_dir)
    HDF5ScanRunner(config, FakeProcessor(paths, tables)).run(
        "sim", [task], HDF5ScanOptions(wait_age_hour=0)
    )
    (task.data_dir / "part-orphan-deadbeef.parquet").write_bytes(b"junk")
    (task.data_dir / "part-orphan-deadbeef.parquet.tmp").write_bytes(b"junk")

    manifest = json.loads(task.manifest_path.read_text())
    task.write_cache_and_meta(task.read_cache(), manifest["processed_files"], HDF5ScanOptions())

    remaining = {p.name for p in task.data_dir.iterdir()}
    assert "part-orphan-deadbeef.parquet" not in remaining
    assert "part-orphan-deadbeef.parquet.tmp" not in remaining


def test_dataset_cache_mid_run_exception_keeps_manifest_consistent(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    paths, tables = _make_files(tmp_path, 2)
    cache_dir = tmp_path / "cache" / "feature"
    task = FakeDatasetTask(cache_dir)
    runner = HDF5ScanRunner(config, FailingProcessor(paths, tables, {paths[1]}))

    with pytest.raises(RuntimeError, match="failed to read"):
        runner.run("sim", [task], HDF5ScanOptions(wait_age_hour=0, checkpoint_every_files=100))

    manifest = json.loads(task.manifest_path.read_text())
    assert set(manifest["processed_files"]) == {paths[0]}
    assert len(list(task.data_dir.glob("*.parquet"))) == 1

    resumed_task = FakeDatasetTask(cache_dir)
    resumed_processor = FakeProcessor(paths, tables)
    HDF5ScanRunner(config, resumed_processor).run(
        "sim", [resumed_task], HDF5ScanOptions(wait_age_hour=0, checkpoint_every_files=100)
    )
    assert resumed_processor.read_paths == [paths[1]]
    resumed_manifest = json.loads(resumed_task.manifest_path.read_text())
    assert set(resumed_manifest["processed_files"]) == set(paths)


def test_dataset_cache_writes_empty_typed_part_for_zero_rows(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    paths, tables = _make_files(tmp_path, 1)
    cache_dir = tmp_path / "cache" / "feature"
    task = ZeroRowDatasetTask(cache_dir)

    HDF5ScanRunner(config, FakeProcessor(paths, tables)).run(
        "sim", [task], HDF5ScanOptions(wait_age_hour=0)
    )

    parts = list(task.data_dir.glob("*.parquet"))
    assert len(parts) == 1
    df = pd.read_parquet(parts[0])
    assert df.empty
    assert list(df.columns) == list(TEST_SCHEMA.column_names())


def test_dataset_cache_merge_file_result_rejects_schema_violation(tmp_path: Path) -> None:
    task = FakeDatasetTask(tmp_path / "cache" / "feature")
    bad_rows = pd.DataFrame({"path": ["x"], "ttot": [1.0]})

    with pytest.raises(SchemaValidationError):
        task.merge_file_result(task.read_cache(), "somepath", {"rows": bad_rows, "file_meta": {}})


def test_table_cache_round_trip_and_ttot_replace_merge(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    paths, tables = _make_files(tmp_path, 2)
    cache_path = tmp_path / "cache" / "table_feature.parquet"
    task = FakeTableTask(cache_path)

    HDF5ScanRunner(config, FakeProcessor(paths, tables)).run(
        "sim", [task], HDF5ScanOptions(wait_age_hour=0)
    )

    df = pd.read_parquet(cache_path)
    assert sorted(df["ttot"].tolist()) == [0.0, 1.0]

    touched_path = paths[-1]
    tables[touched_path] = {"scalars": pd.DataFrame({"TTOT": [1.0]})}
    os.utime(touched_path, None)
    task2 = FakeTableTask(cache_path)
    processor2 = FakeProcessor(paths, tables)
    HDF5ScanRunner(config, processor2).run("sim", [task2], HDF5ScanOptions(wait_age_hour=0))

    assert processor2.read_paths == [touched_path]
    df2 = pd.read_parquet(cache_path)
    assert len(df2) == 2
    assert sorted(df2["ttot"].tolist()) == [0.0, 1.0]
    replaced_row = df2[(df2["path"] == touched_path) & (df2["ttot"] == 1.0)]
    assert replaced_row["value"].iloc[0] == float(len(touched_path))


def test_table_cache_write_rejects_schema_violation(tmp_path: Path) -> None:
    task = FakeTableTask(tmp_path / "cache" / "bad.parquet")
    bad_df = pd.DataFrame({"path": ["x"]})

    with pytest.raises(SchemaValidationError):
        task.write_cache_and_meta(bad_df, {}, HDF5ScanOptions())


def test_feather_and_parquet_tasks_share_one_read_per_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    Path(hdf5_path).write_text("fake")
    tables = {
        hdf5_path: {
            "scalars": pd.DataFrame({"TTOT": [1.0]}),
            "binaries": pd.DataFrame({"TTOT": [1.0]}),
        }
    }
    processor = FakeProcessor([hdf5_path], tables)
    feather_task = FileBackedTask("feather_task", tmp_path / "feather_cache")
    parquet_task = FakeDatasetTask(tmp_path / "cache" / "feature")
    runner = HDF5ScanRunner(config, processor)

    runner.run("sim", [feather_task, parquet_task], HDF5ScanOptions(wait_age_hour=0))

    assert processor.read_count == 1
    assert (tmp_path / "feather_cache" / "feather_task.feather").exists()
    assert list((tmp_path / "cache" / "feature" / "data").glob("*.parquet"))
