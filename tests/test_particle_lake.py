"""Tests for the full particle-lake scan tasks (snapshot_{singles,binaries,mergers,scalars})."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from nbody_pipeline.analysis.cache_paths import (
    SNAPSHOT_BINARIES_FEATURE,
    SNAPSHOT_MERGERS_FEATURE,
    SNAPSHOT_SINGLES_FEATURE,
    analysis_cache_dir,
)
from nbody_pipeline.analysis.hdf5_scan import HDF5ScanOptions, HDF5ScanRunner
from nbody_pipeline.analysis.particle_lake import (
    ParticleLakeProcessor,
    SnapshotBinariesTask,
    SnapshotMergersTask,
    SnapshotSinglesTask,
    compute_ttot_dedup_exclusions,
)
from nbody_pipeline.io.text_parsers import SCALAR_KEYS
from nbody_pipeline.schemas import load_table_schema
from tests.test_hdf5_scan import make_config

RBAR = 2.0
TSCALE = 0.1
RDENS = (1.0, 0.5, -0.5)


def _scalars_df(ttot: float, **overrides: float) -> pd.DataFrame:
    data = {key: [0.0] for key in SCALAR_KEYS}
    data["TTOT"] = [ttot]
    data["RBAR"] = [RBAR]
    data["ZMBAR"] = [1.0e6]
    data["VSTAR"] = [10.0]
    data["TSCALE"] = [TSCALE]
    data["RDENS(1)"], data["RDENS(2)"], data["RDENS(3)"] = [RDENS[0]], [RDENS[1]], [RDENS[2]]
    for key, value in overrides.items():
        data[key] = [value]
    return pd.DataFrame(data)


def _singles_df(ttot: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TTOT": [ttot, ttot],
            "Name": pd.array([1, 2], dtype="int32"),
            "KW": pd.array([1, 14], dtype="int32"),
            "Type": pd.array([0, 0], dtype="int32"),
            "M": pd.array([1.0, 20.0], dtype="float32"),
            "X1": pd.array([5.0, 10.0], dtype="float32"),
            "X2": pd.array([1.0, 2.0], dtype="float32"),
            "X3": pd.array([1.0, -1.0], dtype="float32"),
            "V1": pd.array([1.0, 2.0], dtype="float32"),
            "V2": pd.array([0.0, 1.0], dtype="float32"),
            "V3": pd.array([0.0, 0.0], dtype="float32"),
            "POT": pd.array([-2.0, -5.0], dtype="float32"),
            "R*": pd.array([1.0, 0.0001], dtype="float32"),
            # Deliberately non-physical BH luminosity/temperature (row 2, KW=14):
            # the raw reader must NOT clip these to the display placeholder that
            # nbody_pipeline.io.hdf5_reader.read_file substitutes for NS/BH.
            "L*": pd.array([1.0, 999.0], dtype="float32"),
            "Teff*": pd.array([5000.0, 123456.0], dtype="float32"),
            "RC*": pd.array([0.1, 0.0], dtype="float32"),
            "MC*": pd.array([0.05, 0.0], dtype="float32"),
            "ASPN": pd.array([0.01, 0.5], dtype="float32"),
            "EPOCH": pd.array([100.0, 200.0], dtype="float32"),
        }
    )


def _binaries_df(ttot: float) -> pd.DataFrame:
    columns = SnapshotBinariesTask.columns_by_table["binaries"]
    data = {col: pd.array([0.0], dtype="float32") for col in columns}
    data["TTOT"] = [ttot]
    data["Bin Name1"] = pd.array([10], dtype="int32")
    data["Bin Name2"] = pd.array([11], dtype="int32")
    data["Bin cm Name"] = pd.array([100], dtype="int32")
    data["Bin KW1"] = pd.array([13], dtype="int32")
    data["Bin KW2"] = pd.array([1], dtype="int32")
    data["Bin cm KW"] = pd.array([0], dtype="int32")
    data["Bin Label"] = pd.array([1], dtype="int32")
    data["Bin M1*"] = pd.array([1.4], dtype="float32")
    data["Bin M2*"] = pd.array([1.0], dtype="float32")
    data["Bin cm X1"] = pd.array([6.0], dtype="float32")
    data["Bin cm X2"] = pd.array([1.5], dtype="float32")
    data["Bin cm X3"] = pd.array([0.5], dtype="float32")
    data["Bin A[au]"] = pd.array([10.0], dtype="float32")
    data["Bin ECC"] = pd.array([0.1], dtype="float32")
    return pd.DataFrame(data)


def _mergers_df(ttot: float) -> pd.DataFrame:
    columns = SnapshotMergersTask.columns_by_table["mergers"]
    data = {col: pd.array([0.0], dtype="float32") for col in columns}
    data["TTOT"] = [ttot]
    data["Mer NAM1"] = pd.array([20], dtype="int32")
    data["Mer NAM2"] = pd.array([21], dtype="int32")
    data["Mer NAM3"] = pd.array([0], dtype="int32")
    data["Mer NAMC"] = pd.array([200], dtype="int32")
    data["Mer KW1"] = pd.array([14], dtype="int32")
    data["Mer KW2"] = pd.array([14], dtype="int32")
    data["Mer M1"] = pd.array([15.0], dtype="float32")
    data["Mer M2"] = pd.array([16.0], dtype="float32")
    data["Mer XC1"] = pd.array([7.0], dtype="float32")
    data["Mer XC2"] = pd.array([2.0], dtype="float32")
    data["Mer XC3"] = pd.array([1.0], dtype="float32")
    return pd.DataFrame(data)


def _lake_tables(hdf5_path: str, *, ttot: float = 5.0) -> dict[str, dict[str, pd.DataFrame]]:
    return {
        hdf5_path: {
            "scalars": _scalars_df(ttot),
            "singles": _singles_df(ttot),
            "binaries": _binaries_df(ttot),
            "mergers": _mergers_df(ttot),
        }
    }


class FakeRawProcessor:
    """Mirrors tests.test_hdf5_scan.FakeProcessor but for the raw reader path."""

    def __init__(self, hdf5_paths: list[str], tables_by_path: dict[str, dict[str, pd.DataFrame]]):
        self.hdf5_paths = hdf5_paths
        self.tables_by_path = tables_by_path
        self.read_count = 0
        self.read_paths: list[str] = []

    def get_all_hdf5_paths(self, *args, **kwargs):
        return self.hdf5_paths

    def read_raw_tables(self, hdf5_path, tables, columns_by_table=None):
        self.read_count += 1
        self.read_paths.append(hdf5_path)
        return {table: self.tables_by_path[hdf5_path][table] for table in tables}

    def read_step_times(self, hdf5_path):
        return self.tables_by_path[hdf5_path]["scalars"]["TTOT"].tolist()


def _run_lake_tasks(tmp_path: Path, *, ttot: float = 5.0):
    config = make_config(tmp_path)
    config.lake_dir_of = {"sim": str(tmp_path / "lake" / "sim")}
    config.particle_lake = {"enabled": True, "scan": {}}
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    Path(hdf5_path).write_text("fake")
    processor = FakeRawProcessor([hdf5_path], _lake_tables(hdf5_path, ttot=ttot))

    lake = ParticleLakeProcessor(config, processor)
    jobs = lake.build_scan_jobs("sim")
    runner = HDF5ScanRunner(config, processor)
    outputs = runner.run("sim", [job.task for job in jobs], jobs[0].options)
    return config, processor, outputs, hdf5_path


def test_build_scan_jobs_share_one_options_instance(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.lake_dir_of = {"sim": str(tmp_path / "lake" / "sim")}
    config.particle_lake = {"enabled": True, "scan": {"sample_every_nb_time": None}}
    lake = ParticleLakeProcessor(config)

    jobs = lake.build_scan_jobs("sim")

    assert [job.task.name for job in jobs] == [
        "snapshot_singles",
        "snapshot_binaries",
        "snapshot_mergers",
        "snapshot_scalars",
    ]
    assert len({job.options for job in jobs}) == 1
    assert jobs[0].options.sample_every_nb_time is None
    assert all(job.task.hdf5_reader_kind == "raw" for job in jobs)


def test_singles_task_rdens_correction_and_no_luminosity_clipping(tmp_path: Path) -> None:
    config, processor, _outputs, _path = _run_lake_tasks(tmp_path)

    data_dir = analysis_cache_dir(config, "sim", SNAPSHOT_SINGLES_FEATURE) / "data"
    rows = pd.read_parquet(data_dir)
    schema = load_table_schema("snapshot_singles")
    schema.validate_dataframe(rows[list(schema.column_names())])

    assert processor.read_count == 1
    assert len(rows) == 2
    assert set(rows.columns) == set(schema.column_names())

    star = rows.loc[rows["object_id"] == 1].iloc[0]
    assert star["x_pc"] == pytest.approx(5.0 - RDENS[0] * RBAR)
    assert star["y_pc"] == pytest.approx(1.0 - RDENS[1] * RBAR)
    assert star["z_pc"] == pytest.approx(1.0 - RDENS[2] * RBAR)
    assert star["r_pc"] == pytest.approx(
        (star["x_pc"] ** 2 + star["y_pc"] ** 2 + star["z_pc"] ** 2) ** 0.5
    )
    assert star["time_myr"] == pytest.approx(5.0 * TSCALE)

    bh = rows.loc[rows["object_id"] == 2].iloc[0]
    assert bh["kw"] == 14
    # Untouched source values -- not the display-clipping constant that
    # nbody_pipeline.io.hdf5_reader.read_file substitutes for NS/BH.
    assert bh["luminosity_lsun"] == pytest.approx(999.0)
    assert bh["teff_k"] == pytest.approx(123456.0)


def test_binaries_task_columns_and_rdens_correction(tmp_path: Path) -> None:
    config, _processor, _outputs, _path = _run_lake_tasks(tmp_path)

    data_dir = analysis_cache_dir(config, "sim", SNAPSHOT_BINARIES_FEATURE) / "data"
    rows = pd.read_parquet(data_dir)
    schema = load_table_schema("snapshot_binaries")
    schema.validate_dataframe(rows[list(schema.column_names())])

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["object_id_1"] == 10
    assert row["object_id_2"] == 11
    assert row["cm_id"] == 100
    assert row["bin_label"] == 1
    assert row["kw_1"] == 13
    assert row["cm_x_pc"] == pytest.approx(6.0 - RDENS[0] * RBAR)
    assert row["semi_major_axis_au"] == pytest.approx(10.0)


def test_mergers_task_columns(tmp_path: Path) -> None:
    config, _processor, _outputs, _path = _run_lake_tasks(tmp_path)

    data_dir = analysis_cache_dir(config, "sim", SNAPSHOT_MERGERS_FEATURE) / "data"
    rows = pd.read_parquet(data_dir)
    schema = load_table_schema("snapshot_mergers")
    schema.validate_dataframe(rows[list(schema.column_names())])

    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["object_id_1"] == 20
    assert row["object_id_2"] == 21
    assert row["cm_id"] == 200
    assert row["cm_x_pc"] == pytest.approx(7.0 - RDENS[0] * RBAR)


def test_scalars_task_one_row_per_ttot(tmp_path: Path) -> None:
    _config, _processor, outputs, _path = _run_lake_tasks(tmp_path)

    scalars = outputs["snapshot_scalars"]
    schema = load_table_schema("snapshot_scalars")
    schema.validate_dataframe(scalars)

    assert len(scalars) == 1
    row = scalars.iloc[0]
    assert row["ttot"] == 5.0
    assert row["rbar_pc"] == pytest.approx(RBAR)
    assert row["tscale_myr"] == pytest.approx(TSCALE)
    assert row["time_myr"] == pytest.approx(5.0 * TSCALE)


def test_dataset_tasks_use_worker_direct_write_shape(tmp_path: Path) -> None:
    """process_file must return the {'part', 'row_count', ...} shape, not 'rows'."""
    config = make_config(tmp_path)
    config.lake_dir_of = {"sim": str(tmp_path / "lake" / "sim")}
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    Path(hdf5_path).write_text("fake")
    tables = _lake_tables(hdf5_path)[hdf5_path]
    task = SnapshotSinglesTask(config, "sim")

    result = task.process_file(hdf5_path, tables, meta={}, cache_df=task.read_cache())

    assert "part" in result
    assert "rows" not in result
    assert result["row_count"] == 2
    part_path = task.data_dir / result["part"]
    assert part_path.exists()


def test_empty_source_tables_produce_empty_parts_with_correct_schema(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.lake_dir_of = {"sim": str(tmp_path / "lake" / "sim")}
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    Path(hdf5_path).write_text("fake")
    tables = {
        "scalars": _scalars_df(5.0),
        "singles": pd.DataFrame(),
        "binaries": pd.DataFrame(),
        "mergers": pd.DataFrame(),
    }
    processor = FakeRawProcessor([hdf5_path], {hdf5_path: tables})

    for task_cls, feature in (
        (SnapshotSinglesTask, SNAPSHOT_SINGLES_FEATURE),
        (SnapshotBinariesTask, SNAPSHOT_BINARIES_FEATURE),
        (SnapshotMergersTask, SNAPSHOT_MERGERS_FEATURE),
    ):
        task = task_cls(config, "sim")
        runner = HDF5ScanRunner(config, processor)
        runner.run("sim", [task], HDF5ScanOptions(wait_age_hour=0))
        data_dir = analysis_cache_dir(config, "sim", feature) / "data"
        rows = pd.read_parquet(data_dir)
        assert rows.empty
        load_table_schema(task.table_schema.table).validate_dataframe(rows)


def test_freshness_skips_reprocessing_and_force_rebuilds(tmp_path: Path) -> None:
    config, processor, _outputs, hdf5_path = _run_lake_tasks(tmp_path)
    assert processor.read_count == 1

    lake = ParticleLakeProcessor(config, processor)
    jobs = lake.build_scan_jobs("sim")
    runner = HDF5ScanRunner(config, processor)
    runner.run("sim", [job.task for job in jobs], jobs[0].options)
    assert processor.read_count == 1  # nothing stale -> no re-read

    jobs_forced = lake.build_scan_jobs("sim", force=True)
    runner.run("sim", [job.task for job in jobs_forced], jobs_forced[0].options)
    assert processor.read_count == 2  # force -> full rebuild re-reads the file


def test_compute_ttot_dedup_exclusions_prefers_latest_mtime(tmp_path: Path) -> None:
    path_a = str(tmp_path / "a.h5part")
    path_b = str(tmp_path / "b.h5part")
    Path(path_a).write_text("a")
    Path(path_b).write_text("b")
    os.utime(path_a, (1000, 1000))
    os.utime(path_b, (2000, 2000))
    step_times = {path_a: [1.0, 2.0, 3.0], path_b: [3.0, 4.0]}

    excluded = compute_ttot_dedup_exclusions([path_a, path_b], step_times.get)

    # ttot=3.0 is shared; path_b has the later mtime, so path_a loses it.
    assert excluded == {path_a: {3.0}}


def test_compute_ttot_dedup_exclusions_no_overlap_means_no_exclusions(tmp_path: Path) -> None:
    path_a = str(tmp_path / "a.h5part")
    path_b = str(tmp_path / "b.h5part")
    Path(path_a).write_text("a")
    Path(path_b).write_text("b")
    step_times = {path_a: [1.0, 2.0], path_b: [3.0, 4.0]}

    excluded = compute_ttot_dedup_exclusions([path_a, path_b], step_times.get)

    assert excluded == {}


def test_dedup_drops_loser_files_duplicate_ttot_rows(tmp_path: Path) -> None:
    """Two files both claim ttot=5.5 (a restart boundary); the later-mtime file
    wins and the earlier file's rows for that ttot are dropped from its part,
    while its unique ttot=5.0 rows are kept."""
    config = make_config(tmp_path)
    config.lake_dir_of = {"sim": str(tmp_path / "lake" / "sim")}
    # sample_every_nb_time must be null (not the global default 1.0) or the
    # non-integer ttot=5.5 rows used below would be dropped by NB-time sampling
    # before dedup ever runs -- matches the real particle_lake config sample.
    config.particle_lake = {"enabled": True, "scan": {"sample_every_nb_time": None}}

    path_old = str(tmp_path / "snap.40_5.0.h5part")
    path_new = str(tmp_path / "snap.40_5.5.h5part")
    Path(path_old).write_text("old")
    Path(path_new).write_text("new")
    os.utime(path_old, (1000, 1000))
    os.utime(path_new, (2000, 2000))

    old_singles = pd.concat([_singles_df(5.0), _singles_df(5.5)], ignore_index=True)
    old_scalars = pd.concat([_scalars_df(5.0), _scalars_df(5.5)], ignore_index=True)
    new_singles = _singles_df(5.5).copy()
    new_singles["Name"] = pd.array([101, 102], dtype="int32")
    new_scalars = _scalars_df(5.5)

    empty = pd.DataFrame()
    tables = {
        path_old: {
            "scalars": old_scalars,
            "singles": old_singles,
            "binaries": empty,
            "mergers": empty,
        },
        path_new: {
            "scalars": new_scalars,
            "singles": new_singles,
            "binaries": empty,
            "mergers": empty,
        },
    }
    processor = FakeRawProcessor([path_old, path_new], tables)
    lake = ParticleLakeProcessor(config, processor)
    jobs = lake.build_scan_jobs("sim")
    runner = HDF5ScanRunner(config, processor)
    runner.run("sim", [job.task for job in jobs], jobs[0].options)

    data_dir = analysis_cache_dir(config, "sim", SNAPSHOT_SINGLES_FEATURE) / "data"
    rows = pd.read_parquet(data_dir)

    assert len(rows) == 4
    assert set(rows.loc[rows["ttot"] == 5.0, "object_id"]) == {1, 2}
    assert set(rows.loc[rows["ttot"] == 5.5, "object_id"]) == {101, 102}
    # No duplicate (ttot, object_id) pairs anywhere in the unioned dataset.
    assert not rows.duplicated(subset=["ttot", "object_id"]).any()


def test_ttot_dedup_map_is_cached_and_reused_when_file_list_unchanged(tmp_path: Path) -> None:
    config, processor, _outputs, _hdf5_path = _run_lake_tasks(tmp_path)
    lake = ParticleLakeProcessor(config, processor)
    cache_path = lake._ttot_dedup_cache_path("sim")
    assert cache_path.exists()

    def _boom(hdf5_path):
        raise AssertionError("should not recompute dedup map for an unchanged file list")

    processor.read_step_times = _boom
    lake.build_scan_jobs("sim")  # must reuse the cached map, not call read_step_times
