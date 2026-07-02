"""Tests for shared HDF5 scan data-reduction helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from dragon3_pipelines.analysis.b_type_binary import BTypeBinaryExtractor
from dragon3_pipelines.analysis.binary_stellar_type import BinaryStellarTypeExtractor
from dragon3_pipelines.analysis.compact_binary_counter import CompactBinaryCounter
from dragon3_pipelines.analysis.intermediate_mass_black_hole import (
    IntermediateMassBlackHoleAnalyzer,
)
from dragon3_pipelines.analysis import hdf5_scan
from dragon3_pipelines.analysis.primordial_binary import PrimordialBinaryIdentifier
from dragon3_pipelines.analysis.hdf5_scan import (
    HDF5ScanJob,
    HDF5ScanOptions,
    HDF5ScanRunner,
    HDF5ScanSession,
    ScanBackedAnalysisBase,
    hdf5_scan_options_from_config,
)
from dragon3_pipelines.io import HDF5FileProcessor


def make_config(tmp_path: Path) -> Mock:
    config = Mock()
    config.pathof = {"sim": str(tmp_path)}
    config.analysis_cache_dir_of = {"sim": str(tmp_path / "cache" / "sim")}
    config.particle_df_cache_dir_of = {"sim": str(tmp_path / "cache" / "sim" / "particle_df")}
    config.input_file_path_of = {"sim": str(tmp_path / "input.inp")}
    config.processes_count = 2
    config.tasks_per_child = 3
    config.hdf5 = {
        "file_selection": {
            "wait_age_hour": 0,
            "sample_every_nb_time": 1.0,
            "exclude_bad_dirname": True,
        },
        "table_cache": {"use_hdf5_cache": True},
        "scan": {"parallel": False, "incremental_from_cache_tail": True},
    }
    config.kw_to_stellar_type = {1: "MS", 13: "NS", 14: "BH"}
    config.stellar_type_to_kw = {"MS": 1, "NS": 13, "BH": 14}
    config.compact_object_KW = np.array([10, 11, 12, 13, 14])
    config.binary_stellar_type_extraction = {
        "cache_filename_template": "binaries_with_{target}_until_{last_ttot:.6f}.feather",
    }
    return config


def test_binary_stellar_type_resolves_type_and_kw(tmp_path: Path) -> None:
    extractor = BinaryStellarTypeExtractor(make_config(tmp_path))

    assert extractor.resolve_target(stellar_type="bh") == (14, "BH")
    assert extractor.resolve_target(stellar_type="MS") == (1, "MS")
    assert extractor.resolve_target(kw=14) == (14, "BH")
    assert extractor.resolve_target(kw="13") == (13, "NS")

    with pytest.raises(ValueError, match="exactly one"):
        extractor.resolve_target()
    with pytest.raises(ValueError, match="exactly one"):
        extractor.resolve_target(stellar_type="BH", kw=14)
    with pytest.raises(ValueError, match="Unknown stellar_type"):
        extractor.resolve_target(stellar_type="NOPE")
    with pytest.raises(ValueError, match="Unknown KW"):
        extractor.resolve_target(kw=99)


def test_analysis_cache_helper_accepts_legacy_particle_cache_mock(tmp_path: Path) -> None:
    config = Mock()
    config.particle_df_cache_dir_of = {"sim": str(tmp_path / "legacy_cache")}
    config.kw_to_stellar_type = {14: "BH"}
    config.stellar_type_to_kw = {"BH": 14}
    config.binary_stellar_type_extraction = {}

    task = BinaryStellarTypeExtractor(config)
    assert task.load_binaries_with_stellar_type("sim", stellar_type="BH", update=False).empty
    assert (tmp_path / "legacy_cache" / "binary_stellar_type").exists() is False


class FakeProcessor:
    def __init__(self, hdf5_paths: list[str], tables_by_path: dict[str, dict[str, pd.DataFrame]]):
        self.hdf5_paths = hdf5_paths
        self.tables_by_path = tables_by_path
        self.read_count = 0

    def get_all_hdf5_paths(self, *args, **kwargs):
        return self.hdf5_paths

    def read_tables(self, hdf5_path, simu_name, tables, columns_by_table=None, use_cache=True):
        self.read_count += 1
        return {table: self.tables_by_path[hdf5_path][table] for table in tables}


class FakeProgress:
    instances: list["FakeProgress"] = []

    def __init__(self):
        self.tasks: list[dict] = []
        self.advance_calls: list[int] = []
        FakeProgress.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def add_task(self, description, total):
        self.tasks.append({"description": description, "total": total})
        return len(self.tasks) - 1

    def advance(self, task_id):
        self.advance_calls.append(task_id)


def test_binary_extractor_returns_full_matching_binary_rows_and_writes_meta(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    tables = {
        str(hdf5_path): {
            "scalars": pd.DataFrame({"TTOT": [1.0, 2.0]}).set_index("TTOT", drop=False),
            "binaries": pd.DataFrame(
                {
                    "Bin KW1": [14, 1, 13],
                    "Bin KW2": [1, 14, 1],
                    "TTOT": [1.0, 1.0, 2.0],
                    "Time[Myr]": [10.0, 10.0, 20.0],
                    "Bin Name1": [101, 102, 103],
                    "Bin Name2": [201, 202, 203],
                    "extra_processed_column": ["keep-a", "keep-b", "keep-c"],
                }
            ),
        }
    }
    extractor = BinaryStellarTypeExtractor(config)
    fake_processor = FakeProcessor([str(hdf5_path)], tables)
    extractor.hdf5_file_processor = fake_processor

    result = extractor.load_binaries_with_stellar_type("sim", stellar_type="bh")

    assert len(result) == 2
    assert set(result["extra_processed_column"]) == {"keep-a", "keep-b"}
    assert list(result["TTOT"]) == [1.0, 1.0]
    cache_files = list((tmp_path / "cache" / "sim" / "binary_stellar_type").glob("*.feather"))
    assert cache_files[0].name == "binaries_with_14_BH_until_2.000000.feather"
    meta = json.loads(cache_files[0].with_name(cache_files[0].stem + ".meta.json").read_text())
    assert meta["target_kw"] == 14
    assert meta["target_stellar_type"] == "BH"
    assert meta["processed_files"][str(hdf5_path)]["ttot"] == [1.0, 2.0]

    result_again = extractor.load_binaries_with_stellar_type("sim", stellar_type="BH")
    assert fake_processor.read_count == 1
    pd.testing.assert_frame_equal(result, result_again)


def test_binary_extractor_writes_metadata_for_empty_match(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    tables = {
        str(hdf5_path): {
            "scalars": pd.DataFrame({"TTOT": [1.0]}).set_index("TTOT", drop=False),
            "binaries": pd.DataFrame({"Bin KW1": [1], "Bin KW2": [1], "TTOT": [1.0]}),
        }
    }
    extractor = BinaryStellarTypeExtractor(config)
    extractor.hdf5_file_processor = FakeProcessor([str(hdf5_path)], tables)

    result = extractor.load_binaries_with_stellar_type("sim", kw=14)

    assert result.empty
    cache_files = list((tmp_path / "cache" / "sim" / "binary_stellar_type").glob("*.feather"))
    assert cache_files
    meta = json.loads(cache_files[0].with_name(cache_files[0].stem + ".meta.json").read_text())
    assert meta["processed_files"][str(hdf5_path)]["ttot"] == [1.0]


def test_hdf5_file_processor_reads_selected_feather_tables(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    processor = HDF5FileProcessor(config)
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    pd.DataFrame({"TTOT": [1.0], "Time[Myr]": [10.0], "unused": [0]}).to_feather(
        hdf5_path + ".scalars.df.feather"
    )
    pd.DataFrame({"TTOT": [1.0], "Bin KW1": [14], "unused": [0]}).to_feather(
        hdf5_path + ".binaries.df.feather"
    )
    processor.read_file = Mock(side_effect=AssertionError("should not parse full HDF5"))

    result = processor.read_tables(
        hdf5_path,
        "sim",
        tables=["scalars", "binaries"],
        columns_by_table={"scalars": ["TTOT"], "binaries": ["TTOT", "Bin KW1"]},
        use_cache=True,
    )

    assert list(result["scalars"].columns) == ["TTOT"]
    assert list(result["binaries"].columns) == ["TTOT", "Bin KW1"]
    processor.read_file.assert_not_called()


def test_hdf5_file_processor_falls_back_when_feather_columns_missing(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    processor = HDF5FileProcessor(config)
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    pd.DataFrame({"TTOT": [1.0]}).to_feather(hdf5_path + ".scalars.df.feather")
    pd.DataFrame({"TTOT": [1.0]}).to_feather(hdf5_path + ".binaries.df.feather")
    fallback = {
        "scalars": pd.DataFrame({"TTOT": [1.0], "Time[Myr]": [10.0]}).set_index("TTOT", drop=False),
        "binaries": pd.DataFrame({"TTOT": [1.0], "Bin KW1": [14]}),
    }
    processor.read_file = Mock(return_value=fallback)

    result = processor.read_tables(
        hdf5_path,
        "sim",
        tables=["scalars", "binaries"],
        columns_by_table={"scalars": ["TTOT", "Time[Myr]"], "binaries": ["Bin KW1"]},
        use_cache=True,
    )

    assert result["binaries"]["Bin KW1"].iloc[0] == 14
    processor.read_file.assert_called_once()


class FakeTask:
    required_tables = ("scalars", "binaries")
    columns_by_table = {"scalars": ["TTOT"], "binaries": ["TTOT"]}

    def __init__(self, name: str):
        self.name = name
        self.writes = 0

    def read_cache(self):
        return pd.DataFrame()

    def read_meta(self):
        return {}

    def is_file_fresh(self, hdf5_path, meta, cache_df):
        return False

    def process_file(self, hdf5_path, df_dict, meta, cache_df):
        return {
            "rows": pd.DataFrame({"task": [self.name], "path": [hdf5_path]}),
            "file_meta": {"mtime": 1.0, "ttot": [1.0]},
        }

    def merge_file_result(self, cache_df, hdf5_path, result):
        return pd.concat([cache_df, result["rows"]], ignore_index=True)

    def write_cache_and_meta(self, cache_df, processed_files, options):
        self.writes += 1

    def finalize_cache(self, cache_df):
        return cache_df


class MetaTask(FakeTask):
    def __init__(self, name: str, meta: dict, cache_df: pd.DataFrame | None = None):
        super().__init__(name)
        self.meta = meta
        self.cache_df = cache_df if cache_df is not None else pd.DataFrame()
        self.fresh_checks: list[str] = []
        self.fresh_paths: set[str] = set()
        self.processed_paths: list[str] = []

    def read_cache(self):
        return self.cache_df

    def read_meta(self):
        return self.meta

    def is_file_fresh(self, hdf5_path, meta, cache_df):
        self.fresh_checks.append(hdf5_path)
        return hdf5_path in self.fresh_paths

    def process_file(self, hdf5_path, df_dict, meta, cache_df):
        self.processed_paths.append(hdf5_path)
        return super().process_file(hdf5_path, df_dict, meta, cache_df)


class ScanBackedAnalysisForTest(ScanBackedAnalysisBase):
    pass


def test_scan_backed_analysis_scan_options_merge_and_validate(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.hdf5["file_selection"]["sample_every_nb_time"] = 2.0
    config.hdf5["scan"]["parallel"] = True
    config.hdf5["table_cache"]["use_hdf5_cache"] = False
    analysis = ScanBackedAnalysisForTest(config, FakeProcessor([], {}))

    options = analysis._scan_options(force=True)
    assert options.sample_every_nb_time == 2.0
    assert options.parallel is True
    assert options.use_hdf5_cache is False
    assert not hasattr(options, "processes")
    assert options.force is True
    assert hdf5_scan_options_from_config(config).wait_age_hour == 0


def test_scan_backed_analysis_cache_only_path_does_not_touch_hdf5(tmp_path: Path) -> None:
    analysis = ScanBackedAnalysisForTest(make_config(tmp_path), FakeProcessor([], {}))
    task = MetaTask("cached", {}, pd.DataFrame({"value": [2, 1]}))

    def finalize_cache(cache_df):
        return cache_df.sort_values("value").reset_index(drop=True)

    task.finalize_cache = finalize_cache
    job = HDF5ScanJob("sim", task, HDF5ScanOptions())

    result = analysis._load_or_update_scan_job(job, update=False)

    assert result["value"].tolist() == [1, 2]
    assert analysis.hdf5_file_processor.read_count == 0
    assert task.writes == 0


def test_scan_backed_analysis_run_scan_job_matches_runner_result(tmp_path: Path) -> None:
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
    analysis = ScanBackedAnalysisForTest(config, processor)
    task = FakeTask("single")
    job = HDF5ScanJob("sim", task, HDF5ScanOptions(wait_age_hour=0))

    result = analysis._run_scan_job(job)

    expected = pd.DataFrame({"task": ["single"], "path": [hdf5_path]})
    pd.testing.assert_frame_equal(result, expected)
    assert processor.read_count == 1


def test_scan_runner_reads_each_hdf5_file_once_for_multiple_tasks(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    tables = {
        str(hdf5_path): {
            "scalars": pd.DataFrame({"TTOT": [1.0]}),
            "binaries": pd.DataFrame({"TTOT": [1.0]}),
        }
    }
    processor = FakeProcessor([str(hdf5_path)], tables)
    runner = HDF5ScanRunner(config, processor)
    task_a = FakeTask("a")
    task_b = FakeTask("b")

    result = runner.run("sim", [task_a, task_b], HDF5ScanOptions(wait_age_hour=0))

    assert processor.read_count == 1
    assert set(result) == {"a", "b"}
    assert result["a"]["task"].tolist() == ["a"]
    assert result["b"]["task"].tolist() == ["b"]
    assert task_a.writes == 1
    assert task_b.writes == 1


def test_scan_runner_progress_advances_once_per_stale_hdf5_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeProgress.instances.clear()
    monkeypatch.setattr(hdf5_scan, "Progress", FakeProgress)
    config = make_config(tmp_path)
    paths = [str(tmp_path / f"snap.40_{idx}.h5part") for idx in range(2)]
    for path in paths:
        Path(path).write_text("fake")
    tables = {
        path: {
            "scalars": pd.DataFrame({"TTOT": [float(idx)]}),
            "binaries": pd.DataFrame({"TTOT": [float(idx)]}),
        }
        for idx, path in enumerate(paths)
    }
    processor = FakeProcessor(paths, tables)
    runner = HDF5ScanRunner(config, processor)

    runner.run("sim", [FakeTask("progress")], HDF5ScanOptions(wait_age_hour=0))

    assert len(FakeProgress.instances) == 1
    progress = FakeProgress.instances[0]
    assert progress.tasks == [{"description": "sim HDF5 scan", "total": 2}]
    assert progress.advance_calls == [0, 0]


def test_scan_runner_skips_progress_when_no_stale_hdf5_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeProgress.instances.clear()
    monkeypatch.setattr(hdf5_scan, "Progress", FakeProgress)
    config = make_config(tmp_path)
    paths = [str(tmp_path / f"snap.40_{idx}.h5part") for idx in range(2)]
    for path in paths:
        Path(path).write_text("fake")
    tables = {
        path: {
            "scalars": pd.DataFrame({"TTOT": [float(idx)]}),
            "binaries": pd.DataFrame({"TTOT": [float(idx)]}),
        }
        for idx, path in enumerate(paths)
    }
    task = MetaTask("fresh", {})
    task.fresh_paths = set(paths)
    runner = HDF5ScanRunner(config, FakeProcessor(paths, tables))

    runner.run("sim", [task], HDF5ScanOptions(wait_age_hour=0))

    assert FakeProgress.instances == []


def test_scan_runner_rejects_duplicate_task_names(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    runner = HDF5ScanRunner(config, FakeProcessor([], {}))

    with pytest.raises(ValueError, match="unique names"):
        runner.run("sim", [FakeTask("same"), FakeTask("same")], HDF5ScanOptions())


def test_scan_session_batches_compatible_jobs_and_clears_queue(tmp_path: Path) -> None:
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
    session = HDF5ScanSession(config, processor)

    session.add_task("sim", FakeTask("a"), HDF5ScanOptions(wait_age_hour=0))
    session.add_task("sim", FakeTask("b"), HDF5ScanOptions(wait_age_hour=0))

    result = session.run()

    assert processor.read_count == 1
    assert set(result["sim"]) == {"a", "b"}
    assert session.jobs == []


def test_scan_session_keeps_different_options_in_separate_groups(tmp_path: Path) -> None:
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
    session = HDF5ScanSession(config, processor)

    session.add_task("sim", FakeTask("a"), HDF5ScanOptions(wait_age_hour=0))
    session.add_task("sim", FakeTask("b"), HDF5ScanOptions(wait_age_hour=1))

    session.run()

    assert processor.read_count == 2


def test_scan_runner_incremental_tail_checks_only_last_processed_and_later(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    paths = [str(tmp_path / f"snap.40_{idx}.h5part") for idx in range(4)]
    for path in paths:
        Path(path).write_text("fake")
    tables = {
        path: {
            "scalars": pd.DataFrame({"TTOT": [float(idx)]}),
            "binaries": pd.DataFrame({"TTOT": [float(idx)]}),
        }
        for idx, path in enumerate(paths)
    }
    processor = FakeProcessor(paths, tables)
    meta = {
        "scan_options": {
            "sample_every_nb_time": 1.0,
            "wait_age_hour": 0,
            "use_hdf5_cache": True,
            "parallel": False,
            "exclude_bad_dirname": True,
            "incremental_from_cache_tail": True,
        },
        "processed_files": {paths[0]: {"mtime": 1.0}, paths[1]: {"mtime": 1.0}},
    }
    task = MetaTask("tail", meta)
    runner = HDF5ScanRunner(config, processor)

    runner.run("sim", [task], HDF5ScanOptions(wait_age_hour=0))

    assert task.fresh_checks == paths[1:]
    assert processor.read_count == 3


def test_scan_runner_force_ignores_old_cache_and_meta(tmp_path: Path) -> None:
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
    task = MetaTask(
        "force",
        {"processed_files": {hdf5_path: {"mtime": 1.0}}},
        pd.DataFrame({"old": [1]}),
    )
    runner = HDF5ScanRunner(config, processor)

    result = runner.run("sim", [task], HDF5ScanOptions(wait_age_hour=0, force=True))

    assert processor.read_count == 1
    assert task.fresh_checks == []
    assert "old" not in result["force"].columns


def test_scan_runner_rebuilds_cache_when_old_meta_has_no_scan_options(tmp_path: Path) -> None:
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
    task = MetaTask(
        "legacy",
        {"processed_files": {hdf5_path: {"mtime": 1.0}}},
        pd.DataFrame({"old": [1]}),
    )
    runner = HDF5ScanRunner(config, processor)

    result = runner.run("sim", [task], HDF5ScanOptions(wait_age_hour=0))

    assert processor.read_count == 1
    assert task.fresh_checks == [hdf5_path]
    assert "old" not in result["legacy"].columns


def compact_tables(rows: list[tuple[int, int, int, int, float, float]]) -> dict[str, pd.DataFrame]:
    binaries = pd.DataFrame(
        rows,
        columns=["Bin Name1", "Bin Name2", "Bin KW1", "Bin KW2", "TTOT", "Time[Myr]"],
    )
    scalars = pd.DataFrame(
        {
            "TTOT": sorted(binaries["TTOT"].unique()),
            "Time[Myr]": [float(t) * 10.0 for t in sorted(binaries["TTOT"].unique())],
        }
    ).set_index("TTOT", drop=False)
    return {"scalars": scalars, "binaries": binaries}


def imbh_tables(
    *,
    singles: pd.DataFrame | None = None,
    binaries: pd.DataFrame | None = None,
    mergers: pd.DataFrame | None = None,
    times: list[float] | None = None,
) -> dict[str, pd.DataFrame]:
    if times is None:
        values = []
        for df in [singles, binaries, mergers]:
            if df is not None and "TTOT" in df.columns:
                values.extend(float(ttot) for ttot in df["TTOT"].dropna().unique())
        times = sorted(set(values)) or [1.0]
    scalars = pd.DataFrame(
        {"TTOT": times, "Time[Myr]": [float(ttot) * 10.0 for ttot in times]}
    ).set_index("TTOT", drop=False)
    if singles is None:
        singles = pd.DataFrame(columns=["Name", "KW", "M", "TTOT", "Time[Myr]"])
    if binaries is None:
        binaries = pd.DataFrame(
            columns=["Bin Name1", "Bin Name2", "Bin KW1", "Bin KW2", "Bin M1*", "Bin M2*", "TTOT"]
        )
    if mergers is None:
        mergers = pd.DataFrame(
            columns=[
                "Mer NAM1",
                "Mer NAM2",
                "Mer NAM3",
                "Mer NAMC",
                "Mer KW1",
                "Mer KW2",
                "Mer KW3",
                "Mer KWC",
                "Mer M1",
                "Mer M2",
                "Mer M3",
                "TTOT",
            ]
        )
    return {"scalars": scalars, "singles": singles, "binaries": binaries, "mergers": mergers}


def test_imbh_snapshot_extraction_handles_singles_binaries_and_boundaries(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    Path(hdf5_path).write_text("fake")
    singles = pd.DataFrame(
        {
            "Name": [10, 11, 12, 13],
            "KW": [14, 14, 14, 13],
            "M": [150.0, 100.0, 1.0e5, 200.0],
            "TTOT": [1.0, 1.0, 1.0, 1.0],
            "Time[Myr]": [10.0, 10.0, 10.0, 10.0],
            "Distance_to_cluster_center[pc]": [1.5, 2.0, 3.0, 4.0],
        }
    )
    binaries = pd.DataFrame(
        {
            "Bin Name1": [20, 22, 30],
            "Bin Name2": [21, 23, 31],
            "Bin KW1": [14, 1, 14],
            "Bin KW2": [1, 14, 14],
            "Bin M1*": [300.0, 5.0, 120.0],
            "Bin M2*": [5.0, 400.0, 130.0],
            "Bin A[au]": [1.0, 2.0, 3.0],
            "Bin ECC": [0.1, 0.2, 0.3],
            "TTOT": [1.0, 1.0, 1.0],
            "Time[Myr]": [10.0, 10.0, 10.0],
        }
    )
    processor = FakeProcessor(
        [hdf5_path], {hdf5_path: imbh_tables(singles=singles, binaries=binaries)}
    )
    analyzer = IntermediateMassBlackHoleAnalyzer(config, processor)

    result = analyzer.load_imbh_snapshots("sim")

    assert result["object_name"].tolist() == [10, 20, 23, 30, 31]
    assert set(result["state"]) == {"single", "binary"}
    assert result[result["object_name"] == 30]["component_index"].iloc[0] == 1
    assert result[result["object_name"] == 31]["component_index"].iloc[0] == 2
    assert 11 not in set(result["object_name"])
    assert 12 not in set(result["object_name"])
    assert 13 not in set(result["object_name"])


def test_imbh_binary_rows_win_over_single_rows_at_same_snapshot(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    Path(hdf5_path).write_text("fake")
    singles = pd.DataFrame({"Name": [50], "KW": [14], "M": [200.0], "TTOT": [1.0]})
    binaries = pd.DataFrame(
        {
            "Bin Name1": [50],
            "Bin Name2": [51],
            "Bin KW1": [14],
            "Bin KW2": [1],
            "Bin M1*": [210.0],
            "Bin M2*": [3.0],
            "TTOT": [1.0],
        }
    )
    analyzer = IntermediateMassBlackHoleAnalyzer(
        config,
        FakeProcessor([hdf5_path], {hdf5_path: imbh_tables(singles=singles, binaries=binaries)}),
    )

    result = analyzer.load_imbh_snapshots("sim")

    assert len(result) == 1
    assert result["state"].iloc[0] == "binary"
    assert result["companion_name"].iloc[0] == 51


def test_imbh_merger_lineage_and_fate_summary(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    path0 = str(tmp_path / "snap.40_0.0.h5part")
    path1 = str(tmp_path / "snap.40_1.0.h5part")
    path2 = str(tmp_path / "snap.40_2.0.h5part")
    for path in [path0, path1, path2]:
        Path(path).write_text("fake")
    tables = {
        path0: imbh_tables(
            singles=pd.DataFrame({"Name": [1], "KW": [14], "M": [150.0], "TTOT": [0.0]}),
            times=[0.0],
        ),
        path1: imbh_tables(
            singles=pd.DataFrame({"Name": [10], "KW": [14], "M": [180.0], "TTOT": [1.0]}),
            mergers=pd.DataFrame(
                {
                    "Mer NAM1": [2],
                    "Mer NAM2": [3],
                    "Mer NAM3": [np.nan],
                    "Mer NAMC": [10],
                    "Mer KW1": [14],
                    "Mer KW2": [14],
                    "Mer KW3": [np.nan],
                    "Mer KWC": [14],
                    "Mer M1": [90.0],
                    "Mer M2": [90.0],
                    "Mer M3": [np.nan],
                    "TTOT": [1.0],
                    "Time[Myr]": [10.0],
                }
            ),
            times=[1.0],
        ),
        path2: imbh_tables(
            singles=pd.DataFrame({"Name": [20], "KW": [14], "M": [350.0], "TTOT": [2.0]}),
            mergers=pd.DataFrame(
                {
                    "Mer NAM1": [10],
                    "Mer NAM2": [11],
                    "Mer NAM3": [np.nan],
                    "Mer NAMC": [20],
                    "Mer KW1": [14],
                    "Mer KW2": [14],
                    "Mer KW3": [np.nan],
                    "Mer KWC": [14],
                    "Mer M1": [180.0],
                    "Mer M2": [170.0],
                    "Mer M3": [np.nan],
                    "TTOT": [2.0],
                    "Time[Myr]": [20.0],
                }
            ),
            times=[2.0],
        ),
    }
    analyzer = IntermediateMassBlackHoleAnalyzer(
        config, FakeProcessor([path0, path1, path2], tables)
    )

    result = analyzer.summarize_simulation("sim")
    objects = result["objects"].set_index("object_name")

    assert result["summary"]["n_objects"] == 3
    assert result["summary"]["scanned_files"] == 3
    assert objects.loc[1, "formation_channel"] == "already_imbh_at_scan_start"
    assert objects.loc[1, "fate_label"] == "not_seen_at_scan_end"
    assert objects.loc[10, "formation_channel"] == "first_generation_merger"
    assert objects.loc[10, "fate_label"] == "merged_into_other"
    assert objects.loc[10, "fate_product_name"] == 20
    assert objects.loc[20, "formation_channel"] == "hierarchical_merger"
    assert objects.loc[20, "hierarchical_merger_generation"] == 2
    assert objects.loc[20, "fate_label"] == "retained_candidate"
    assert len(result["merger_events"]) == 2


def test_imbh_scan_cache_reuse_tail_append_and_stale_replacement(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    path1 = str(tmp_path / "snap.40_1.0.h5part")
    path2 = str(tmp_path / "snap.40_2.0.h5part")
    for path in [path1, path2]:
        Path(path).write_text("fake")
    tables = {
        path1: imbh_tables(
            singles=pd.DataFrame({"Name": [1], "KW": [14], "M": [150.0], "TTOT": [1.0]}),
            times=[1.0],
        ),
        path2: imbh_tables(
            singles=pd.DataFrame({"Name": [2], "KW": [14], "M": [250.0], "TTOT": [2.0]}),
            times=[2.0],
        ),
    }
    processor = FakeProcessor([path1], tables)
    analyzer = IntermediateMassBlackHoleAnalyzer(config, processor)

    first = analyzer.load_imbh_snapshots("sim")
    second = analyzer.load_imbh_snapshots("sim")
    assert processor.read_count == 1
    assert first[["object_name", "TTOT", "state", "mass_msun"]].to_dict("records") == second[
        ["object_name", "TTOT", "state", "mass_msun"]
    ].to_dict("records")

    processor.hdf5_paths = [path1, path2]
    appended = analyzer.load_imbh_snapshots("sim")
    assert processor.read_count == 2
    assert appended["object_name"].tolist() == [1, 2]

    tables[path2] = imbh_tables(
        singles=pd.DataFrame({"Name": [3], "KW": [14], "M": [300.0], "TTOT": [2.0]}),
        times=[2.0],
    )
    os.utime(path2, None)
    replaced = analyzer.load_imbh_snapshots("sim", force=True)
    assert replaced["object_name"].tolist() == [1, 3]


def test_compact_counter_reuses_fresh_scan_cache(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    Path(hdf5_path).write_text("fake")
    processor = FakeProcessor(
        [hdf5_path],
        {hdf5_path: compact_tables([(1, 2, 14, 14, 1.0, 10.0)])},
    )
    counter = CompactBinaryCounter(config)
    counter.hdf5_file_processor = processor

    first = counter.summarize_simulation("sim")
    second = counter.summarize_simulation("sim")

    assert processor.read_count == 1
    assert first["summary"] == second["summary"]
    assert first["summary"]["gw_source"] == 1


def test_compact_counter_append_scans_only_tail_range(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    first_path = str(tmp_path / "snap.40_1.0.h5part")
    second_path = str(tmp_path / "snap.40_2.0.h5part")
    Path(first_path).write_text("fake")
    Path(second_path).write_text("fake")
    processor = FakeProcessor(
        [first_path],
        {
            first_path: compact_tables([(1, 2, 14, 14, 1.0, 10.0)]),
            second_path: compact_tables([(3, 4, 13, 1, 2.0, 20.0)]),
        },
    )
    counter = CompactBinaryCounter(config)
    counter.hdf5_file_processor = processor

    counter.summarize_simulation("sim")
    processor.hdf5_paths = [first_path, second_path]
    result = counter.summarize_simulation("sim")

    assert processor.read_count == 2
    assert result["summary"]["scanned_files"] == 2
    assert result["summary"]["scanned_snapshots"] == 2
    assert result["summary"]["gw_source"] == 1
    assert result["summary"]["pulsar"] == 1


def test_compact_counter_replaces_stale_last_file_ttot_rows(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    processor = FakeProcessor(
        [str(hdf5_path)],
        {str(hdf5_path): compact_tables([(1, 2, 14, 14, 1.0, 10.0)])},
    )
    counter = CompactBinaryCounter(config)
    counter.hdf5_file_processor = processor

    first = counter.summarize_simulation("sim")
    processor.tables_by_path[str(hdf5_path)] = compact_tables([(5, 6, 13, 1, 1.0, 10.0)])
    hdf5_path.write_text("changed")
    os.utime(hdf5_path, (hdf5_path.stat().st_atime + 5, hdf5_path.stat().st_mtime + 5))
    second = counter.summarize_simulation("sim")

    assert processor.read_count == 2
    assert first["summary"]["gw_source"] == 1
    assert second["summary"]["gw_source"] == 0
    assert second["summary"]["pulsar"] == 1


def make_primordial_tables(
    hdf5_path: Path, binaries: pd.DataFrame
) -> dict[str, dict[str, pd.DataFrame]]:
    return {
        str(hdf5_path): {
            "scalars": pd.DataFrame({"TTOT": sorted(binaries["TTOT"].unique())}).set_index(
                "TTOT", drop=False
            ),
            "binaries": binaries,
        }
    }


def test_primordial_identifier_filters_adjacent_integer_name_pairs(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame(
        {
            "Bin Name1": [1, 10, 4],
            "Bin Name2": [2, 9, 6],
            "TTOT": [0.0, 0.0, 0.0],
            "extra_processed_column": ["keep-a", "keep-b", "drop-c"],
        }
    )
    identifier = PrimordialBinaryIdentifier(config)
    fake_processor = FakeProcessor([str(hdf5_path)], make_primordial_tables(hdf5_path, binaries))
    identifier.hdf5_file_processor = fake_processor

    result = identifier.load_primordial_binaries("sim", wait_age_hour=0)

    assert result["extra_processed_column"].tolist() == ["keep-a", "keep-b"]
    assert result["primordial_name_min"].tolist() == [1, 9]
    assert result["primordial_name_max"].tolist() == [2, 10]
    assert result["primordial_pair_key"].tolist() == ["1-2", "9-10"]
    assert result["is_primordial_binary"].tolist() == [True, True]
    assert fake_processor.read_count == 1

    cache_path = tmp_path / "cache" / "sim" / "primordial_binary" / "primordial_binaries.feather"
    meta_path = tmp_path / "cache" / "sim" / "primordial_binary" / "primordial_binaries.meta.json"
    assert cache_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["schema_version"] == 1
    assert meta["source_hdf5_path"] == str(hdf5_path)
    assert meta["source_mtime"] == hdf5_path.stat().st_mtime
    assert meta["discovered_ttot_values"] == [0.0]
    assert meta["row_count"] == 2
    assert meta["ttot_rule"] == "binaries['TTOT'].astype(float) == 0.0"


def test_primordial_identifier_uses_strict_zero_ttot_snapshot(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame(
        {
            "Bin Name1": [1, 20, 30],
            "Bin Name2": [2, 21, 31],
            "TTOT": [0.0, 0.1, 1.0],
            "extra_processed_column": ["keep-zero", "drop-nonzero", "drop-one"],
        }
    )
    identifier = PrimordialBinaryIdentifier(config)
    identifier.hdf5_file_processor = FakeProcessor(
        [str(hdf5_path)], make_primordial_tables(hdf5_path, binaries)
    )

    result = identifier.load_primordial_binaries("sim", wait_age_hour=0)

    assert result["extra_processed_column"].tolist() == ["keep-zero"]
    assert result["TTOT"].tolist() == [0.0]


def test_primordial_identifier_reuses_fresh_cache_without_rereading_hdf5(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame({"Bin Name1": [1], "Bin Name2": [2], "TTOT": [0.0]})
    identifier = PrimordialBinaryIdentifier(config)
    fake_processor = FakeProcessor([str(hdf5_path)], make_primordial_tables(hdf5_path, binaries))
    identifier.hdf5_file_processor = fake_processor

    first = identifier.load_primordial_binaries("sim", wait_age_hour=0)
    second = identifier.load_primordial_binaries("sim", wait_age_hour=0)

    assert fake_processor.read_count == 1
    pd.testing.assert_frame_equal(first, second)


def test_primordial_identifier_update_false_reads_cache(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame({"Bin Name1": [1], "Bin Name2": [2], "TTOT": [0.0]})
    identifier = PrimordialBinaryIdentifier(config)
    fake_processor = FakeProcessor([str(hdf5_path)], make_primordial_tables(hdf5_path, binaries))
    identifier.hdf5_file_processor = fake_processor

    first = identifier.load_primordial_binaries("sim", wait_age_hour=0)
    fake_processor.hdf5_paths = []
    second = identifier.load_primordial_binaries("sim", update=False)

    assert fake_processor.read_count == 1
    pd.testing.assert_frame_equal(first, second)


def test_primordial_identifier_fails_when_no_hdf5_files(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    identifier = PrimordialBinaryIdentifier(config)
    identifier.hdf5_file_processor = FakeProcessor([], {})

    with pytest.raises(ValueError, match="No HDF5 files"):
        identifier.load_primordial_binaries("sim", wait_age_hour=0)


def test_primordial_identifier_fails_when_required_name_columns_missing(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame({"Bin Name1": [1], "TTOT": [0.0]})
    identifier = PrimordialBinaryIdentifier(config)
    identifier.hdf5_file_processor = FakeProcessor(
        [str(hdf5_path)], make_primordial_tables(hdf5_path, binaries)
    )

    with pytest.raises(ValueError, match="Bin Name2"):
        identifier.load_primordial_binaries("sim", wait_age_hour=0)


def test_primordial_identifier_fails_when_zero_ttot_snapshot_missing(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_0.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame({"Bin Name1": [1], "Bin Name2": [2], "TTOT": [0.1]})
    identifier = PrimordialBinaryIdentifier(config)
    identifier.hdf5_file_processor = FakeProcessor(
        [str(hdf5_path)], make_primordial_tables(hdf5_path, binaries)
    )

    with pytest.raises(ValueError, match="TTOT == 0.0"):
        identifier.load_primordial_binaries("sim", wait_age_hour=0)


def test_b_type_extractor_filters_members_marks_primordial_and_writes_meta(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame(
        {
            "Bin KW1": [2, 2, 1, 2, 1, 2, 1, 1, 2],
            "Bin KW2": [2, 2, 2, 1, 1, 1, 2, 2, 2],
            "Bin Teff1*": [9000, 9000, 10500, 9000, 20000, 20000, 10499, 20000, 20000],
            "Bin Teff2*": [9000, 9000, 9000, 31500, 20000, 20000, 20000, 20000, 20000],
            "Bin M1*": [1.0, 1.0, 2.75, 1.0, 10.0, 10.0, 10.0, 200.0, 10.0],
            "Bin M2*": [1.0, 1.0, 1.0, 17.7, 10.0, 2.74, 10.0, 10.0, 10.0],
            "TTOT": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Time[Myr]": [0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "Bin Name1": [1, 9, 2, 30, 10, 50, 60, 70, 80],
            "Bin Name2": [2, 10, 1, 31, 9, 51, 61, 71, 81],
            "extra_processed_column": [
                "primordial-a",
                "primordial-b",
                "member1-lower-boundary",
                "member2-upper-boundary",
                "both-members",
                "drop-mass-low",
                "drop-teff-low",
                "drop-mass-high",
                "drop-kw",
            ],
        }
    )
    extractor = BTypeBinaryExtractor(config)
    fake_processor = FakeProcessor([str(hdf5_path)], make_primordial_tables(hdf5_path, binaries))
    extractor.hdf5_file_processor = fake_processor

    result = extractor.load_b_type_binaries("sim")

    assert result["extra_processed_column"].tolist() == [
        "member1-lower-boundary",
        "both-members",
        "member2-upper-boundary",
    ]
    assert result["b_type_member1"].tolist() == [True, True, False]
    assert result["b_type_member2"].tolist() == [False, True, True]
    assert result["b_type_member_count"].tolist() == [1, 2, 1]
    assert result["b_type_pair_key"].tolist() == ["1-2", "9-10", "30-31"]
    assert result["is_primordial_binary"].tolist() == [True, True, False]
    assert "extra_processed_column" in result.columns

    cache_path = (
        tmp_path / "cache" / "sim" / "b_type_binary" / "b_type_binaries_until_1.000000.feather"
    )
    meta_path = cache_path.with_name(cache_path.stem + ".meta.json")
    assert cache_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["schema_version"] == 1
    assert meta["b_type_criteria"] == {
        "kw": 1,
        "teff_min": 10500,
        "teff_max": 31500,
        "mass_min": 2.75,
        "mass_max": 17.7,
    }
    assert meta["primordial_signature"]["meta"]["row_count"] == 2
    assert meta["processed_files"][str(hdf5_path)]["ttot"] == [0.0, 1.0]

    result_again = extractor.load_b_type_binaries("sim")
    assert fake_processor.read_count == 2
    pd.testing.assert_frame_equal(result, result_again)


def test_b_type_extractor_refreshes_when_primordial_cache_changes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame(
        {
            "Bin KW1": [2, 1],
            "Bin KW2": [2, 2],
            "Bin Teff1*": [9000, 20000],
            "Bin Teff2*": [9000, 9000],
            "Bin M1*": [1.0, 5.0],
            "Bin M2*": [1.0, 1.0],
            "TTOT": [0.0, 1.0],
            "Bin Name1": [1, 2],
            "Bin Name2": [2, 1],
        }
    )
    extractor = BTypeBinaryExtractor(config)
    fake_processor = FakeProcessor([str(hdf5_path)], make_primordial_tables(hdf5_path, binaries))
    extractor.hdf5_file_processor = fake_processor

    first = extractor.load_b_type_binaries("sim")
    assert first["is_primordial_binary"].tolist() == [True]
    assert fake_processor.read_count == 2

    primordial_meta = (
        tmp_path / "cache" / "sim" / "primordial_binary" / "primordial_binaries.meta.json"
    )
    meta = json.loads(primordial_meta.read_text())
    meta["row_count"] = 0
    primordial_meta.write_text(json.dumps(meta, indent=2, sort_keys=True))

    second = extractor.load_b_type_binaries("sim")

    assert fake_processor.read_count == 3
    assert second["is_primordial_binary"].tolist() == [True]


def test_b_type_extractor_handles_duplicate_binary_indices(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = tmp_path / "snap.40_1.0.h5part"
    hdf5_path.write_text("fake")
    binaries = pd.DataFrame(
        {
            "Bin KW1": [2, 1, 1],
            "Bin KW2": [2, 2, 1],
            "Bin Teff1*": [9000, 20000, 20000],
            "Bin Teff2*": [9000, 9000, 20000],
            "Bin M1*": [1.0, 5.0, 5.0],
            "Bin M2*": [1.0, 1.0, 5.0],
            "TTOT": [0.0, 1.0, 2.0],
            "Bin Name1": [1, 10, 20],
            "Bin Name2": [2, 11, 21],
        },
        index=[0, 0, 0],
    )
    extractor = BTypeBinaryExtractor(config)
    extractor.hdf5_file_processor = FakeProcessor(
        [str(hdf5_path)], make_primordial_tables(hdf5_path, binaries)
    )

    result = extractor.load_b_type_binaries("sim")

    assert result["TTOT"].tolist() == [1.0, 2.0]
    assert result["b_type_member1"].tolist() == [True, True]
    assert result["b_type_member2"].tolist() == [False, True]
