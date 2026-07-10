"""Tests for the compact_object_history and snapshot_summary pilot scan tasks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dragon3_pipelines.analysis.cache_paths import (
    COMPACT_OBJECT_HISTORY_FEATURE,
    analysis_cache_dir,
)
from dragon3_pipelines.analysis.compact_object_history import (
    CompactObjectHistoryProcessor,
    CompactObjectHistoryTask,
)
from dragon3_pipelines.analysis.hdf5_scan import HDF5ScanOptions, HDF5ScanRunner, HDF5ScanSession
from dragon3_pipelines.analysis.snapshot_summary import (
    SnapshotSummaryProcessor,
    SnapshotSummaryTask,
)
from dragon3_pipelines.schemas import load_table_schema
from tests.test_hdf5_scan import FakeProcessor, make_config


def _fixture_tables(hdf5_path: str) -> dict[str, dict[str, pd.DataFrame]]:
    scalars = pd.DataFrame(
        {
            "TTOT": [5.0],
            "Time[Myr]": [50.0],
            "N_SINGLE": [3.0],
            "N_BINARY": [2.0],
            "RC": [0.5],
            "RBAR": [2.0],
            "RG(1)": [8000.0],
            "RG(2)": [10.0],
            "RG(3)": [-10.0],
            "VG(1)": [200.0],
            "VG(2)": [1.0],
            "VG(3)": [-1.0],
        }
    ).set_index("TTOT", drop=False)

    singles = pd.DataFrame(
        {
            "TTOT": [5.0, 5.0, 5.0],
            "Time[Myr]": [50.0, 50.0, 50.0],
            "Name": [1, 2, 3],
            "KW": [1, 14, 10],
            "M": [1.0, 20.0, 0.6],
            "X [pc]": [1.0, 2.0, 3.0],
            "Y [pc]": [0.0, 0.0, 0.0],
            "Z [pc]": [0.0, 0.0, 0.0],
            "V1": [1.0, 2.0, 0.0],
            "V2": [0.0, 0.0, 1.0],
            "V3": [0.0, 0.0, 0.0],
            "Distance_to_cluster_center[pc]": [1.0, 2.0, 3.0],
        }
    )

    binaries = pd.DataFrame(
        {
            "TTOT": [5.0, 5.0],
            "Time[Myr]": [50.0, 50.0],
            "Bin Name1": [10, 20],
            "Bin Name2": [11, 21],
            "Bin cm Name": [100, 200],
            "Bin KW1": [13, 14],
            "Bin KW2": [1, 13],
            "Bin M1*": [1.4, 15.0],
            "Bin M2*": [1.0, 1.3],
            "Bin cm X [pc]": [5.0, 7.0],
            "Bin cm Y [pc]": [0.0, 1.0],
            "Bin cm Z [pc]": [0.0, 2.0],
            "Bin cm V1": [0.0, 3.0],
            "Bin cm V2": [0.0, 4.0],
            "Bin cm V3": [0.0, 0.0],
            "Distance_to_cluster_center[pc]": [5.0, 7.5],
            "Bin A[au]": [10.0, 3.0],
            "Bin ECC": [0.1, 0.2],
        }
    )

    return {hdf5_path: {"scalars": scalars, "singles": singles, "binaries": binaries}}


def _run_pilot_tasks(tmp_path: Path):
    config = make_config(tmp_path)
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    Path(hdf5_path).write_text("fake")
    processor = FakeProcessor([hdf5_path], _fixture_tables(hdf5_path))

    coh_task = CompactObjectHistoryTask(config, "sim")
    summary_task = SnapshotSummaryTask(config, "sim")
    runner = HDF5ScanRunner(config, processor)
    outputs = runner.run("sim", [coh_task, summary_task], HDF5ScanOptions(wait_age_hour=0))
    return config, processor, outputs


def test_compact_object_history_selects_only_compact_objects(tmp_path: Path) -> None:
    config, processor, _outputs = _run_pilot_tasks(tmp_path)

    data_dir = analysis_cache_dir(config, "sim", COMPACT_OBJECT_HISTORY_FEATURE) / "data"
    rows = pd.read_parquet(data_dir)
    load_table_schema("compact_object_history").validate_dataframe(
        rows[list(load_table_schema("compact_object_history").column_names())]
    )

    assert processor.read_count == 1
    assert len(rows) == 5
    assert set(rows["object_id"]) == {2, 3, 10, 20, 21}
    assert set(rows["object_type"]) == {"BH", "WD", "NS"}

    single_bh = rows.loc[rows["object_id"] == 2].iloc[0]
    assert bool(single_bh["is_in_binary"]) is False
    assert pd.isna(single_bh["companion_id"])
    assert single_bh["mass_msun"] == 20.0
    assert single_bh["x_pc"] == 2.0

    binary1_member = rows.loc[rows["object_id"] == 10].iloc[0]
    assert binary1_member["is_in_binary"] == True  # noqa: E712
    assert binary1_member["companion_id"] == 11
    assert binary1_member["companion_kw"] == 1
    assert binary1_member["companion_mass_msun"] == 1.0
    assert binary1_member["binary_cm_id"] == 100
    assert binary1_member["semi_major_axis_au"] == 10.0
    assert binary1_member["eccentricity"] == pytest.approx(0.1)
    assert binary1_member["x_pc"] == 5.0

    binary2_bh = rows.loc[rows["object_id"] == 20].iloc[0]
    binary2_ns = rows.loc[rows["object_id"] == 21].iloc[0]
    assert binary2_bh["companion_id"] == 21
    assert binary2_ns["companion_id"] == 20
    assert binary2_bh["companion_mass_msun"] == 1.3
    assert binary2_ns["companion_mass_msun"] == 15.0
    # Both members of the same binary share the binary's center-of-mass position.
    assert binary2_bh["x_pc"] == binary2_ns["x_pc"] == 7.0
    assert binary2_bh["y_pc"] == binary2_ns["y_pc"] == 1.0
    assert binary2_bh["z_pc"] == binary2_ns["z_pc"] == 2.0
    assert binary2_bh["binary_cm_id"] == binary2_ns["binary_cm_id"] == 200


def test_snapshot_summary_counts_mass_and_half_mass_radius(tmp_path: Path) -> None:
    _config, _processor, outputs = _run_pilot_tasks(tmp_path)

    summary = outputs["snapshot_summary"]
    load_table_schema("snapshot_summary").validate_dataframe(summary)
    assert len(summary) == 1
    row = summary.iloc[0]

    assert row["ttot"] == 5.0
    assert row["time_myr"] == 50.0
    assert row["n_singles"] == 3
    assert row["n_binaries"] == 2
    assert row["n_bh"] == 2  # single BH + binary2 member1
    assert row["n_ns"] == 2  # binary1 member1 + binary2 member2
    assert row["n_wd"] == 1
    assert row["total_mass_msun"] == pytest.approx(1.0 + 20.0 + 0.6 + 1.4 + 1.0 + 15.0 + 1.3)
    assert row["core_radius_pc"] == pytest.approx(0.5 * 2.0)
    # Half-mass radius: cumulative-mass median crossing on (mass, r_pc) sorted by r_pc;
    # hand-computed in the test docstring-equivalent comment above the fixture.
    assert row["half_mass_radius_pc"] == pytest.approx(2.0)
    assert row["rg_x_pc"] == 8000.0
    assert row["rg_y_pc"] == 10.0
    assert row["rg_z_pc"] == -10.0
    assert row["vg_x_kmps"] == 200.0
    assert row["vg_y_kmps"] == 1.0
    assert row["vg_z_kmps"] == -1.0


def test_pilot_tasks_share_one_read_per_file_via_session(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    Path(hdf5_path).write_text("fake")
    processor = FakeProcessor([hdf5_path], _fixture_tables(hdf5_path))

    session = HDF5ScanSession(config, processor)
    session.add_job(CompactObjectHistoryProcessor(config, processor).build_scan_job("sim"))
    session.add_job(SnapshotSummaryProcessor(config, processor).build_scan_job("sim"))
    session.run()

    assert processor.read_count == 1


def test_compact_object_history_empty_result_when_no_compact_objects(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    hdf5_path = str(tmp_path / "snap.40_1.0.h5part")
    Path(hdf5_path).write_text("fake")
    tables = _fixture_tables(hdf5_path)
    tables[hdf5_path]["singles"] = tables[hdf5_path]["singles"].iloc[0:0]
    tables[hdf5_path]["binaries"] = tables[hdf5_path]["binaries"].iloc[0:0]
    processor = FakeProcessor([hdf5_path], tables)

    task = CompactObjectHistoryTask(config, "sim")
    runner = HDF5ScanRunner(config, processor)
    runner.run("sim", [task], HDF5ScanOptions(wait_age_hour=0))

    data_dir = analysis_cache_dir(config, "sim", COMPACT_OBJECT_HISTORY_FEATURE) / "data"
    rows = pd.read_parquet(data_dir)
    assert rows.empty
