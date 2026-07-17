"""Tests for nbody_pipeline.analysis.bin_label_backfill."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nbody_pipeline.analysis.bin_label_backfill import (
    ALGORITHM_VERSION,
    backfill_parts,
    reconstruct_bin_label,
    resolve_nzero,
    validate_reconstruction,
)
from nbody_pipeline.analysis.hdf5_scan import HDF5ScanOptions, HDF5ScanRunner
from nbody_pipeline.analysis.particle_lake import _BIN_LABEL_UNKNOWN, SnapshotBinariesTask
from nbody_pipeline.io.text_parsers import SCALAR_KEYS
from tests.test_hdf5_scan import make_config

NZERO = 1000

RBAR = 2.0
TSCALE = 0.1
RDENS = (1.0, 0.5, -0.5)


def _int_df(cm_id, obj1, obj2, pert_gamma, cm_kw, bin_label) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cm_id": pd.array(cm_id, dtype="int64"),
            "object_id_1": pd.array(obj1, dtype="int64"),
            "object_id_2": pd.array(obj2, dtype="int64"),
            "pert_gamma": pd.array(pert_gamma, dtype="float32"),
            "cm_kw": pd.array(cm_kw, dtype="int32"),
            "bin_label": pd.array(bin_label, dtype="int32"),
        }
    )


# ---------------------------------------------------------------------------
# reconstruct_bin_label: pure-function unit tests
# ---------------------------------------------------------------------------


def test_reconstruct_merger_rule_negative_cm_id():
    df = _int_df([-5], [1], [2], [0.05], [0], [_BIN_LABEL_UNKNOWN])
    result = reconstruct_bin_label(df, NZERO)
    assert result.tolist() == [-1]


def test_reconstruct_wide_rule_zero_gamma_and_kw_minus_one():
    df = _int_df([9999], [1], [2], [0.0], [-1], [_BIN_LABEL_UNKNOWN])
    result = reconstruct_bin_label(df, NZERO)
    assert result.tolist() == [0]


def test_reconstruct_ks_rule_matches_object_id_1():
    df = _int_df([NZERO + 5], [5], [6], [0.05], [0], [_BIN_LABEL_UNKNOWN])
    result = reconstruct_bin_label(df, NZERO)
    assert result.tolist() == [1]


def test_reconstruct_ks_rule_matches_object_id_2():
    df = _int_df([NZERO + 6], [5], [6], [0.05], [0], [_BIN_LABEL_UNKNOWN])
    result = reconstruct_bin_label(df, NZERO)
    assert result.tolist() == [1]


def test_reconstruct_galimit_fallback_rule():
    df = _int_df([42], [1], [2], [0.1], [0], [_BIN_LABEL_UNKNOWN])
    result = reconstruct_bin_label(df, NZERO)
    assert result.tolist() == [-1]


def test_reconstruct_unclassifiable_row_stays_unknown():
    df = _int_df([42], [1], [2], [0.05], [0], [_BIN_LABEL_UNKNOWN])
    result = reconstruct_bin_label(df, NZERO)
    assert result.tolist() == [_BIN_LABEL_UNKNOWN]


def test_reconstruct_wide_intercepted_before_ks_on_numeric_collision():
    """A row that satisfies both the wide pattern (G=0, cm_kw=-1) and the KS cm-name
    pattern (cm_id - NZERO == object_id_1) must resolve as wide -- rule order matters,
    see cm_id's column description in schemas/snapshot_binaries.yaml (IWBINC can
    collide with the KS name range)."""
    df = _int_df([NZERO + 5], [5], [6], [0.0], [-1], [_BIN_LABEL_UNKNOWN])
    result = reconstruct_bin_label(df, NZERO)
    assert result.tolist() == [0]


@pytest.mark.parametrize("true_label", [1, 0, -1])
def test_reconstruct_never_changes_a_real_label(true_label):
    """Even if the row's other fields would reconstruct to a different rule, an
    already-real bin_label must be returned untouched."""
    df = _int_df([-5], [1], [2], [0.0], [-1], [true_label])
    result = reconstruct_bin_label(df, NZERO)
    assert result.tolist() == [true_label]


def test_reconstruct_missing_columns_raises():
    df = pd.DataFrame({"bin_label": pd.array([_BIN_LABEL_UNKNOWN], dtype="int32")})
    with pytest.raises(ValueError, match="missing required columns"):
        reconstruct_bin_label(df, NZERO)


# ---------------------------------------------------------------------------
# resolve_nzero
# ---------------------------------------------------------------------------


def test_resolve_nzero_infers_from_ttot_zero_row():
    scalars_df = pd.DataFrame(
        {"simulation_id": ["sim", "sim"], "ttot": [0.0, 1.0], "n": [50000, 49998]}
    )
    assert resolve_nzero(scalars_df, "sim") == 50000


def test_resolve_nzero_override_takes_precedence():
    scalars_df = pd.DataFrame({"simulation_id": ["sim"], "ttot": [5.0], "n": [1]})
    assert resolve_nzero(scalars_df, "sim", override=12345) == 12345


def test_resolve_nzero_raises_when_ttot_min_not_zero():
    scalars_df = pd.DataFrame({"simulation_id": ["sim"], "ttot": [5.0], "n": [100]})
    with pytest.raises(ValueError, match="not 0.0"):
        resolve_nzero(scalars_df, "sim")


def test_resolve_nzero_raises_when_no_rows_for_simulation():
    scalars_df = pd.DataFrame({"simulation_id": ["other"], "ttot": [0.0], "n": [100]})
    with pytest.raises(ValueError, match="No cached snapshot_scalars"):
        resolve_nzero(scalars_df, "sim")


# ---------------------------------------------------------------------------
# backfill_parts / validate_reconstruction integration tests
# ---------------------------------------------------------------------------


def _scalars_df(ttot: float) -> pd.DataFrame:
    data = {key: [0.0] for key in SCALAR_KEYS}
    data["TTOT"] = [ttot]
    data["RBAR"] = [RBAR]
    data["ZMBAR"] = [1.0e6]
    data["VSTAR"] = [10.0]
    data["TSCALE"] = [TSCALE]
    data["RDENS(1)"], data["RDENS(2)"], data["RDENS(3)"] = [RDENS[0]], [RDENS[1]], [RDENS[2]]
    return pd.DataFrame(data)


def _binaries_df(ttot: float, rows: list[dict], *, include_bin_label: bool) -> pd.DataFrame:
    columns = list(SnapshotBinariesTask.columns_by_table["binaries"])
    if not include_bin_label:
        columns = [c for c in columns if c != "Bin Label"]
    full_rows = []
    for overrides in rows:
        row = {col: 0.0 for col in columns}
        row["TTOT"] = ttot
        row.update(overrides)
        full_rows.append(row)
    return pd.DataFrame(full_rows)


class FakeRawProcessor:
    """Minimal raw-reader-shaped fake: only what SnapshotBinariesTask needs."""

    def __init__(self, hdf5_paths: list[str], tables_by_path: dict[str, dict[str, pd.DataFrame]]):
        self.hdf5_paths = hdf5_paths
        self.tables_by_path = tables_by_path
        self.read_count = 0

    def get_all_hdf5_paths(self, *args, **kwargs):
        return self.hdf5_paths

    def read_raw_tables(self, hdf5_path, tables, columns_by_table=None):
        self.read_count += 1
        return {table: self.tables_by_path[hdf5_path][table] for table in tables}


# object_id_1 -> expected reconstructed bin_label for the "unknown" file's rows.
_UNKNOWN_ROWS = [
    {"Bin Name1": 1, "Bin Name2": 2, "Bin cm Name": -5, "Bin cm KW": 0, "Bin G": 0.05},  # merger
    {"Bin Name1": 3, "Bin Name2": 4, "Bin cm Name": 12000, "Bin cm KW": -1, "Bin G": 0.0},  # wide
    {
        "Bin Name1": 5,
        "Bin Name2": 6,
        "Bin cm Name": NZERO + 5,
        "Bin cm KW": 0,
        "Bin G": 0.05,
    },  # KS
    {"Bin Name1": 7, "Bin Name2": 8, "Bin cm Name": 42, "Bin cm KW": 0, "Bin G": 0.1},  # galimit
    {
        "Bin Name1": 9,
        "Bin Name2": 10,
        "Bin cm Name": 43,
        "Bin cm KW": 0,
        "Bin G": 0.07,
    },  # unresolved
]
_EXPECTED_UNKNOWN_LABELS = {1: -1, 3: 0, 5: 1, 7: -1, 9: _BIN_LABEL_UNKNOWN}

_KNOWN_ROWS = [
    {
        "Bin Name1": 60,
        "Bin Name2": 61,
        "Bin cm Name": NZERO + 60,
        "Bin Label": 1,
        "Bin cm KW": 0,
        "Bin G": 0.05,
    },
    {
        "Bin Name1": 70,
        "Bin Name2": 71,
        "Bin cm Name": 9999,
        "Bin Label": 0,
        "Bin cm KW": -1,
        "Bin G": 0.0,
    },
    {
        "Bin Name1": 80,
        "Bin Name2": 81,
        "Bin cm Name": -80,
        "Bin Label": -1,
        "Bin cm KW": 0,
        "Bin G": 0.05,
    },
]


def _build_binaries_cache(tmp_path: Path):
    config = make_config(tmp_path)
    file_known = str(tmp_path / "snap.40_1.h5part")
    file_unknown = str(tmp_path / "snap.40_2.h5part")
    Path(file_known).write_text("fake")
    Path(file_unknown).write_text("fake")

    tables = {
        file_known: {
            "scalars": _scalars_df(1.0),
            "binaries": _binaries_df(1.0, _KNOWN_ROWS, include_bin_label=True),
        },
        file_unknown: {
            "scalars": _scalars_df(2.0),
            "binaries": _binaries_df(2.0, _UNKNOWN_ROWS, include_bin_label=False),
        },
    }
    processor = FakeRawProcessor([file_known, file_unknown], tables)
    task = SnapshotBinariesTask(config, "sim")
    runner = HDF5ScanRunner(config, processor)
    runner.run("sim", [task], HDF5ScanOptions(wait_age_hour=0))
    return task, file_known, file_unknown


def test_backfill_parts_reconstructs_only_unknown_rows(tmp_path: Path):
    task, file_known, file_unknown = _build_binaries_cache(tmp_path)

    report = backfill_parts(task, {"sim": NZERO})

    combined = pd.read_parquet(task.data_dir)
    known = combined[combined["source_hdf5_path"] == file_known].set_index("object_id_1")
    assert known.loc[60, "bin_label"] == 1
    assert known.loc[70, "bin_label"] == 0
    assert known.loc[80, "bin_label"] == -1

    reconstructed = combined[combined["source_hdf5_path"] == file_unknown].set_index("object_id_1")
    for object_id_1, expected_label in _EXPECTED_UNKNOWN_LABELS.items():
        assert reconstructed.loc[object_id_1, "bin_label"] == expected_label

    assert report["simulation_id"] == "sim"
    assert report["nzero"] == NZERO
    (part_report,) = report["parts"].values()
    assert part_report["unknown_before"] == 5
    assert part_report["to_hard"] == 1
    assert part_report["to_wide"] == 1
    assert part_report["to_merger_or_hierarchical"] == 2
    assert part_report["still_unknown"] == 1
    assert part_report["written"] is True


def test_backfill_parts_skips_part_with_no_unknown_rows(tmp_path: Path):
    task, file_known, _file_unknown = _build_binaries_cache(tmp_path)

    report = backfill_parts(task, {"sim": NZERO})

    manifest = json.loads(task.manifest_path.read_text())
    known_part_name = manifest["processed_files"][file_known]["part"]
    assert known_part_name not in report["parts"]


def test_backfill_parts_does_not_create_new_part_files(tmp_path: Path):
    task, _file_known, _file_unknown = _build_binaries_cache(tmp_path)
    parts_before = sorted(p.name for p in task.data_dir.glob("*.parquet"))

    backfill_parts(task, {"sim": NZERO})

    parts_after = sorted(p.name for p in task.data_dir.glob("*.parquet"))
    assert parts_before == parts_after


def test_backfill_parts_preserves_schema_hash(tmp_path: Path):
    task, _file_known, _file_unknown = _build_binaries_cache(tmp_path)
    schema_hash_before = json.loads(task.manifest_path.read_text())["schema_hash"]

    backfill_parts(task, {"sim": NZERO})

    schema_hash_after = json.loads(task.manifest_path.read_text())["schema_hash"]
    assert schema_hash_before == schema_hash_after


def test_backfill_parts_dry_run_writes_nothing(tmp_path: Path):
    task, _file_known, file_unknown = _build_binaries_cache(tmp_path)
    manifest = json.loads(task.manifest_path.read_text())
    part_path = task.data_dir / manifest["processed_files"][file_unknown]["part"]
    mtime_before = part_path.stat().st_mtime_ns

    report = backfill_parts(task, {"sim": NZERO}, dry_run=True)

    assert part_path.stat().st_mtime_ns == mtime_before
    assert not (task.cache_dir / "backfill_bin_label.json").exists()
    combined = pd.read_parquet(task.data_dir)
    assert int((combined["bin_label"] == _BIN_LABEL_UNKNOWN).sum()) == 5
    assert report["dry_run"] is True


def test_backfill_parts_is_idempotent(tmp_path: Path):
    task, _file_known, file_unknown = _build_binaries_cache(tmp_path)
    backfill_parts(task, {"sim": NZERO})

    manifest = json.loads(task.manifest_path.read_text())
    part_path = task.data_dir / manifest["processed_files"][file_unknown]["part"]
    mtime_after_first = part_path.stat().st_mtime_ns

    report_second = backfill_parts(task, {"sim": NZERO})

    assert part_path.stat().st_mtime_ns == mtime_after_first
    (part_report,) = report_second["parts"].values()
    assert part_report["written"] is False


def test_backfill_parts_raises_when_nzero_missing(tmp_path: Path):
    task, _file_known, _file_unknown = _build_binaries_cache(tmp_path)
    with pytest.raises(KeyError):
        backfill_parts(task, {})


def test_backfill_parts_writes_provenance_sidecar(tmp_path: Path):
    task, _file_known, _file_unknown = _build_binaries_cache(tmp_path)

    backfill_parts(task, {"sim": NZERO})

    sidecar = json.loads((task.cache_dir / "backfill_bin_label.json").read_text())
    assert sidecar["algorithm_version"] == ALGORITHM_VERSION
    assert sidecar["nzero"] == NZERO
    assert "generated_at" in sidecar


def test_validate_reconstruction_matches_known_labels(tmp_path: Path):
    task, _file_known, _file_unknown = _build_binaries_cache(tmp_path)

    result = validate_reconstruction(task, NZERO)

    assert result["n_known"] == 3
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["nzero_consistent"] is True
