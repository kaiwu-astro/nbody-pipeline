"""Tests for raw_dataframes_from_hdf5_file: the h5py-level raw HDF5 reader.

Builds small synthetic .h5part-shaped files with h5py (no dependency on real
simulation data) to verify column projection, source dtype preservation,
intra-file TTOT dedup, and missing-table handling -- independent of any
particular scan task.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from nbody_pipeline.io.text_parsers import SCALAR_KEYS, raw_dataframes_from_hdf5_file


def _scalar_array(overrides: dict[str, float]) -> np.ndarray:
    """Build a 100-slot float32 '000 Scalars' array (slots 71-100 unused/zero)."""
    arr = np.zeros(100, dtype="float32")
    for i, key in enumerate(SCALAR_KEYS):
        arr[i] = overrides.get(key, 0.0)
    return arr


def _write_step(
    group: h5py.Group,
    *,
    ttot: float,
    scalar_overrides: dict[str, float] | None = None,
    singles: dict[str, np.ndarray] | None = None,
    binaries: dict[str, np.ndarray] | None = None,
    binary_label_key: str = "176 Bin Label",
) -> None:
    overrides = {"TTOT": ttot, **(scalar_overrides or {})}
    n_star = len(next(iter(singles.values()))) if singles else 0
    group.attrs["Time"] = np.float32(ttot)
    group.attrs["N_STAR"] = np.int32(n_star)
    group.create_dataset("000 Scalars", data=_scalar_array(overrides))

    if singles is not None:
        dataset_key_of = {
            "Name": "032 Name",
            "KW": "031 KW",
            "M": "023 M",
            "X1": "001 X1",
            "X2": "002 X2",
            "X3": "003 X3",
            "V1": "004 V1",
            "V2": "005 V2",
            "V3": "006 V3",
            "POT": "025 POT",
            "L*": "027 L*",
            "Teff*": "028 Teff*",
        }
        for logical_name, values in singles.items():
            group.create_dataset(dataset_key_of[logical_name], data=np.asarray(values))

    if binaries is not None:
        dataset_key_of = {
            "Bin Name1": "161 Bin Name1",
            "Bin Name2": "162 Bin Name2",
            "Bin cm Name": "163 Bin cm Name",
            "Bin Label": binary_label_key,
            "Bin M1*": "123 Bin M1*",
        }
        for logical_name, values in binaries.items():
            group.create_dataset(dataset_key_of[logical_name], data=np.asarray(values))


def _write_h5part(path: Path, steps: list[dict]) -> None:
    with h5py.File(path, "w") as f:
        for index, step in enumerate(steps):
            group = f.create_group(f"Step#{index}")
            _write_step(group, **step)


def test_column_projection_reads_only_requested_columns_and_keeps_source_dtype(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snap.40_1.h5part"
    _write_h5part(
        path,
        [
            dict(
                ttot=1.0,
                singles={
                    "Name": np.array([10, 11], dtype="int32"),
                    "KW": np.array([1, 14], dtype="int32"),
                    "M": np.array([1.5, 20.0], dtype="float32"),
                    "L*": np.array([2.0, 3.0], dtype="float32"),
                },
            )
        ],
    )

    result = raw_dataframes_from_hdf5_file(
        str(path), tables=["singles"], columns_by_table={"singles": ["Name", "KW"]}
    )

    singles = result["singles"]
    assert list(singles.columns) == ["Name", "KW", "TTOT"]
    assert singles["Name"].dtype == np.int32
    assert singles["KW"].dtype == np.int32
    assert singles["Name"].tolist() == [10, 11]
    assert singles["TTOT"].tolist() == [1.0, 1.0]


def test_columns_by_table_none_reads_every_available_column(tmp_path: Path) -> None:
    path = tmp_path / "snap.40_1.h5part"
    _write_h5part(
        path,
        [
            dict(
                ttot=1.0,
                singles={
                    "Name": np.array([10], dtype="int32"),
                    "KW": np.array([1], dtype="int32"),
                    "M": np.array([1.5], dtype="float32"),
                },
            )
        ],
    )

    result = raw_dataframes_from_hdf5_file(str(path), tables=["singles"])

    assert set(result["singles"].columns) == {"Name", "KW", "M", "TTOT"}
    assert result["singles"]["M"].dtype == np.float32


def test_ttot_dedup_within_one_file_keeps_first_occurrence(tmp_path: Path) -> None:
    path = tmp_path / "snap.40_1.h5part"
    _write_h5part(
        path,
        [
            dict(ttot=1.0, singles={"Name": np.array([1, 2], dtype="int32")}),
            # Same TTOT again with different data -> must be dropped (matches
            # dataframes_from_hdf5_file's intra-file dedup).
            dict(ttot=1.0, singles={"Name": np.array([99], dtype="int32")}),
            dict(ttot=2.0, singles={"Name": np.array([3], dtype="int32")}),
        ],
    )

    result = raw_dataframes_from_hdf5_file(
        str(path), tables=["singles"], columns_by_table={"singles": ["Name"]}
    )

    singles = result["singles"]
    assert sorted(singles["TTOT"].unique().tolist()) == [1.0, 2.0]
    assert singles.loc[singles["TTOT"] == 1.0, "Name"].tolist() == [1, 2]
    assert singles.loc[singles["TTOT"] == 2.0, "Name"].tolist() == [3]


def test_missing_table_returns_empty_dataframe(tmp_path: Path) -> None:
    path = tmp_path / "snap.40_1.h5part"
    _write_h5part(
        path,
        [dict(ttot=1.0, singles={"Name": np.array([1], dtype="int32")})],
    )

    result = raw_dataframes_from_hdf5_file(str(path), tables=["scalars", "mergers"])

    assert result["mergers"].empty
    assert not result["scalars"].empty


def test_scalars_table_projection_matches_written_values(tmp_path: Path) -> None:
    path = tmp_path / "snap.40_1.h5part"
    _write_h5part(
        path,
        [dict(ttot=5.0, scalar_overrides={"RBAR": 2.5, "N_SINGLE": 3.0})],
    )

    result = raw_dataframes_from_hdf5_file(
        str(path), tables=["scalars"], columns_by_table={"scalars": ["RBAR", "N_SINGLE"]}
    )

    scalars = result["scalars"]
    assert set(scalars.columns) == {"RBAR", "N_SINGLE", "TTOT"}
    assert scalars["RBAR"].iloc[0] == pytest.approx(2.5)
    assert scalars["N_SINGLE"].iloc[0] == pytest.approx(3.0)
    assert scalars["TTOT"].iloc[0] == pytest.approx(5.0)


def test_binary_label_falls_back_to_legacy_dataset_name(tmp_path: Path) -> None:
    """Older archived runs store the 'Bin Label' quantity under dataset key
    '176 Bin cm Name' instead of '176 Bin Label'; the raw reader must still
    surface it under the 'Bin Label' logical name (not collide with the real
    '163 Bin cm Name' column)."""
    path = tmp_path / "snap.40_1.h5part"
    _write_h5part(
        path,
        [
            dict(
                ttot=1.0,
                binaries={
                    "Bin cm Name": np.array([500], dtype="int32"),
                    "Bin Label": np.array([1], dtype="int32"),
                },
                binary_label_key="176 Bin cm Name",
            )
        ],
    )

    result = raw_dataframes_from_hdf5_file(
        str(path), tables=["binaries"], columns_by_table={"binaries": ["Bin cm Name", "Bin Label"]}
    )

    binaries = result["binaries"]
    assert binaries["Bin cm Name"].tolist() == [500]
    assert binaries["Bin Label"].tolist() == [1]
