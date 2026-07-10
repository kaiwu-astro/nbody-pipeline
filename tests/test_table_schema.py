"""Tests for the VO-safe table schema registry."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from dragon3_pipelines.schemas import (
    SchemaValidationError,
    TableSchema,
    load_table_schema,
)

REGISTERED_TABLES = ["compact_object_history", "snapshot_summary"]

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VALID_DTYPES = {"int64", "float64", "float32", "bool", "string", "Int64"}


def _row_dataframe(schema: TableSchema, values: dict) -> pd.DataFrame:
    """Build a one-row DataFrame with per-column dtypes matching ``schema``."""
    data = {
        column.name: pd.array([values.get(column.name)], dtype=column.dtype)
        for column in schema.columns
    }
    return pd.DataFrame(data, columns=schema.column_names())


@pytest.mark.parametrize("table_name", REGISTERED_TABLES)
def test_schema_loads_with_vo_safe_columns(table_name: str) -> None:
    schema = load_table_schema(table_name)
    assert schema.table == table_name
    assert schema.columns
    for column in schema.columns:
        assert _NAME_RE.match(column.name), column.name
        assert column.dtype in _VALID_DTYPES


def test_load_table_schema_unknown_table_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_table_schema("does_not_exist")


_SNAPSHOT_SUMMARY_ROW = {
    "simulation_id": "20sb",
    "ttot": 1.0,
    "time_myr": 0.1,
    "n_singles": 100,
    "n_binaries": 10,
    "n_bh": 1,
    "n_ns": 2,
    "n_wd": 3,
    "total_mass_msun": 1000.0,
    "core_radius_pc": 0.5,
    "half_mass_radius_pc": 1.5,
    "rg_x_pc": 0.0,
    "rg_y_pc": 0.0,
    "rg_z_pc": 0.0,
    "vg_x_kmps": 0.0,
    "vg_y_kmps": 0.0,
    "vg_z_kmps": 0.0,
}


def test_validate_dataframe_pass() -> None:
    schema = load_table_schema("snapshot_summary")
    df = _row_dataframe(schema, _SNAPSHOT_SUMMARY_ROW)
    schema.validate_dataframe(df)


def test_validate_dataframe_missing_column() -> None:
    schema = load_table_schema("snapshot_summary")
    df = schema.empty_dataframe().drop(columns=["n_bh"])
    with pytest.raises(SchemaValidationError, match="column mismatch"):
        schema.validate_dataframe(df)


def test_validate_dataframe_extra_column() -> None:
    schema = load_table_schema("snapshot_summary")
    df = schema.empty_dataframe()
    df["unexpected_column"] = pd.array([], dtype="float64")
    with pytest.raises(SchemaValidationError, match="column mismatch"):
        schema.validate_dataframe(df)


def test_validate_dataframe_wrong_dtype() -> None:
    schema = load_table_schema("snapshot_summary")
    df = schema.empty_dataframe()
    df["n_bh"] = df["n_bh"].astype("string")
    with pytest.raises(SchemaValidationError, match="dtype family"):
        schema.validate_dataframe(df)


def test_validate_dataframe_non_nullable_with_null() -> None:
    schema = load_table_schema("snapshot_summary")
    row = dict(_SNAPSHOT_SUMMARY_ROW)
    row["total_mass_msun"] = None
    df = _row_dataframe(schema, row)
    with pytest.raises(SchemaValidationError, match="non-nullable"):
        schema.validate_dataframe(df)


def test_validate_dataframe_nullable_column_allows_null() -> None:
    schema = load_table_schema("compact_object_history")
    row = {
        "simulation_id": "20sb",
        "ttot": 1.0,
        "time_myr": 0.1,
        "object_id": 1,
        "kw": 14,
        "object_type": "BH",
        "mass_msun": 20.0,
        "x_pc": 0.0,
        "y_pc": 0.0,
        "z_pc": 0.0,
        "vx_kms": 0.0,
        "vy_kms": 0.0,
        "vz_kms": 0.0,
        "r_pc": 0.0,
        "is_in_binary": False,
        "source_hdf5_path": "/tmp/snap.h5part",
    }
    df = _row_dataframe(schema, row)
    schema.validate_dataframe(df)
    nullable_columns = {c.name for c in schema.columns if c.nullable}
    assert all(df.loc[0, name] is None or pd.isna(df.loc[0, name]) for name in nullable_columns)


def test_empty_dataframe_has_correct_dtypes() -> None:
    schema = load_table_schema("compact_object_history")
    df = schema.empty_dataframe()
    assert list(df.columns) == list(schema.column_names())
    assert len(df) == 0
    dtype_by_name = {column.name: column.dtype for column in schema.columns}
    for name in df.columns:
        expected_family = dtype_by_name[name]
        actual_kind = df[name].dtype
        if expected_family in ("int64", "Int64"):
            assert pd.api.types.is_integer_dtype(actual_kind)
        elif expected_family in ("float64", "float32"):
            assert pd.api.types.is_float_dtype(actual_kind)
        elif expected_family == "bool":
            assert pd.api.types.is_bool_dtype(actual_kind)
        elif expected_family == "string":
            assert pd.api.types.is_string_dtype(actual_kind)


def test_schema_hash_is_stable() -> None:
    schema = load_table_schema("snapshot_summary")
    assert schema.schema_hash() == schema.schema_hash()


def test_schema_hash_changes_with_dtype() -> None:
    from dragon3_pipelines.schemas import ColumnSchema, TableSchema

    base_column = ColumnSchema(
        name="mass_msun",
        dtype="float64",
        description="mass",
        unit="solMass",
        ucd=None,
        public=True,
        nullable=False,
    )
    schema_a = TableSchema(table="t", version=1, description="d", columns=(base_column,))
    schema_b = TableSchema(
        table="t",
        version=1,
        description="d",
        columns=(
            ColumnSchema(
                name="mass_msun",
                dtype="float32",
                description="mass",
                unit="solMass",
                ucd=None,
                public=True,
                nullable=False,
            ),
        ),
    )
    assert schema_a.schema_hash() != schema_b.schema_hash()
