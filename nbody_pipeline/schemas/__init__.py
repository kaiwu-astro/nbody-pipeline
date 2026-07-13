"""
VO-safe table schema registry for the Parquet/DuckDB analysis layer.

Each persistent analysis table (see docs/analysis_architecture.md) has a
schema YAML file in this package describing its columns: name, dtype, unit,
UCD, description, visibility, and nullability. Column names must be VO-safe
identifiers (``^[a-z][a-z0-9_]*$``) so tables can be exported to VOTable/TAP
without renaming.

Validation here is intentionally lightweight: column set/order and dtype
*family* (int/float/bool/string) must match, and non-nullable columns must
not contain nulls. Units and UCDs are metadata only; no astropy unit
enforcement is performed at write time.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np
import pandas as pd
import yaml

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Allowed YAML dtype tokens -> pandas dtype string used to build empty frames.
_PANDAS_DTYPE = {
    "int64": "int64",
    "int32": "int32",
    "float64": "float64",
    "float32": "float32",
    "bool": "bool",
    "string": "string",
    "Int64": "Int64",
}


class SchemaValidationError(ValueError):
    """Raised when a DataFrame does not conform to a TableSchema."""


@dataclass(frozen=True)
class ColumnSchema:
    """Metadata for one column of a registered table."""

    name: str
    dtype: str
    description: str
    unit: str | None
    ucd: str | None
    public: bool
    nullable: bool


@dataclass(frozen=True)
class TableSchema:
    """Metadata for one registered table (its full column list)."""

    table: str
    version: int
    description: str
    columns: tuple[ColumnSchema, ...]

    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def empty_dataframe(self) -> pd.DataFrame:
        """Return an empty DataFrame with the correct dtype for every column."""
        data = {
            column.name: pd.array([], dtype=_PANDAS_DTYPE[column.dtype]) for column in self.columns
        }
        return pd.DataFrame(data, columns=self.column_names())

    def schema_hash(self) -> str:
        """Stable hash over (name, dtype) pairs; changes if columns/dtypes change."""
        payload = json.dumps([[column.name, column.dtype] for column in self.columns])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """Raise SchemaValidationError if ``df`` does not conform to this schema."""
        expected_names = self.column_names()
        actual_names = tuple(df.columns)
        if actual_names != expected_names:
            raise SchemaValidationError(
                f"{self.table}: column mismatch: expected {expected_names}, got {actual_names}"
            )
        errors: list[str] = []
        for column in self.columns:
            series = df[column.name]
            expected_family = _dtype_family(_PANDAS_DTYPE[column.dtype])
            actual_family = _dtype_family(series.dtype)
            if expected_family != actual_family:
                errors.append(
                    f"column {column.name!r}: expected dtype family "
                    f"{expected_family!r} ({column.dtype}), got {actual_family!r} ({series.dtype})"
                )
            elif not column.nullable and series.isna().any():
                errors.append(f"column {column.name!r} is non-nullable but contains nulls")
        if errors:
            raise SchemaValidationError(f"{self.table}: " + "; ".join(errors))


def _dtype_family(dtype: Any) -> str:
    """Classify a dtype (or dtype string) into int/float/bool/string."""
    pandas_dtype = pd.api.types.pandas_dtype(dtype) if isinstance(dtype, str) else dtype
    if pd.api.types.is_bool_dtype(pandas_dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(pandas_dtype):
        return "int"
    if pd.api.types.is_float_dtype(pandas_dtype):
        return "float"
    if pd.api.types.is_string_dtype(pandas_dtype) or pandas_dtype == np.dtype(object):
        return "string"
    raise SchemaValidationError(f"unsupported dtype: {pandas_dtype!r}")


@lru_cache(maxsize=None)
def load_table_schema(table_name: str) -> TableSchema:
    """Load and validate a table schema YAML from this package by table name."""
    resource = resources.files(__package__).joinpath(f"{table_name}.yaml")
    try:
        raw_text = resource.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No schema YAML registered for table {table_name!r}") from exc
    raw = yaml.safe_load(raw_text)
    return _build_table_schema(table_name, raw)


def _build_table_schema(table_name: str, raw: dict[str, Any]) -> TableSchema:
    if raw.get("table") != table_name:
        raise ValueError(
            f"{table_name}.yaml: 'table' key {raw.get('table')!r} does not match filename"
        )
    try:
        version = int(raw["version"])
        description = str(raw["description"])
        raw_columns = raw["columns"]
    except KeyError as exc:
        raise ValueError(f"{table_name}.yaml: missing required key {exc}") from exc
    if not raw_columns:
        raise ValueError(f"{table_name}.yaml: must define at least one column")

    columns: list[ColumnSchema] = []
    seen_names: set[str] = set()
    for entry in raw_columns:
        try:
            name = str(entry["name"])
            dtype = str(entry["dtype"])
            column_description = str(entry["description"])
            nullable = bool(entry["nullable"])
            public = bool(entry["public"])
        except KeyError as exc:
            raise ValueError(f"{table_name}.yaml: column missing required key {exc}") from exc
        if not _NAME_RE.match(name):
            raise ValueError(f"{table_name}.yaml: column name {name!r} is not VO-safe")
        if name in seen_names:
            raise ValueError(f"{table_name}.yaml: duplicate column name {name!r}")
        if dtype not in _PANDAS_DTYPE:
            raise ValueError(f"{table_name}.yaml: column {name!r} has unknown dtype {dtype!r}")
        seen_names.add(name)
        columns.append(
            ColumnSchema(
                name=name,
                dtype=dtype,
                description=column_description,
                unit=entry.get("unit"),
                ucd=entry.get("ucd"),
                public=public,
                nullable=nullable,
            )
        )
    return TableSchema(
        table=table_name, version=version, description=description, columns=tuple(columns)
    )


__all__ = [
    "ColumnSchema",
    "TableSchema",
    "SchemaValidationError",
    "load_table_schema",
]
