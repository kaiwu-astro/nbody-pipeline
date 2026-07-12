"""
DuckDB query entry point over the Parquet feature store.

Analysis tasks that use ``ParquetDatasetCacheMixin`` or ``ParquetTableCacheMixin``
(see ``nbody_pipeline.analysis.parquet_cache``) persist their output as Parquet
under ``analysis_cache_dir(config, simu_name, feature)``. This module is the
read path over that store: small/medium result sets go through DuckDB's
``read_parquet`` and come back as a pandas DataFrame, so callers never need to
know whether a feature is stored as a directory of per-file parts (dataset
style) or a single merged file (table style).

See docs/analysis_architecture.md for the caching-layer design.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import duckdb
import pandas as pd

from nbody_pipeline.analysis.cache_paths import (
    COMPACT_OBJECT_HISTORY_FEATURE,
    SNAPSHOT_BINARIES_FEATURE,
    SNAPSHOT_MERGERS_FEATURE,
    SNAPSHOT_SCALARS_FEATURE,
    SNAPSHOT_SINGLES_FEATURE,
    SNAPSHOT_SUMMARY_FEATURE,
    AnalysisCacheFeature,
    analysis_cache_dir,
)
from nbody_pipeline.schemas import load_table_schema

logger = logging.getLogger(__name__)

PARQUET_FEATURES: tuple[AnalysisCacheFeature, ...] = (
    COMPACT_OBJECT_HISTORY_FEATURE,
    SNAPSHOT_SUMMARY_FEATURE,
    SNAPSHOT_SINGLES_FEATURE,
    SNAPSHOT_BINARIES_FEATURE,
    SNAPSHOT_MERGERS_FEATURE,
    SNAPSHOT_SCALARS_FEATURE,
)


def feature_dataset_glob(config: Any, simu_name: str, feature: AnalysisCacheFeature) -> str:
    """Return a DuckDB-readable glob for ``feature``'s Parquet output.

    Dispatches on-disk: a dataset-style feature (``ParquetDatasetCacheMixin``)
    stores parts under a ``data/`` subdirectory; a table-style feature
    (``ParquetTableCacheMixin``) stores a single ``<feature>.parquet`` file
    directly in the feature's cache directory.
    """
    feature_dir = analysis_cache_dir(config, simu_name, feature)
    data_dir = feature_dir / "data"
    if data_dir.is_dir():
        return str(data_dir / "*.parquet")
    table_path = feature_dir / f"{feature}.parquet"
    if table_path.exists():
        return str(table_path)
    raise FileNotFoundError(
        f"No Parquet data found for feature {feature!r} in simulation {simu_name!r} "
        f"(looked in {feature_dir}). Build it first with "
        f"`python -m nbody_pipeline analyze --simu {simu_name}`."
    )


def load_feature(
    config: Any,
    simu_name: str,
    feature: AnalysisCacheFeature,
    *,
    columns: Sequence[str] | None = None,
    where: str | None = None,
    params: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Load rows from one feature's Parquet store for one simulation.

    ``where`` is a raw SQL boolean expression with ``?`` placeholders bound
    positionally from ``params`` (this is an internal research tool, not a
    public-facing filter DSL, so DuckDB's own parameter binding is enough).
    """
    glob = feature_dataset_glob(config, simu_name, feature)

    if columns is not None:
        valid_columns = load_table_schema(feature).column_names()
        unknown = [column for column in columns if column not in valid_columns]
        if unknown:
            raise ValueError(
                f"Unknown column(s) {unknown} for feature {feature!r}; "
                f"valid columns are {list(valid_columns)}"
            )
        select_clause = ", ".join(f'"{column}"' for column in columns)
    else:
        select_clause = "*"

    query = f"SELECT {select_clause} FROM read_parquet('{glob}')"
    if where:
        query += f" WHERE {where}"

    connection = duckdb.connect(database=":memory:")
    try:
        return connection.execute(query, list(params) if params else []).df()
    finally:
        connection.close()


def duckdb_connect(
    config: Any,
    *,
    simu_names: Iterable[str] | None = None,
    features: Iterable[AnalysisCacheFeature] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with one VIEW per feature.

    Each VIEW unions the requested (or all configured) simulations' Parquet
    output for that feature, so ``simulation_id`` is the column that
    disambiguates rows across simulations -- matching the VO/TAP convention
    the analysis architecture is built toward. A simulation/feature pair with
    no data on disk yet is skipped (logged), not an error, so an in-progress
    analysis run for one simulation doesn't block queries against others.
    """
    resolved_simu_names = list(simu_names) if simu_names is not None else list(config.pathof.keys())
    resolved_features = list(features) if features is not None else list(PARQUET_FEATURES)

    connection = duckdb.connect(database=":memory:")
    for feature in resolved_features:
        globs = []
        for simu_name in resolved_simu_names:
            try:
                globs.append(feature_dataset_glob(config, simu_name, feature))
            except FileNotFoundError:
                logger.info(
                    "Skipping feature %r for simulation %r: no Parquet data on disk yet",
                    feature,
                    simu_name,
                )
        if not globs:
            logger.info("Skipping feature %r: no simulation has data on disk yet", feature)
            continue
        glob_list = ", ".join(f"'{glob}'" for glob in globs)
        connection.execute(f"CREATE VIEW {feature} AS SELECT * FROM read_parquet([{glob_list}])")
    return connection


__all__ = ["PARQUET_FEATURES", "feature_dataset_glob", "load_feature", "duckdb_connect"]
