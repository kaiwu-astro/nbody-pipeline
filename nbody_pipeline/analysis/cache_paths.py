"""Helpers for analysis-produced cache directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

AnalysisCacheFeature = Literal[
    "particle_df",
    "primordial_binary",
    "b_type_binary",
    "binary_stellar_type",
    "current_lagrangian",
    "compact_binary_count",
    "galactic_orbit",
    "intermediate_mass_black_hole",
    "initial_total_mass",
    "compact_object_history",
    "snapshot_summary",
    "snapshot_singles",
    "snapshot_binaries",
    "snapshot_mergers",
    "snapshot_scalars",
]

PARTICLE_DF_FEATURE: AnalysisCacheFeature = "particle_df"
PRIMORDIAL_BINARY_FEATURE: AnalysisCacheFeature = "primordial_binary"
B_TYPE_BINARY_FEATURE: AnalysisCacheFeature = "b_type_binary"
BINARY_STELLAR_TYPE_FEATURE: AnalysisCacheFeature = "binary_stellar_type"
CURRENT_LAGRANGIAN_FEATURE: AnalysisCacheFeature = "current_lagrangian"
COMPACT_BINARY_COUNT_FEATURE: AnalysisCacheFeature = "compact_binary_count"
GALACTIC_ORBIT_FEATURE: AnalysisCacheFeature = "galactic_orbit"
INTERMEDIATE_MASS_BLACK_HOLE_FEATURE: AnalysisCacheFeature = "intermediate_mass_black_hole"
INITIAL_TOTAL_MASS_FEATURE: AnalysisCacheFeature = "initial_total_mass"
COMPACT_OBJECT_HISTORY_FEATURE: AnalysisCacheFeature = "compact_object_history"
SNAPSHOT_SUMMARY_FEATURE: AnalysisCacheFeature = "snapshot_summary"
SNAPSHOT_SINGLES_FEATURE: AnalysisCacheFeature = "snapshot_singles"
SNAPSHOT_BINARIES_FEATURE: AnalysisCacheFeature = "snapshot_binaries"
SNAPSHOT_MERGERS_FEATURE: AnalysisCacheFeature = "snapshot_mergers"
SNAPSHOT_SCALARS_FEATURE: AnalysisCacheFeature = "snapshot_scalars"

# Features routed to the second (particle-lake) storage root when configured
# (paths.lake_dir -> config.lake_dir_of), instead of analysis_cache_dir.
LAKE_FEATURES: frozenset[AnalysisCacheFeature] = frozenset(
    {
        SNAPSHOT_SINGLES_FEATURE,
        SNAPSHOT_BINARIES_FEATURE,
        SNAPSHOT_MERGERS_FEATURE,
        SNAPSHOT_SCALARS_FEATURE,
    }
)


def analysis_cache_dir(config: Any, simu_name: str, feature: AnalysisCacheFeature) -> Path:
    """Return ``<root>/<simu_name>/<feature>`` for analysis caches.

    ``feature`` in ``LAKE_FEATURES`` is routed to the second storage root
    (``config.lake_dir_of``) when configured, falling back to
    ``analysis_cache_dir_of`` otherwise (so lake tasks work unmodified in
    tests/configs that never set ``paths.lake_dir``).

    Lightweight tests and legacy external callers may still provide only
    ``particle_df_cache_dir_of``. In that case, preserve the old base path for
    particle history caches and append the requested feature for other caches.
    """
    if feature in LAKE_FEATURES:
        lake_dir_of = getattr(config, "lake_dir_of", None)
        if isinstance(lake_dir_of, Mapping) and lake_dir_of.get(simu_name):
            return Path(lake_dir_of[simu_name]) / feature

    analysis_cache_dir_of = getattr(config, "analysis_cache_dir_of", None)
    if isinstance(analysis_cache_dir_of, Mapping):
        return Path(analysis_cache_dir_of[simu_name]) / feature

    particle_cache_dir_of = getattr(config, "particle_df_cache_dir_of", None)
    if isinstance(particle_cache_dir_of, Mapping):
        base = Path(particle_cache_dir_of[simu_name])
        if feature == PARTICLE_DF_FEATURE:
            return base
        return base / feature

    raise AttributeError(
        "Config must define analysis_cache_dir_of or particle_df_cache_dir_of "
        "to resolve analysis cache paths."
    )
