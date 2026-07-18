"""HDF5 file reading and processing"""

import logging
import os
import time
import warnings
from glob import glob
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import astropy.constants as constants
import astropy.units as u

from nbody_pipeline.utils import log_time

logger = logging.getLogger(__name__)
_CONFIG_DEFAULT = object()
# Sentinel for binaries missing "Bin Label" (very old archived files predate that
# dataset) -- matches nbody_pipeline.analysis.particle_lake._BIN_LABEL_UNKNOWN.
_BIN_LABEL_UNKNOWN = -9


def _map_scalar_to_rows(
    ttot_series: pd.Series, scalar_df_all: pd.DataFrame, column: str
) -> pd.Series:
    """Broadcast a per-snapshot scalar column onto per-row TTOT.

    Uses the same ``TTOT.map(scalar_df_all[col])`` idiom as the existing
    RDENS broadcast (see ``read_file`` ~:309-314). If ``column`` is missing
    (old archived files, or minimal mocks in tests), returns an all-NaN
    series and logs a warning instead of raising.
    """
    if column not in scalar_df_all.columns:
        logger.warning(
            "[hdf5_reader] scalars table missing column %r; filling with NaN "
            "(binary_class/mean_core_interparticle_distance[au]/Ebind/kT columns "
            "will be affected).",
            column,
        )
        return pd.Series(np.nan, index=ttot_series.index)
    return ttot_series.map(scalar_df_all[column])


# ---------------------------------------------------------------------------
# Lake-first raw-table reconstruction.
#
# The old L1 cache (``{hdf5_path}.{table}.df.feather``, existence-only
# invalidated, see docs/analysis_architecture.md's retired Roadmap #1/#2) has
# been retired in favor of the parquet particle lake
# (``nbody_pipeline.analysis.particle_lake``): every raw column ``read_file``
# needs for its derived-column block already lives in
# snapshot_singles/binaries/mergers/scalars, keyed per source file by
# ``source_hdf5_path``. These helpers reconstruct DataFrames with the *old
# raw HDF5 column names* (``X1``, ``Bin Name1``, ``RDENS(1)``, ...) from the
# lake, so that ``read_file``'s derived-column computation below runs
# completely unchanged on top of them -- this is deliberate: it minimizes the
# risk of a numeric regression versus re-deriving the same quantities with
# new code. Density-center-corrected positions (``x_pc``/``cm_x_pc``/...) are
# inverted back to raw N-body-unit coordinates using the same
# RDENS-times-RBAR offset formula the lake used to compute them in the first
# place (see ``nbody_pipeline.analysis.particle_lake._rdens_corrected_pc``),
# so the round trip is exact in floating point (``(a - b) + b == a``).
#
# Force-derivative/integrator-state columns (``A1-3``, ``AD1-3``, ``D21-33``,
# ``STEP``, ``STEPR``, ``T0``, ``T0R``, ``NB-Sph``, ``TEV``, ``TEV0``) are not
# stored in the lake and are not reconstructed here: confirmed zero
# consumers repo-wide (only ``particle_tracker.py`` used to pass them through
# untouched), so this is an accepted, user-approved column drop rather than a
# gap.
_SCALARS_LAKE_TO_RAW: Dict[str, str] = {
    "ttot": "TTOT",
    "npairs": "NPAIRS",
    "rbar_pc": "RBAR",
    "zmbar_msun": "ZMBAR",
    "n": "N",
    "tstar_myr": "TSTAR",
    "rdens_x_nb": "RDENS(1)",
    "rdens_y_nb": "RDENS(2)",
    "rdens_z_nb": "RDENS(3)",
    "ttot_over_tcr0": "TTOT/TCR0",
    "tscale_myr": "TSCALE",
    "vstar_kms": "VSTAR",
    "rc_nb": "RC",
    "nc": "NC",
    "vc_nb": "VC",
    "rhom_nb": "RHOM",
    "cmax": "CMAX",
    "rscale_nb": "RSCALE",
    "rsmin_nb": "RSMIN",
    "dmin1_nb": "DMIN1",
    "rg_x_pc": "RG(1)",
    "rg_y_pc": "RG(2)",
    "rg_z_pc": "RG(3)",
    "vg_x_kmps": "VG(1)",
    "vg_y_kmps": "VG(2)",
    "vg_z_kmps": "VG(3)",
    "tidal_1": "TIDAL(1)",
    "tidal_2": "TIDAL(2)",
    "tidal_3": "TIDAL(3)",
    "tidal_4": "TIDAL(4)",
    "gmg": "GMG",
    "omega": "OMEGA",
    "disk": "DISK",
    "disk_a": "A",
    "disk_b": "B",
    "zmet": "ZMET",
    **{f"zpars_{i}": f"ZPARS({i})" for i in range(1, 21)},
    "etai": "ETAI",
    "etar": "ETAR",
    "etau": "ETAU",
    "eclose_nb": "ECLOSE",
    "dtmin_nb": "DTMIN",
    "rmin_nb": "RMIN",
    "gmin": "GMIN",
    "gmax": "GMAX",
    "smax": "SMAX",
    "nnbopt": "NNBOPT",
    "epoch0_myr": "EPOCH0",
    "n_single": "N_SINGLE",
    "n_binary": "N_BINARY",
    "n_merger": "N_MERGER",
}

_SINGLES_LAKE_TO_RAW_DIRECT: Dict[str, str] = {
    "object_id": "Name",
    "kw": "KW",
    "type_code": "Type",
    "mass_msun": "M",
    "vx_kms": "V1",
    "vy_kms": "V2",
    "vz_kms": "V3",
    "pot_nb": "POT",
    "radius_rsun": "R*",
    "luminosity_lsun": "L*",
    "teff_k": "Teff*",
    "core_radius_rsun": "RC*",
    "core_mass_msun": "MC*",
    "spin_aspn": "ASPN",
    "epoch_myr": "EPOCH",
    "ttot": "TTOT",
}

_BINARIES_LAKE_TO_RAW_DIRECT: Dict[str, str] = {
    "object_id_1": "Bin Name1",
    "object_id_2": "Bin Name2",
    "cm_id": "Bin cm Name",
    "kw_1": "Bin KW1",
    "kw_2": "Bin KW2",
    "cm_kw": "Bin cm KW",
    "bin_label": "Bin Label",
    "mass_1_msun": "Bin M1*",
    "mass_2_msun": "Bin M2*",
    "cm_vx_kms": "Bin cm V1",
    "cm_vy_kms": "Bin cm V2",
    "cm_vz_kms": "Bin cm V3",
    "rel_x_pc": "Bin rel X1",
    "rel_y_pc": "Bin rel X2",
    "rel_z_pc": "Bin rel X3",
    "rel_vx_kms": "Bin rel V1",
    "rel_vy_kms": "Bin rel V2",
    "rel_vz_kms": "Bin rel V3",
    "cm_pot_nb": "Bin POT",
    "semi_major_axis_au": "Bin A[au]",
    "eccentricity": "Bin ECC",
    "period_days": "Bin P[d]",
    "pert_gamma": "Bin G",
    "radius_1_rsun": "Bin RS1*",
    "radius_2_rsun": "Bin RS2*",
    "luminosity_1_lsun": "Bin L1*",
    "luminosity_2_lsun": "Bin L2*",
    "teff_1_k": "Bin Teff1*",
    "teff_2_k": "Bin Teff2*",
    "core_radius_1_rsun": "Bin RC1*",
    "core_radius_2_rsun": "Bin RC2*",
    "core_mass_1_msun": "Bin MC1*",
    "core_mass_2_msun": "Bin MC2*",
    "spin_aspn_1": "ASPN1",
    "spin_aspn_2": "ASPN2",
    "epoch_1_myr": "EPOCH1",
    "epoch_2_myr": "EPOCH2",
    "ttot": "TTOT",
}

_MERGERS_LAKE_TO_RAW_DIRECT: Dict[str, str] = {
    "object_id_1": "Mer NAM1",
    "object_id_2": "Mer NAM2",
    "object_id_3": "Mer NAM3",
    "cm_id": "Mer NAMC",
    "kw_1": "Mer KW1",
    "kw_2": "Mer KW2",
    "kw_3": "Mer KW3",
    "cm_kw": "Mer KWC",
    "mass_1_msun": "Mer M1",
    "mass_2_msun": "Mer M2",
    "mass_3_msun": "Mer M3",
    "cm_vx_kms": "Mer VC1",
    "cm_vy_kms": "Mer VC2",
    "cm_vz_kms": "Mer VC3",
    "rel0_x_pc": "Mer XR01",
    "rel0_y_pc": "Mer XR02",
    "rel0_z_pc": "Mer XR03",
    "rel0_vx_kms": "Mer VR01",
    "rel0_vy_kms": "Mer VR02",
    "rel0_vz_kms": "Mer VR03",
    "rel1_x_pc": "Mer XR11",
    "rel1_y_pc": "Mer XR12",
    "rel1_z_pc": "Mer XR13",
    "rel1_vx_kms": "Mer VR11",
    "rel1_vy_kms": "Mer VR12",
    "rel1_vz_kms": "Mer VR13",
    "cm_pot_nb": "Mer POT",
    "radius_1_rsun": "Mer RS1",
    "radius_2_rsun": "Mer RS2",
    "radius_3_rsun": "Mer RS3",
    "luminosity_1_lsun": "Mer L1",
    "luminosity_2_lsun": "Mer L2",
    "luminosity_3_lsun": "Mer L3",
    "teff_1_k": "Mer TE1",
    "teff_2_k": "Mer TE2",
    "teff_3_k": "Mer TE3",
    "core_radius_1_rsun": "Mer RC1",
    "core_radius_2_rsun": "Mer RC2",
    "core_radius_3_rsun": "Mer RC3",
    "core_mass_1_msun": "Mer MC1",
    "core_mass_2_msun": "Mer MC2",
    "core_mass_3_msun": "Mer MC3",
    "semi_major_axis_0_au": "Mer A0[au]",
    "eccentricity_0": "Mer ECC0",
    "period_0_days": "Mer P0[d]",
    "semi_major_axis_1_au": "Mer A1[au]",
    "eccentricity_1": "Mer ECC1",
    "period_1_days": "Mer P1[d]",
    "ttot": "TTOT",
}


# Position-column maps (raw HDF5 column -> lake column, in RDENS(1)/(2)/(3) axis
# order) shared by ``_object_raw_from_lake`` and ``_lake_columns_for_request``.
_SINGLES_POS_MAP: Dict[str, str] = {"X1": "x_pc", "X2": "y_pc", "X3": "z_pc"}
_BINARIES_POS_MAP: Dict[str, str] = {
    "Bin cm X1": "cm_x_pc",
    "Bin cm X2": "cm_y_pc",
    "Bin cm X3": "cm_z_pc",
}
_MERGERS_POS_MAP: Dict[str, str] = {
    "Mer XC1": "cm_x_pc",
    "Mer XC2": "cm_y_pc",
    "Mer XC3": "cm_z_pc",
}

# raw -> lake reverse maps, generated from the *_LAKE_TO_RAW* dicts above, used by
# read_raw_tables' lake-first path to translate a caller's requested raw columns
# into the lake columns to load.
_SINGLES_RAW_TO_LAKE_DIRECT: Dict[str, str] = {v: k for k, v in _SINGLES_LAKE_TO_RAW_DIRECT.items()}
_BINARIES_RAW_TO_LAKE_DIRECT: Dict[str, str] = {
    v: k for k, v in _BINARIES_LAKE_TO_RAW_DIRECT.items()
}
_MERGERS_RAW_TO_LAKE_DIRECT: Dict[str, str] = {v: k for k, v in _MERGERS_LAKE_TO_RAW_DIRECT.items()}
_SCALARS_RAW_TO_LAKE: Dict[str, str] = {v: k for k, v in _SCALARS_LAKE_TO_RAW.items()}

_OBJECT_TABLE_RAW_TO_LAKE: Dict[str, Dict[str, str]] = {
    "singles": {**_SINGLES_RAW_TO_LAKE_DIRECT, **_SINGLES_POS_MAP},
    "binaries": {**_BINARIES_RAW_TO_LAKE_DIRECT, **_BINARIES_POS_MAP},
    "mergers": {**_MERGERS_RAW_TO_LAKE_DIRECT, **_MERGERS_POS_MAP},
}


def _scalars_raw_from_lake(lake_scalars: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the raw ``scalars`` table (indexed by TTOT) from a lake slice."""
    raw = lake_scalars.rename(columns=_SCALARS_LAKE_TO_RAW)
    raw = raw[list(_SCALARS_LAKE_TO_RAW.values())]
    return raw.set_index("TTOT", drop=False)


def _rdens_pc_from_raw_scalars(raw_scalars: pd.DataFrame) -> pd.DataFrame:
    """(TTOT -> RDENS*RBAR in pc), same formula as ``read_file``'s own offsets."""
    from nbody_pipeline.io.text_parsers import get_scale_dict_from_hdf5_df

    scale = get_scale_dict_from_hdf5_df(raw_scalars)
    return raw_scalars[["RDENS(1)", "RDENS(2)", "RDENS(3)"]] * scale["r"]


def _invert_rdens_offset(
    pc_values: pd.Series, ttot: pd.Series, rdens_pc_column: pd.Series
) -> pd.Series:
    """raw_nb = pc_value + offset(ttot); exact inverse of the lake's forward correction."""
    offsets = ttot.map(rdens_pc_column)
    return pc_values.astype("float64") + offsets.to_numpy(dtype="float64")


_SINGLES_RAW_COLUMNS = [*_SINGLES_LAKE_TO_RAW_DIRECT.values(), "X1", "X2", "X3"]
_BINARIES_RAW_COLUMNS = [
    *_BINARIES_LAKE_TO_RAW_DIRECT.values(),
    "Bin cm X1",
    "Bin cm X2",
    "Bin cm X3",
]
_MERGERS_RAW_COLUMNS = [*_MERGERS_LAKE_TO_RAW_DIRECT.values(), "Mer XC1", "Mer XC2", "Mer XC3"]


def _object_raw_from_lake(
    lake_df: pd.DataFrame,
    rdens_pc: pd.DataFrame,
    *,
    direct_map: Dict[str, str],
    pos_map: Dict[str, str],
    all_raw_columns: Sequence[str],
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Rebuild a raw singles/binaries/mergers table from a lake slice.

    ``columns=None`` reconstructs every raw column (``all_raw_columns``, in
    the fixed order the retired L1 cache used). Given ``columns``, only the
    requested position columns (from ``pos_map``) are inverted -- other
    requested columns are just renamed -- and the result's column order
    follows ``raw_dataframes_from_hdf5_file``'s convention: requested order
    with ``TTOT`` dropped from wherever it was and appended last. An empty
    ``lake_df`` (this file has no rows for this table) returns an empty
    DataFrame carrying the same column names, matching the raw-HDF5 fallback.
    """
    if columns is None:
        output_columns = list(all_raw_columns)
    else:
        output_columns = [col for col in dict.fromkeys(columns) if col != "TTOT"] + ["TTOT"]

    if lake_df.empty:
        return pd.DataFrame(columns=output_columns)

    raw = lake_df.rename(columns=direct_map)
    ttot = lake_df["ttot"]
    for axis_idx, (raw_col, lake_col) in enumerate(pos_map.items(), start=1):
        if columns is not None and raw_col not in columns:
            continue
        raw[raw_col] = _invert_rdens_offset(lake_df[lake_col], ttot, rdens_pc[f"RDENS({axis_idx})"])
    return raw[output_columns].reset_index(drop=True)


def _singles_raw_from_lake(lake_singles: pd.DataFrame, rdens_pc: pd.DataFrame) -> pd.DataFrame:
    return _object_raw_from_lake(
        lake_singles,
        rdens_pc,
        direct_map=_SINGLES_LAKE_TO_RAW_DIRECT,
        pos_map=_SINGLES_POS_MAP,
        all_raw_columns=_SINGLES_RAW_COLUMNS,
    )


def _binaries_raw_from_lake(lake_binaries: pd.DataFrame, rdens_pc: pd.DataFrame) -> pd.DataFrame:
    return _object_raw_from_lake(
        lake_binaries,
        rdens_pc,
        direct_map=_BINARIES_LAKE_TO_RAW_DIRECT,
        pos_map=_BINARIES_POS_MAP,
        all_raw_columns=_BINARIES_RAW_COLUMNS,
    )


def _mergers_raw_from_lake(lake_mergers: pd.DataFrame, rdens_pc: pd.DataFrame) -> pd.DataFrame:
    return _object_raw_from_lake(
        lake_mergers,
        rdens_pc,
        direct_map=_MERGERS_LAKE_TO_RAW_DIRECT,
        pos_map=_MERGERS_POS_MAP,
        all_raw_columns=_MERGERS_RAW_COLUMNS,
    )


def _load_lake_slice(
    config_manager: Any,
    simu_name: str,
    feature: Any,
    hdf5_path: str,
    *,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load one feature's rows for one source HDF5 file; empty DataFrame if absent.

    Only catches ``FileNotFoundError`` (routine: this feature's Parquet
    dataset hasn't been written for this simulation, or has no rows for this
    file). Callers that need to detect "no lake at all for this simulation"
    (a broader exception set -- ``AttributeError``/``KeyError``/``TypeError``
    from a ``config_manager`` with no cache-dir mapping) must catch those
    themselves around their first lake call.
    """
    from nbody_pipeline.query import load_feature

    try:
        return load_feature(
            config_manager,
            simu_name,
            feature,
            columns=columns,
            where="source_hdf5_path = ?",
            params=[hdf5_path],
        )
    except FileNotFoundError:
        return pd.DataFrame()


def _raw_tables_from_lake(
    config_manager: Any, simu_name: Optional[str], hdf5_path: str
) -> Optional[Dict[str, pd.DataFrame]]:
    """Reconstruct raw scalars/singles/binaries/mergers tables from the parquet lake.

    Returns ``None`` (caller should fall back to parsing the HDF5 file
    directly) if this simulation has no lake built yet, or if this specific
    ``hdf5_path`` has no rows in ``snapshot_scalars`` (not scanned into the
    lake yet, or every one of its TTOT lost the cross-file dedup tie-break --
    see ``nbody_pipeline.analysis.particle_lake.compute_ttot_dedup_exclusions``).
    A file that *is* represented in the lake but legitimately has zero
    binaries/mergers at every one of its snapshots still returns normally
    (with an empty binaries/mergers table), since only ``scalars`` is
    guaranteed non-empty for a scanned file.
    """
    if simu_name is None:
        return None

    from nbody_pipeline.query import load_feature
    from nbody_pipeline.analysis.cache_paths import (
        SNAPSHOT_BINARIES_FEATURE,
        SNAPSHOT_MERGERS_FEATURE,
        SNAPSHOT_SCALARS_FEATURE,
        SNAPSHOT_SINGLES_FEATURE,
    )

    try:
        lake_scalars = load_feature(
            config_manager,
            simu_name,
            SNAPSHOT_SCALARS_FEATURE,
            where="source_hdf5_path = ?",
            params=[hdf5_path],
        )
    except (FileNotFoundError, AttributeError, KeyError, TypeError):
        # No lake for this simulation: no Parquet on disk yet (FileNotFoundError), or
        # config_manager doesn't even define a cache-dir mapping / this simu_name isn't
        # in it (AttributeError/KeyError from analysis_cache_dir -- routine for
        # lightweight test doubles and callers that never touch the lake). Any of these
        # just means "fall back to parsing the HDF5 file directly", never a hard error.
        return None
    if lake_scalars.empty:
        return None

    raw_scalars = _scalars_raw_from_lake(lake_scalars)
    rdens_pc = _rdens_pc_from_raw_scalars(raw_scalars)

    lake_singles = _load_lake_slice(config_manager, simu_name, SNAPSHOT_SINGLES_FEATURE, hdf5_path)
    lake_binaries = _load_lake_slice(
        config_manager, simu_name, SNAPSHOT_BINARIES_FEATURE, hdf5_path
    )
    lake_mergers = _load_lake_slice(config_manager, simu_name, SNAPSHOT_MERGERS_FEATURE, hdf5_path)

    return {
        "scalars": raw_scalars,
        "singles": _singles_raw_from_lake(lake_singles, rdens_pc),
        "binaries": _binaries_raw_from_lake(lake_binaries, rdens_pc),
        "mergers": _mergers_raw_from_lake(lake_mergers, rdens_pc),
    }


def _lake_columns_for_request(table: str, raw_columns: Sequence[str]) -> Optional[set]:
    """Map requested raw columns for one object table to the lake columns to load.

    Always includes ``"ttot"`` (needed to invert position-column offsets and
    to attach the ``TTOT`` column every raw row carries). Returns ``None`` if
    any requested column has no lake equivalent -- a force-derivative/
    integrator column dropped by the L1-cache-retirement lake migration (see
    the module docstring above ``_raw_tables_from_lake``) -- signalling the
    caller to fall back to parsing the HDF5 file for the whole request.
    """
    raw_to_lake = _OBJECT_TABLE_RAW_TO_LAKE[table]
    lake_columns = {"ttot"}
    for raw_col in raw_columns:
        if raw_col == "TTOT":
            continue
        lake_col = raw_to_lake.get(raw_col)
        if lake_col is None:
            return None
        lake_columns.add(lake_col)
    return lake_columns


def _projected_raw_tables_from_lake(
    config_manager: Any,
    simu_name: Optional[str],
    hdf5_path: str,
    tables: Sequence[str],
    columns_by_table: Optional[Mapping[str, Optional[Sequence[str]]]],
) -> Optional[Dict[str, pd.DataFrame]]:
    """Lake-first, column-projected counterpart to ``raw_dataframes_from_hdf5_file``.

    Backs ``read_raw_tables``'s lake-first path (unlike ``_raw_tables_from_lake``,
    which backs ``read_file`` and always reconstructs every raw column). Returns
    ``None`` (caller falls back to parsing the HDF5 file directly) if this
    simulation/file has no lake data yet, or if any requested column across any
    requested table has no lake equivalent.

    The mappability precheck (before any lake query) is deliberately
    all-or-nothing across every requested table: mixing a lake read for one
    table with an HDF5 read for another would need two separate file opens/
    TTOT-dedup passes for no real benefit, since every ``read_raw_tables``
    caller requests columns from only a handful of tables at once.
    """
    if simu_name is None:
        return None

    from nbody_pipeline.analysis.cache_paths import (
        SNAPSHOT_BINARIES_FEATURE,
        SNAPSHOT_MERGERS_FEATURE,
        SNAPSHOT_SCALARS_FEATURE,
        SNAPSHOT_SINGLES_FEATURE,
    )

    requested_tables = list(dict.fromkeys(tables))
    columns_by_table = columns_by_table or {}
    object_tables = [table for table in requested_tables if table in _OBJECT_TABLE_RAW_TO_LAKE]

    lake_columns_by_table: Dict[str, Optional[set]] = {}
    for table in object_tables:
        raw_columns = columns_by_table.get(table)
        if raw_columns is None:
            lake_columns_by_table[table] = None
            continue
        lake_columns = _lake_columns_for_request(table, raw_columns)
        if lake_columns is None:
            return None
        lake_columns_by_table[table] = lake_columns

    from nbody_pipeline.query import load_feature

    try:
        lake_scalars = load_feature(
            config_manager,
            simu_name,
            SNAPSHOT_SCALARS_FEATURE,
            where="source_hdf5_path = ?",
            params=[hdf5_path],
        )
    except (FileNotFoundError, AttributeError, KeyError, TypeError, ValueError):
        return None
    if lake_scalars.empty:
        return None

    raw_scalars = _scalars_raw_from_lake(lake_scalars).reset_index(drop=True)
    rdens_pc = _rdens_pc_from_raw_scalars(raw_scalars.set_index("TTOT", drop=False))

    result: Dict[str, pd.DataFrame] = {}
    if "scalars" in requested_tables:
        requested_scalar_cols = columns_by_table.get("scalars")
        if requested_scalar_cols is None:
            result["scalars"] = raw_scalars
        else:
            ordered = [col for col in dict.fromkeys(requested_scalar_cols) if col != "TTOT"] + [
                "TTOT"
            ]
            result["scalars"] = raw_scalars[ordered]

    feature_by_table = {
        "singles": SNAPSHOT_SINGLES_FEATURE,
        "binaries": SNAPSHOT_BINARIES_FEATURE,
        "mergers": SNAPSHOT_MERGERS_FEATURE,
    }
    reconstruct_by_table = {
        "singles": (_SINGLES_LAKE_TO_RAW_DIRECT, _SINGLES_POS_MAP, _SINGLES_RAW_COLUMNS),
        "binaries": (_BINARIES_LAKE_TO_RAW_DIRECT, _BINARIES_POS_MAP, _BINARIES_RAW_COLUMNS),
        "mergers": (_MERGERS_LAKE_TO_RAW_DIRECT, _MERGERS_POS_MAP, _MERGERS_RAW_COLUMNS),
    }
    for table in object_tables:
        lake_columns = lake_columns_by_table[table]
        try:
            lake_slice = _load_lake_slice(
                config_manager,
                simu_name,
                feature_by_table[table],
                hdf5_path,
                columns=sorted(lake_columns) if lake_columns is not None else None,
            )
        except ValueError:
            return None
        direct_map, pos_map, all_raw_columns = reconstruct_by_table[table]
        result[table] = _object_raw_from_lake(
            lake_slice,
            rdens_pc,
            direct_map=direct_map,
            pos_map=pos_map,
            all_raw_columns=all_raw_columns,
            columns=columns_by_table.get(table),
        )

    return {table: result[table] for table in requested_tables if table in result}


class HDF5FileProcessor:
    """Read and preprocess HDF5 data for plotting"""

    def __init__(self, config_manager):
        self.config = config_manager

    def read_tables(
        self,
        hdf5_path: str,
        simu_name: Optional[str],
        tables: Sequence[str],
        columns_by_table: Optional[Mapping[str, Sequence[str] | None]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Read selected processed HDF5 tables (thin subset of ``read_file``).

        ``read_file`` already resolves its raw tables lake-first (falling
        back to parsing the HDF5 file only when this file isn't in the lake
        yet), so this is just a pass-through: run ``read_file`` once, then
        project down to the requested tables/columns from the in-memory
        result.
        """
        columns_by_table = columns_by_table or {}
        requested_tables = list(dict.fromkeys(tables))
        full_df_dict = self.read_file(hdf5_path, simu_name)
        result: Dict[str, pd.DataFrame] = {}
        for table in requested_tables:
            df = full_df_dict.get(table, pd.DataFrame())
            columns = columns_by_table.get(table)
            if columns is not None and not df.empty:
                df = df[[col for col in columns if col in df.columns]]
            result[table] = df
        return result

    def read_raw_tables(
        self,
        hdf5_path: str,
        tables: Sequence[str],
        columns_by_table: Optional[Mapping[str, Sequence[str] | None]] = None,
        *,
        simu_name: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Read selected raw (original column names/dtypes, no derived columns)
        HDF5 tables, lake-first with a source-HDF5 fallback.

        Never applies ``read_file``'s derived columns or NS/BH display
        clipping -- for scan tasks/callers that need original source values
        (``hdf5_reader_kind = "raw"``, see ``nbody_pipeline.analysis.hdf5_scan``).

        ``simu_name`` (non-``None``) tries the particle lake first
        (``nbody_pipeline.analysis.particle_lake``): a projected reconstruction
        of exactly the requested tables/columns from
        snapshot_scalars/singles/binaries/mergers, offset-inverted back to raw
        N-body-unit values (see ``_projected_raw_tables_from_lake``). Falls back
        to parsing ``hdf5_path`` directly (``raw_dataframes_from_hdf5_file``,
        with real h5py-level column projection) whenever this file isn't in the
        lake yet, or any requested column has no lake equivalent -- a
        force-derivative/integrator column (``A1-3``, ``STEP``, ``T0``, ...)
        dropped by the L1-cache-retirement lake migration, confirmed zero
        consumers repo-wide (see the module docstring above
        ``_raw_tables_from_lake``). ``simu_name=None`` skips the lake lookup
        entirely and always parses the HDF5 file -- this is what the particle-lake
        build tasks themselves use (``hdf5_reader_kind = "source"``), so that
        building the lake never reads back from the (possibly stale/incomplete)
        lake it is building.

        A ``columns_by_table[table] is None`` request means "all columns of
        that table" -- post-migration, that means the retired L1 cache's raw
        column set (``_SINGLES_RAW_COLUMNS``/``_BINARIES_RAW_COLUMNS``/
        ``_MERGERS_RAW_COLUMNS``), not literally every original HDF5 dataset,
        whichever path serves the request (lake or HDF5 fallback both apply
        this projection consistently for a "full" request).

        Stale-lake semantics: if the source HDF5 file changes after being
        scanned into the lake (e.g. a restarted job overwrites it), a
        lake-first read here returns the lake's (stale) snapshot of that file,
        not the current on-disk content, until the lake is rebuilt -- the same
        already-accepted semantics as ``read_file``'s lake-first raw tables
        (see its docstring).
        """
        from nbody_pipeline.io.text_parsers import raw_dataframes_from_hdf5_file

        if simu_name is not None:
            lake_result = _projected_raw_tables_from_lake(
                self.config, simu_name, hdf5_path, tables, columns_by_table
            )
            if lake_result is not None:
                logger.info(
                    "[hdf5-lake] Sourced raw tables from the particle lake for %s", hdf5_path
                )
                return lake_result

        return raw_dataframes_from_hdf5_file(
            hdf5_path, tables=tables, columns_by_table=columns_by_table
        )

    def read_step_times(self, hdf5_path: str) -> List[float]:
        """Every Step# group's TTOT for this file, via attrs only (no dataset reads).

        Thin wrapper around ``nbody_pipeline.io.text_parsers.read_step_times``,
        exposed here (rather than called directly) so callers that need it
        injected -- e.g. ``ParticleLakeProcessor``'s cross-file TTOT dedup --
        can substitute a fake processor in tests the same way they already do
        for ``read_tables``/``read_raw_tables``.
        """
        from nbody_pipeline.io.text_parsers import read_step_times

        return read_step_times(hdf5_path)

    @log_time(logger)
    def read_file(
        self,
        hdf5_path: str,
        simu_name: Optional[str] = None,
        N0: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess HDF5 data. Extract multiple DataFrames from a single HDF5 file containing snapshots at multiple times.

        Note: One HDF5 file (.h5part) contains MULTIPLE snapshots (typically 8) at different time points.
        Each snapshot represents the simulation state at one specific TTOT value.

        Raw scalars/singles/binaries/mergers are sourced from the parquet
        particle lake when this file has already been scanned into it (see
        ``nbody_pipeline.analysis.particle_lake``), falling back to parsing
        the HDF5 file directly (``nbody_pipeline.io.text_parsers.dataframes_from_hdf5_file``)
        otherwise -- never writing any cache in either case. The derived-column
        computation below is unchanged either way.

        Args:
            hdf5_path: Path to HDF5 file
            simu_name: Used to get initial condition file path and read N0, and to
                       look up this file's lake data (a lake lookup is skipped
                       entirely when simu_name is None).
            N0: Initial particle count (required if simu_name is None)

        Returns:
            df_dict: Dictionary containing 'scalars', 'singles', 'binaries', 'mergers'
                     Each DataFrame contains data for ALL snapshots in this HDF5 file.

        Note:
            Columns of df_dict['scalars']:
            Index(['TTOT', 'NPAIRS', 'RBAR', 'ZMBAR', 'N', 'TSTAR', 'RDENS(1)', 'RDENS(2)',
            'RDENS(3)', 'TTOT/TCR0', 'TSCALE', 'VSTAR', 'RC', 'NC', 'VC', 'RHOM',
            'CMAX', 'RSCALE', 'RSMIN', 'DMIN1', 'RG(1)', 'RG(2)', 'RG(3)', 'VG(1)',
            'VG(2)', 'VG(3)', 'TIDAL(1)', 'TIDAL(2)', 'TIDAL(3)', 'TIDAL(4)', 'GMG',
            'OMEGA', 'DISK', 'A', 'B', 'ZMET', 'ZPARS(1)', 'ZPARS(2)', 'ZPARS(3)',
            'ZPARS(4)', 'ZPARS(5)', 'ZPARS(6)', 'ZPARS(7)', 'ZPARS(8)', 'ZPARS(9)',
            'ZPARS(10)', 'ZPARS(11)', 'ZPARS(12)', 'ZPARS(13)', 'ZPARS(14)',
            'ZPARS(15)', 'ZPARS(16)', 'ZPARS(17)', 'ZPARS(18)', 'ZPARS(19)',
            'ZPARS(20)', 'ETAI', 'ETAR', 'ETAU', 'ECLOSE', 'DTMIN', 'RMIN', 'GMIN',
            'GMAX', 'SMAX', 'NNBOPT', 'EPOCH0', 'N_SINGLE', 'N_BINARY', 'N_MERGER',
            'Time[Myr]', 'CLIGHT'],
            dtype='object')

            Columns of df_dict['singles']:
            Index(['X1', 'X2', 'X3', 'V1', 'V2', 'V3', 'M',
                'POT', 'R*', 'L*', 'Teff*', 'RC*', 'MC*', 'KW',
                'Name', 'Type', 'ASPN', 'TEV', 'TEV0', 'EPOCH', 'TTOT', 'TTOT/TCR0',
                'TTOT/TRH0', 'Time[Myr]', 'X [pc]', 'Y [pc]', 'Z [pc]',
                'Distance_to_cluster_center[pc]', 'mod_velocity[kmps]', 'Stellar Type'],
                dtype='object')

            Columns of df_dict['binaries']:
            Index(['Bin cm X1', 'Bin cm X2', 'Bin cm X3', 'Bin cm V1', 'Bin cm V2',
                'Bin cm V3', 'Bin M1*', 'Bin M2*', 'Bin rel X1',
                'Bin rel X2', 'Bin rel X3', 'Bin rel V1', 'Bin rel V2', 'Bin rel V3',
                'Bin POT', 'Bin RS1*',
                'Bin L1*', 'Bin Teff1*', 'Bin RS2*', 'Bin L2*', 'Bin Teff2*',
                'Bin RC1*', 'Bin MC1*', 'Bin RC2*', 'Bin MC2*', 'Bin A[au]', 'Bin ECC',
                'Bin P[d]', 'Bin G', 'Bin KW1', 'Bin KW2', 'Bin cm KW', 'Bin Name1',
                'Bin Name2', 'Bin cm Name', 'ASPN1', 'ASPN2', 'TEV1', 'TEV2', 'TEV01',
                'TEV02', 'EPOCH1', 'EPOCH2', 'Bin Label', 'TTOT', 'TTOT/TCR0',
                'TTOT/TRH0', 'Time[Myr]', 'Bin cm X [pc]', 'Bin cm Y [pc]',
                'Bin cm Z [pc]', 'primary_mass[solar]', 'secondary_mass[solar]',
                'total_mass[solar]', 'Distance_to_cluster_center[pc]', 'mass_ratio',
                'primary_stellar_type', 'secondary_stellar_type', 'Stellar Type',
                'peri[au]', 'sum_of_radius[solar]', 'sum_of_radius[au]',
                'Ebind_abs_NBODY', 'eclose_nb', 'mean_core_interparticle_distance[au]',
                'Ebind/kT', 'binary_class', 'is_hard_binary', 'tau_gw[Myr]',
                'peri_over_radius'],
                dtype='object')

            Note (2026-07 soft/hard/temporary reclassification, see
            ``examples/soft-hard-temp/meeting.md``): ``Ebind/kT`` now divides by
            ``TEMPORARY_EBIND_FACTOR * ECLOSE`` (the true per-snapshot ``ECLOSE``,
            not the old fixed ``config.ECLOSE_INPUT``); ``binary_class`` is a
            ``pd.Categorical`` in {"hard", "soft", "temporary"} (see
            ``nbody_pipeline.analysis.physics.classify_binaries``); and
            ``is_hard_binary`` is redefined as ``binary_class == "hard"`` (same
            column name, new semantics -- see CHANGELOG).

            Note (2026-07 L1 feather-cache retirement): force-derivative/integrator
            columns (``A1-3``, ``AD1-3``, ``D21-33``, ``STEP``, ``STEPR``, ``T0``,
            ``T0R``, ``TEV``, ``TEV0``, and their ``Bin``/``Bin cm``/``Bin rel``
            equivalents) are no longer present -- confirmed zero consumers
            repo-wide; see ``_raw_tables_from_lake``.
        """
        from nbody_pipeline.io.text_parsers import (
            get_valueStr_of_namelist_key,
            get_scale_dict_from_hdf5_df,
            dataframes_from_hdf5_file,
        )
        from nbody_pipeline.io.text_parsers import tau_gw

        logger.debug(f"\nProcessing {hdf5_path=}...")

        df_dict = _raw_tables_from_lake(self.config, simu_name, hdf5_path)
        if df_dict is not None:
            logger.info("[hdf5-lake] Sourced raw tables from the particle lake for %s", hdf5_path)
        else:
            df_dict = dataframes_from_hdf5_file(hdf5_path)
        if N0 is None:
            N0 = int(
                get_valueStr_of_namelist_key(
                    path=self.config.input_file_path_of[simu_name], key="N"
                )
            )

        # 预处理标量数据
        scalar_df_all = df_dict["scalars"]
        scale_dict = get_scale_dict_from_hdf5_df(scalar_df=scalar_df_all)
        scalar_df_all["Time[Myr]"] = scalar_df_all["TTOT"] * scale_dict["t"]
        scalar_df_all["CLIGHT"] = 3.0e5 / scalar_df_all["VSTAR"]
        rdens_coord_pc = scalar_df_all[["RDENS(1)", "RDENS(2)", "RDENS(3)"]] * scale_dict["r"]

        # 预处理单星数据
        single_df_all = df_dict["singles"]
        single_df_all["TTOT/TCR0"] = single_df_all["TTOT"].map(scalar_df_all["TTOT/TCR0"])
        single_df_all["TTOT/TRH0"] = single_df_all["TTOT/TCR0"] / (0.1 * N0 / np.log(0.4 * N0))
        single_df_all["Time[Myr]"] = single_df_all["TTOT"] * scale_dict["t"]
        offsets_x1 = single_df_all["TTOT"].map(rdens_coord_pc["RDENS(1)"])
        offsets_x2 = single_df_all["TTOT"].map(rdens_coord_pc["RDENS(2)"])
        offsets_x3 = single_df_all["TTOT"].map(rdens_coord_pc["RDENS(3)"])
        single_df_all["X [pc]"] = single_df_all["X1"] - offsets_x1
        single_df_all["Y [pc]"] = single_df_all["X2"] - offsets_x2
        single_df_all["Z [pc]"] = single_df_all["X3"] - offsets_x3
        single_df_all["Distance_to_cluster_center[pc]"] = np.sqrt(
            single_df_all["X [pc]"] ** 2
            + single_df_all["Y [pc]"] ** 2
            + single_df_all["Z [pc]"] ** 2
        )
        single_df_all["mod_velocity[kmps]"] = np.sqrt(
            single_df_all["V1"] ** 2 + single_df_all["V2"] ** 2 + single_df_all["V3"] ** 2
        )
        ## 虽然星团质心已知有偏移，但不知质心速度如何，这里验算一下质心速度。如果质心速度的mod超过0.1km/s，就警告
        vcm_x = (single_df_all["V1"] * single_df_all["M"]).sum() / single_df_all["M"].sum()
        vcm_y = (single_df_all["V2"] * single_df_all["M"]).sum() / single_df_all["M"].sum()
        vcm_z = (single_df_all["V3"] * single_df_all["M"]).sum() / single_df_all["M"].sum()
        vcm_mod = np.sqrt(vcm_x**2 + vcm_y**2 + vcm_z**2)
        TOO_HIGH_CLUSTER_CM_VELOCITY_THRESHOLD_KMPS = 1.0
        if vcm_mod > TOO_HIGH_CLUSTER_CM_VELOCITY_THRESHOLD_KMPS:
            logger.warning(
                f"[{hdf5_path}] Warning: Cluster center of mass velocity = ({vcm_x:.3f}, {vcm_y:.3f}, {vcm_z:.3f}) km/s, mod={vcm_mod:.3f} km/s, seems high"
            )
        single_df_all["Stellar Type"] = single_df_all["KW"].map(
            self.config.kw_to_stellar_type_verbose
        )
        # NS和BH的光度、温度都是artificial。模拟器设置的值离主序太远，修改以方便展示。
        ## 光度统一设为画图的光度下限
        single_df_all.loc[single_df_all["Stellar Type"].isin(["13:NS", "14:BH"]), "L*"] = (
            self.config.limits["L*"][0] * 1.2
        )
        ## 温度小于画图温度下限的，设为下限；超过上限的，设为上限
        single_df_all.loc[single_df_all["Stellar Type"].isin(["13:NS", "14:BH"]), "Teff*"] = (
            np.clip(
                single_df_all.loc[single_df_all["Stellar Type"].isin(["13:NS", "14:BH"]), "Teff*"],
                self.config.limits["Teff*"][0],
                self.config.limits["Teff*"][1],
            )
        )

        # 预处理双星数据
        binary_df_all = df_dict["binaries"]
        binary_df_all["TTOT/TCR0"] = binary_df_all["TTOT"].map(scalar_df_all["TTOT/TCR0"])
        binary_df_all["TTOT/TRH0"] = binary_df_all["TTOT/TCR0"] / (0.1 * N0 / np.log(0.4 * N0))
        binary_df_all["Time[Myr]"] = binary_df_all["TTOT"] * scale_dict["t"]
        offsets_x1b = binary_df_all["TTOT"].map(rdens_coord_pc["RDENS(1)"])
        offsets_x2b = binary_df_all["TTOT"].map(rdens_coord_pc["RDENS(2)"])
        offsets_x3b = binary_df_all["TTOT"].map(rdens_coord_pc["RDENS(3)"])
        binary_df_all["Bin cm X [pc]"] = binary_df_all["Bin cm X1"] - offsets_x1b
        binary_df_all["Bin cm Y [pc]"] = binary_df_all["Bin cm X2"] - offsets_x2b
        binary_df_all["Bin cm Z [pc]"] = binary_df_all["Bin cm X3"] - offsets_x3b
        binary_df_all["primary_mass[solar]"] = np.max(binary_df_all[["Bin M1*", "Bin M2*"]], axis=1)
        binary_df_all["secondary_mass[solar]"] = np.min(
            binary_df_all[["Bin M1*", "Bin M2*"]], axis=1
        )
        binary_df_all["total_mass[solar]"] = binary_df_all["Bin M1*"] + binary_df_all["Bin M2*"]
        binary_df_all["Distance_to_cluster_center[pc]"] = np.sqrt(
            binary_df_all["Bin cm X [pc]"] ** 2
            + binary_df_all["Bin cm Y [pc]"] ** 2
            + binary_df_all["Bin cm Z [pc]"] ** 2
        )
        binary_df_all["mass_ratio"] = (
            binary_df_all["secondary_mass[solar]"] / binary_df_all["primary_mass[solar]"]
        )
        binary_df_all["primary_stellar_type"] = np.maximum(
            binary_df_all["Bin KW1"], binary_df_all["Bin KW2"]
        )
        binary_df_all["secondary_stellar_type"] = np.minimum(
            binary_df_all["Bin KW1"], binary_df_all["Bin KW2"]
        )
        binary_df_all["Stellar Type"] = (
            binary_df_all["primary_stellar_type"].map(self.config.kw_to_stellar_type)
            + "-"
            + binary_df_all["secondary_stellar_type"].map(self.config.kw_to_stellar_type)
        )
        # Ebind_abs = G * M1 * M2 / 2a / (M1 + M2)
        binary_df_all["peri[au]"] = binary_df_all["Bin A[au]"] * (1 - binary_df_all["Bin ECC"])
        binary_df_all["sum_of_radius[solar]"] = (
            binary_df_all["Bin RS1*"] + binary_df_all["Bin RS2*"]
        )
        binary_df_all["sum_of_radius[au]"] = binary_df_all["sum_of_radius[solar]"] * u.solRad.to(
            u.au
        )

        pc_to_AU = constants.pc.to(u.AU).value
        binary_df_all["Ebind_abs_NBODY"] = (
            binary_df_all["Bin M1*"]
            / scale_dict["m"]
            * binary_df_all["Bin M2*"]
            / scale_dict["m"]
            / (2 * binary_df_all["Bin A[au]"] / pc_to_AU / scale_dict["r"])
            / (
                binary_df_all["Bin M1*"] / scale_dict["m"]
                + binary_df_all["Bin M2*"] / scale_dict["m"]
            )
        )
        # Ebind/kT 分母改为每快照真实 ECLOSE（乘 TEMPORARY_EBIND_FACTOR），并按
        # bin_label/a/Ebind/ECLOSE/核区平均星间距做 hard/soft/temporary 三分类
        # (2026-07 会议 examples/soft-hard-temp/meeting.md 落地)。physics 延迟导入，
        # 避免 io<->analysis 循环导入（同上面 text_parsers 的做法，见 :227-232）。
        from nbody_pipeline.analysis.physics import (
            TEMPORARY_EBIND_FACTOR,
            BINARY_CLASS_ORDER,
            classify_binaries,
            mean_core_interparticle_distance_au,
        )

        eclose_nb = _map_scalar_to_rows(binary_df_all["TTOT"], scalar_df_all, "ECLOSE")
        nc = _map_scalar_to_rows(binary_df_all["TTOT"], scalar_df_all, "NC")
        rc_nb = _map_scalar_to_rows(binary_df_all["TTOT"], scalar_df_all, "RC")
        rbar_pc = _map_scalar_to_rows(binary_df_all["TTOT"], scalar_df_all, "RBAR")

        # 暴露每快照真实 ECLOSE 值，供绘图端标注 temporary 判据的具体数值用。
        binary_df_all["eclose_nb"] = eclose_nb.to_numpy(dtype=float)
        binary_df_all["mean_core_interparticle_distance[au]"] = mean_core_interparticle_distance_au(
            nc.to_numpy(dtype=float),
            rc_nb.to_numpy(dtype=float),
            rbar_pc=rbar_pc.to_numpy(dtype=float),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            binary_df_all["Ebind/kT"] = binary_df_all["Ebind_abs_NBODY"] / (
                TEMPORARY_EBIND_FACTOR * eclose_nb
            )
        binary_df_all["Ebind/kT"] = binary_df_all["Ebind/kT"].replace([np.inf, -np.inf], np.nan)

        if "Bin Label" in binary_df_all.columns:
            bin_label = binary_df_all["Bin Label"]
        else:
            bin_label = pd.Series(_BIN_LABEL_UNKNOWN, index=binary_df_all.index)
        binary_df_all["binary_class"] = pd.Categorical(
            classify_binaries(
                bin_label.to_numpy(),
                binary_df_all["Bin A[au]"].to_numpy(dtype=float),
                binary_df_all["Ebind_abs_NBODY"].to_numpy(dtype=float),
                eclose_nb=eclose_nb.to_numpy(dtype=float),
                mean_core_distance_au=binary_df_all[
                    "mean_core_interparticle_distance[au]"
                ].to_numpy(dtype=float),
            ),
            categories=BINARY_CLASS_ORDER,
        )
        # 列名保留（外部消费者较多，改名代价大），语义重定义为 binary_class == "hard"。
        binary_df_all["is_hard_binary"] = binary_df_all["binary_class"] == "hard"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            binary_df_all["tau_gw[Myr]"] = (
                tau_gw(
                    a=binary_df_all["Bin A[au]"].values * u.au,
                    e=binary_df_all["Bin ECC"].values,
                    mu=(
                        (binary_df_all["Bin M1*"] * binary_df_all["Bin M2*"])
                        / (binary_df_all["Bin M1*"] + binary_df_all["Bin M2*"])
                    ).values
                    * u.solMass,
                    M=(binary_df_all["Bin M1*"] + binary_df_all["Bin M2*"]).values * u.solMass,
                    G=constants.G,
                    c=constants.c,
                )
                .to(u.Myr)
                .value
            )
        binary_df_all["tau_gw[Myr]"] = np.minimum(
            self.config.universe_age_myr, binary_df_all["tau_gw[Myr]"]
        )

        # 处理merger数据
        merger_df_all = df_dict.get("mergers")
        if merger_df_all is None:
            merger_df_all = pd.DataFrame()
            df_dict["mergers"] = merger_df_all
        if not merger_df_all.empty and "TTOT" in merger_df_all.columns:
            merger_df_all["TTOT/TCR0"] = merger_df_all["TTOT"].map(scalar_df_all["TTOT/TCR0"])
            merger_df_all["TTOT/TRH0"] = merger_df_all["TTOT/TCR0"] / (0.1 * N0 / np.log(0.4 * N0))
            merger_df_all["Time[Myr]"] = merger_df_all["TTOT"] * scale_dict["t"]

        return df_dict

    def get_hdf5_file_time_from_filename(self, hdf5_path: str) -> float:
        """Extract approximate snapshot time from HDF5 filename

        Note: This extracts the time encoded in the filename, which represents
        the approximate time of snapshots contained in this HDF5 file.
        Each HDF5 file contains multiple snapshots at different times.
        """
        return float(hdf5_path.split("snap.40_")[-1].split(".h5part")[0])

    @log_time(logger)
    def get_snapshot_at_t(
        self, df_dict: Dict[str, pd.DataFrame], ttot: float
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], bool]:
        """
        Get snapshot data at a specific time point

        Note: This extracts data for ONE snapshot (one moment in time) from the
        df_dict that contains multiple snapshots from an HDF5 file.

        Args:
            df_dict: Dictionary containing 'scalars', 'singles', 'binaries', 'mergers'
                     (obtained from read_file, contains MULTIPLE snapshots)
            ttot: Time point to retrieve (one specific snapshot)

        Returns:
            single_df: Single star DataFrame at this time
            binary_df: Binary star DataFrame at this time
            is_valid: True/False, validates if lengths match scalar counts
        """
        single_df = df_dict["singles"][df_dict["singles"]["TTOT"] == ttot].copy()
        binary_df = df_dict["binaries"][df_dict["binaries"]["TTOT"] == ttot].copy()

        # 获取单星和双星的数量
        N_SINGLE = df_dict["scalars"].loc[ttot, "N_SINGLE"]
        N_BINARY = df_dict["scalars"].loc[ttot, "N_BINARY"]

        # 验证数据完整性
        if not isinstance(N_SINGLE, (float, np.float64, np.float32)) or not isinstance(
            N_BINARY, (float, np.float64, np.float32)
        ):
            return None, None, False

        return single_df, binary_df, True

    def get_compact_object_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for binaries containing compact objects"""
        if "KW" in df.columns:
            compact_object_mask = df["KW"].isin(self.config.compact_object_KW)
        elif "Bin KW1" in df.columns and "Bin KW2" in df.columns:
            compact_object_mask = df["Bin KW1"].isin(self.config.compact_object_KW) | df[
                "Bin KW2"
            ].isin(self.config.compact_object_KW)
        else:
            raise ValueError(
                "DataFrame does not contain 'KW' or 'Bin KW1/Bin KW2' columns. Columns: "
                + str(df.columns)
            )
        return compact_object_mask

    def mark_funny_star_single(self, single_df: pd.DataFrame) -> None:
        """Mark 'interesting' single star targets (writes tag_* columns and aggregates to is_funny)"""
        df = single_df
        low, high = self.config.IMBH_mass_range_msun

        mask_imbh = (df["KW"] == 14) & df["M"].between(low, high)
        if "tag_IMBH" not in df.columns:
            df["tag_IMBH"] = False
        df.loc[mask_imbh, "tag_IMBH"] = True

        # NEW: NS
        mask_ns = df["KW"] == 13
        if "tag_NS" not in df.columns:
            df["tag_NS"] = False
        df.loc[mask_ns, "tag_NS"] = True

        # NEW: high_velocity_halo_star: r>20pc & v>50 km/s
        mask_hv_halo = (df["Distance_to_cluster_center[pc]"] > 20) & (df["mod_velocity[kmps]"] > 50)
        if "tag_high_velocity_halo_star" not in df.columns:
            df["tag_high_velocity_halo_star"] = False
        df.loc[mask_hv_halo, "tag_high_velocity_halo_star"] = True

        if "is_funny" not in df.columns:
            df["is_funny"] = False
        df.loc[mask_imbh | mask_ns | mask_hv_halo, "is_funny"] = True

    def mark_funny_star_binary(self, binary_df: pd.DataFrame) -> None:
        """Mark 'interesting' binary star targets (writes tag_* columns and aggregates to is_funny)"""
        df = binary_df
        low, high = self.config.IMBH_mass_range_msun
        gap_low, gap_high = self.config.PISNe_mass_gap

        mask_bh1 = df["Bin KW1"] == 14
        mask_bh2 = df["Bin KW2"] == 14

        mask_imbh = (mask_bh1 & df["Bin M1*"].between(low, high)) | (
            mask_bh2 & df["Bin M2*"].between(low, high)
        )
        if "tag_IMBH" not in df.columns:
            df["tag_IMBH"] = False
        df.loc[mask_imbh, "tag_IMBH"] = True

        mask_bh_bh = mask_bh1 & mask_bh2
        mask_emr = mask_bh_bh & (df["mass_ratio"] < self.config.extreme_mass_ratio_upper)
        if "tag_EMR_BHBH" not in df.columns:
            df["tag_EMR_BHBH"] = False
        df.loc[mask_emr, "tag_EMR_BHBH"] = True

        mask_high_e = mask_bh_bh & (df["Bin ECC"] > self.config.high_BH_ecc_lower)
        if "tag_high_e_BHBH" not in df.columns:
            df["tag_high_e_BHBH"] = False
        df.loc[mask_high_e, "tag_high_e_BHBH"] = True

        mask_pisne = (mask_bh1 & df["Bin M1*"].between(gap_low, gap_high)) | (
            mask_bh2 & df["Bin M2*"].between(gap_low, gap_high)
        )
        if "tag_PISNe_mass_gap_BH" not in df.columns:
            df["tag_PISNe_mass_gap_BH"] = False
        df.loc[mask_pisne, "tag_PISNe_mass_gap_BH"] = True

        # NEW: BH-NS binary
        mask_ns1 = df["Bin KW1"] == 13
        mask_ns2 = df["Bin KW2"] == 13
        mask_bh_ns = (mask_bh1 & mask_ns2) | (mask_bh2 & mask_ns1)
        if "tag_BH_NS" not in df.columns:
            df["tag_BH_NS"] = False
        df.loc[mask_bh_ns, "tag_BH_NS"] = True

        # NEW: any-NS binary
        mask_ns_any = mask_ns1 | mask_ns2
        if "tag_NS_any" not in df.columns:
            df["tag_NS_any"] = False
        df.loc[mask_ns_any, "tag_NS_any"] = True

        # NEW: Bin cm KW odd/even flags (robust to missing column)
        if "Bin cm KW" in df.columns:
            bcmkw = df["Bin cm KW"]
            mask_mass_transferring = (bcmkw > 10) & ((bcmkw.astype(int) % 2) == 1)
            mask_circularized = (bcmkw >= 10) & ((bcmkw.astype(int) % 2) == 0)
        else:
            logger.debug(
                "[mark_funny_star_binary] Column 'Bin cm KW' not found; skip mass_transferring/circularized tags."
            )
            mask_mass_transferring = False
            mask_circularized = False

        if "tag_mass_transferring_binary" not in df.columns:
            df["tag_mass_transferring_binary"] = False
        df.loc[mask_mass_transferring, "tag_mass_transferring_binary"] = True

        if "tag_circularized_binary" not in df.columns:
            df["tag_circularized_binary"] = False
        df.loc[mask_circularized, "tag_circularized_binary"] = True

        combined_mask = (
            mask_imbh
            | mask_emr
            | mask_high_e
            | mask_pisne
            | mask_bh_ns
            | mask_ns_any
            | mask_mass_transferring
            | mask_circularized
        )
        if "is_funny" not in df.columns:
            df["is_funny"] = False
        df.loc[combined_mask, "is_funny"] = True

    def _compute_binding_energy(
        self, m_bin: float, m3: float, r: float, v_rel_x: float, v_rel_y: float, v_rel_z: float
    ) -> float:
        """
        Compute binding energy for a triple system

        Args:
            m_bin: Binary total mass [Msun]
            m3: Third body mass [Msun]
            r: Distance from binary center of mass to third body [pc]
            v_rel_x, v_rel_y, v_rel_z: Relative velocity components [km/s]

        Returns:
            E_bind: Binding energy [Msun * (km/s)^2], negative value indicates bound
        """
        mu = (m_bin * m3) / (m_bin + m3)
        v_rel = np.sqrt(v_rel_x**2 + v_rel_y**2 + v_rel_z**2)
        E_kin = 0.5 * mu * v_rel**2
        G_PCKMSSQ_MSUN = 4.302e-3
        E_pot = -G_PCKMSSQ_MSUN * m_bin * m3 / r
        E_bind = E_kin + E_pot
        return E_bind

    @log_time(logger)
    def get_triples_from_hdf5(self, df_dict: Dict[str, pd.DataFrame], ttot: float) -> pd.DataFrame:
        """
        Identify hierarchical triple systems

        Args:
            df_dict: Dictionary containing 'singles', 'binaries', 'scalars'
            ttot: Current TTOT value

        Returns:
            triples_df: DataFrame containing triple system information
        """
        pc_to_AU = constants.pc.to(u.AU).value

        single_df_at_t, binary_df_at_t, is_valid = self.get_snapshot_at_t(df_dict, ttot)

        if not is_valid or binary_df_at_t.empty or single_df_at_t.empty:
            logger.debug(f"No valid data for triples detection at ttot={ttot}")
            return pd.DataFrame()

        single_positions = single_df_at_t[["X [pc]", "Y [pc]", "Z [pc]"]].copy().values
        single_names = single_df_at_t["Name"].values
        tree = cKDTree(single_positions)

        triples_list = []

        for idx, binary_row in binary_df_at_t.iterrows():
            bin_name1 = binary_row["Bin Name1"]
            bin_name2 = binary_row["Bin Name2"]
            bin_cm_name = binary_row["Bin cm Name"]
            bin_pos = np.array(
                [
                    binary_row["Bin cm X [pc]"],
                    binary_row["Bin cm Y [pc]"],
                    binary_row["Bin cm Z [pc]"],
                ]
            )
            bin_vel = np.array(
                [binary_row["Bin cm V1"], binary_row["Bin cm V2"], binary_row["Bin cm V3"]]
            )
            m_bin = binary_row["total_mass[solar]"]
            a_bin = binary_row["Bin A[au]"]

            r_max_pc = a_bin / pc_to_AU * 1000
            candidate_indices = tree.query_ball_point(bin_pos, r=r_max_pc)

            n_triples_found_for_this_binary = 0
            for cand_idx in candidate_indices:
                third_name = single_names[cand_idx]

                if third_name == bin_name1 or third_name == bin_name2:
                    continue
                n_triples_found_for_this_binary += 1

                third_row = single_df_at_t.iloc[cand_idx]
                third_pos = single_positions[cand_idx]
                third_vel = np.array([third_row["V1"], third_row["V2"], third_row["V3"]])
                m3 = third_row["M"]

                r_vec = third_pos - bin_pos
                r = np.linalg.norm(r_vec)
                v_rel = third_vel - bin_vel

                E_bind = self._compute_binding_energy(m_bin, m3, r, v_rel[0], v_rel[1], v_rel[2])

                if E_bind < 0:
                    triple_info = {
                        "TTOT": ttot,
                        "Bin cm Name": bin_cm_name,
                        "Bin Name1": bin_name1,
                        "Bin Name2": bin_name2,
                        "Third_body_Name": third_name,
                        "cm_distance_bin_to_3rd[pc]": r,
                        "E_bind[Msun*(km/s)^2]": E_bind,
                        "Bin total_mass[solar]": m_bin,
                        "Third_mass[solar]": m3,
                        "Bin A[au]": a_bin,
                        "Bin ECC": binary_row["Bin ECC"],
                        "Bin KW1": binary_row["Bin KW1"],
                        "Bin KW2": binary_row["Bin KW2"],
                        "Third_KW": third_row["KW"],
                    }
                    triples_list.append(triple_info)

            if n_triples_found_for_this_binary > 1:
                logger.warning(
                    f"[Multiples] Found {n_triples_found_for_this_binary} stars with Ebind<0 for binary 1,2,cm={bin_name1},{bin_name2},{bin_cm_name} at ttot={ttot}, indicating possible higher-order multiples."
                )

        if triples_list:
            triples_df = pd.DataFrame(triples_list)
            logger.info(f"Found {len(triples_df)} hierarchical triples at ttot={ttot}")
            return triples_df
        else:
            logger.debug(f"No hierarchical triples found at ttot={ttot}")
            return pd.DataFrame()

    def get_all_hdf5_paths(
        self,
        simu_name: str,
        wait_age_hour: Optional[int | float] = None,
        sample_every_nb_time: Optional[float] | Any = _CONFIG_DEFAULT,
        exclude_bad_dirname: Optional[bool] = None,
    ) -> List[str]:
        """
        Get all HDF5 file paths for a simulation, sorted by time

        Args:
            simu_name: Name of the simulation
            wait_age_hour: Wait time in hours before processing a file (to ensure it's completely written)
            sample_every_nb_time: Sampling interval in N-body time units based on filename time. Disable if None or <=0.
            exclude_bad_dirname: If True, exclude files whose parent dir name has "bad" within

        Returns:
            List of HDF5 file paths sorted by time, excluding files younger than wait_age_hour
        """
        hdf5_config = getattr(self.config, "hdf5", {}) or {}
        file_selection = hdf5_config.get("file_selection", {})
        if wait_age_hour is None:
            wait_age_hour = file_selection.get("wait_age_hour", 24)
        if sample_every_nb_time is _CONFIG_DEFAULT:
            sample_every_nb_time = file_selection.get("sample_every_nb_time", 1.0)
        if exclude_bad_dirname is None:
            exclude_bad_dirname = file_selection.get("exclude_bad_dirname", True)

        hdf5_files = sorted(
            glob(self.config.pathof[simu_name] + "/**/*.h5part", recursive=True),
            key=lambda fn: self.get_hdf5_file_time_from_filename(fn),
        )  # recursive allows symlink
        cutoff = time.time() - wait_age_hour * 3600
        hdf5_files = [fn for fn in hdf5_files if os.path.getmtime(fn) <= cutoff]
        if not hdf5_files:
            return []

        if exclude_bad_dirname:
            # if parent dir (not check parent of parent) of hdf5 file contains "bad", exclude it
            hdf5_files = [
                fn
                for fn in hdf5_files
                if "bad" not in os.path.basename(os.path.dirname(fn)).lower()
            ]

        # Different physical files can share the same filename-derived index -- e.g. a
        # stale archived copy left alongside a later re-generated file with the same
        # "snap.40_N.h5part" name in a different directory (confirmed for 20sb:
        # snap.40_0.h5part exists both in .../archive/ (old, missing "Bin Label") and
        # .../snap.40/ (re-generated later with newer code, has "Bin Label")). Without
        # dedup, sorting by index alone leaves both in the list with a tied sort key, so
        # which one downstream code picks for that TTOT depends on arbitrary glob()
        # traversal order. Keep the larger file per index -- in practice the more
        # complete one (more Step# groups / more fields).
        best_by_index: Dict[float, str] = {}
        for fn in hdf5_files:
            idx = self.get_hdf5_file_time_from_filename(fn)
            current = best_by_index.get(idx)
            if current is None:
                best_by_index[idx] = fn
            elif os.path.getsize(fn) > os.path.getsize(current):
                logger.warning(
                    f"[get_all_hdf5_paths] filename-index {idx} has duplicate files; "
                    f"keeping larger {fn!r} over {current!r}"
                )
                best_by_index[idx] = fn
            elif fn != current:
                logger.warning(
                    f"[get_all_hdf5_paths] filename-index {idx} has duplicate files; "
                    f"keeping larger {current!r} over {fn!r}"
                )
        hdf5_files = sorted(
            best_by_index.values(), key=lambda fn: self.get_hdf5_file_time_from_filename(fn)
        )

        times = np.array(
            [self.get_hdf5_file_time_from_filename(fn) for fn in hdf5_files], dtype=float
        )
        if sample_every_nb_time is None or sample_every_nb_time <= 0:
            return hdf5_files
        ratios = times / sample_every_nb_time
        keep_mask = np.isclose(ratios, np.round(ratios), rtol=0.0, atol=1e-9)
        unique_indices = np.flatnonzero(keep_mask)

        # 检查文件列表中是否漏了：检查文件名时间戳的实际间隔是否严格为 sample_every_nb_time ，否则输出遗漏的时间点
        actual_times = times[unique_indices]
        time_diffs = np.diff(actual_times)
        expected_diff = sample_every_nb_time
        tolerance = expected_diff * 0.1  # 10% 容差
        for i, diff in enumerate(time_diffs):
            if abs(diff - expected_diff) > tolerance:
                n_missing = int(round(diff / expected_diff)) - 1
                for j in range(n_missing):
                    missing_time = actual_times[i] + (j + 1) * expected_diff
                    logger.warning(
                        f"[get_all_hdf5_paths] Missing HDF5 file for time ~{missing_time:.2f} (between {actual_times[i]:.2f} and {actual_times[i + 1]:.2f})"
                    )

        return [hdf5_files[i] for i in unique_indices]
