"""Intermediate-mass black hole population analysis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from dragon3_pipelines.analysis.cache_paths import (
    INTERMEDIATE_MASS_BLACK_HOLE_FEATURE,
    analysis_cache_dir,
)
from dragon3_pipelines.analysis.hdf5_scan import (
    FeatherMetaCacheMixin,
    HDF5ScanJob,
    HDF5ScanOptions,
    ScanBackedAnalysisBase,
    default_file_meta,
    file_is_fresh,
)

IMBH_MIN_MASS_MSUN = 100.0
IMBH_MAX_MASS_MSUN = 1.0e5
BH_KW = 14

SNAPSHOT_COLUMNS = [
    "simu_name",
    "object_name",
    "TTOT",
    "Time[Myr]",
    "state",
    "component_index",
    "mass_msun",
    "kw",
    "radius_pc",
    "x_pc",
    "y_pc",
    "z_pc",
    "speed_kmps",
    "companion_name",
    "companion_mass_msun",
    "companion_kw",
    "binary_pair_key",
    "binary_cm_name",
    "binary_total_mass_msun",
    "binary_a_au",
    "binary_ecc",
    "tau_gw_myr",
    "source_hdf5_path",
]

MERGER_COLUMNS = [
    "simu_name",
    "TTOT",
    "Time[Myr]",
    "product_name",
    "parent_name_1",
    "parent_name_2",
    "parent_name_3",
    "product_kw",
    "parent_kw_1",
    "parent_kw_2",
    "parent_kw_3",
    "product_mass_msun",
    "parent_mass_1_msun",
    "parent_mass_2_msun",
    "parent_mass_3_msun",
    "is_imbh_product",
    "has_imbh_parent",
    "source_hdf5_path",
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
]

OBJECT_COLUMNS = [
    "simu_name",
    "object_name",
    "first_ttot",
    "last_ttot",
    "first_time_myr",
    "last_time_myr",
    "max_mass_msun",
    "n_snapshots",
    "states_seen",
    "formation_channel",
    "hierarchical_merger_generation",
    "fate_label",
    "fate_product_name",
]


class IntermediateMassBlackHoleAnalyzer(ScanBackedAnalysisBase):
    """Identify and summarize IMBH candidates across HDF5 snapshots."""

    def load_imbh_snapshots(
        self,
        simu_name: str,
        *,
        update: bool = True,
        force: bool = False,
    ) -> pd.DataFrame:
        """Return one row per IMBH candidate per snapshot."""
        job = self.build_scan_job(simu_name, force=force)
        return self._load_or_update_scan_job(job, update=update)

    def load_imbh_merger_events(
        self,
        simu_name: str,
        *,
        update: bool = True,
        force: bool = False,
    ) -> pd.DataFrame:
        """Return normalized merger events linked to IMBH candidates or products."""
        job = self.build_scan_job(simu_name, force=force)
        if update:
            self._run_scan_job(job)
        return job.task.read_merger_cache()

    def summarize_simulation(
        self,
        simu_name: str,
        *,
        update: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        """Summarize IMBH objects, formation channels, and conservative fates."""
        job = self.build_scan_job(simu_name, force=force)
        snapshots = self._load_or_update_scan_job(job, update=update)
        merger_events = job.task.read_merger_cache()
        meta = job.task.read_meta()
        objects = self._summarize_objects(simu_name, snapshots, merger_events, meta)

        summary = self._summary_counts(objects, meta)
        return {
            "summary": summary,
            "objects": objects,
            "snapshots": snapshots,
            "merger_events": merger_events,
        }

    def build_scan_jobs(self, simu_name: str, *, force: bool = False) -> list[HDF5ScanJob]:
        """Build scan jobs for batched execution by ``HDF5ScanSession``."""
        return [self.build_scan_job(simu_name, force=force)]

    def build_scan_job(self, simu_name: str, *, force: bool = False) -> HDF5ScanJob:
        """Build the IMBH population scan job."""
        task = IntermediateMassBlackHoleTask(self.config, simu_name)
        return HDF5ScanJob(simu_name, task, self._scan_options(force=force))

    def _summarize_objects(
        self,
        simu_name: str,
        snapshots: pd.DataFrame,
        merger_events: pd.DataFrame,
        meta: Mapping[str, Any],
    ) -> pd.DataFrame:
        if snapshots.empty:
            return pd.DataFrame(columns=OBJECT_COLUMNS)

        min_ttot, max_ttot = _processed_ttot_bounds(meta)
        generation_by_product = _merger_generations(merger_events)
        events_by_parent = _events_by_parent(merger_events)
        rows = []
        for object_name, group in snapshots.groupby("object_name", sort=True):
            group = group.sort_values(["TTOT", "state"]).reset_index(drop=True)
            first = group.iloc[0]
            last = group.iloc[-1]
            obj_key = _normalize_name(object_name)
            generation = generation_by_product.get(obj_key, 0)
            formation_channel = self._formation_channel(group, obj_key, generation, min_ttot)
            fate_label, fate_product = self._fate(group, obj_key, max_ttot, events_by_parent)
            rows.append(
                {
                    "simu_name": simu_name,
                    "object_name": object_name,
                    "first_ttot": float(first["TTOT"]),
                    "last_ttot": float(last["TTOT"]),
                    "first_time_myr": _float_or_nan(first.get("Time[Myr]")),
                    "last_time_myr": _float_or_nan(last.get("Time[Myr]")),
                    "max_mass_msun": float(group["mass_msun"].max()),
                    "n_snapshots": int(len(group)),
                    "states_seen": ",".join(
                        sorted(str(value) for value in group["state"].unique())
                    ),
                    "formation_channel": formation_channel,
                    "hierarchical_merger_generation": int(generation),
                    "fate_label": fate_label,
                    "fate_product_name": fate_product,
                }
            )
        return pd.DataFrame(rows, columns=OBJECT_COLUMNS)

    def _formation_channel(
        self,
        group: pd.DataFrame,
        object_name: Any,
        generation: int,
        min_ttot: float | None,
    ) -> str:
        first_ttot = float(group["TTOT"].min())
        if min_ttot is not None and np.isclose(first_ttot, min_ttot, rtol=0.0, atol=1e-9):
            return "already_imbh_at_scan_start"
        if generation >= 2:
            return "hierarchical_merger"
        if generation == 1:
            return "first_generation_merger"
        if len(group) > 1 and float(group["mass_msun"].min()) < float(group["mass_msun"].max()):
            return "unlinked_mass_growth"
        return "unknown"

    def _fate(
        self,
        group: pd.DataFrame,
        object_name: Any,
        max_ttot: float | None,
        events_by_parent: Mapping[Any, list[pd.Series]],
    ) -> tuple[str, Any]:
        last_ttot = float(group["TTOT"].max())
        if max_ttot is not None and np.isclose(last_ttot, max_ttot, rtol=0.0, atol=1e-9):
            return "retained_candidate", pd.NA
        later_events = [
            event
            for event in events_by_parent.get(object_name, [])
            if pd.notna(event.get("TTOT")) and float(event["TTOT"]) >= last_ttot
        ]
        if later_events:
            event = sorted(later_events, key=lambda row: float(row["TTOT"]))[0]
            return "merged_into_other", event.get("product_name", pd.NA)
        if max_ttot is not None and last_ttot < max_ttot:
            return "not_seen_at_scan_end", pd.NA
        return "unknown", pd.NA

    def _summary_counts(self, objects: pd.DataFrame, meta: Mapping[str, Any]) -> dict[str, Any]:
        processed_files = meta.get("processed_files", {}) if isinstance(meta, Mapping) else {}
        ttot_values = [
            float(ttot)
            for file_meta in processed_files.values()
            for ttot in file_meta.get("ttot", [])
            if pd.notna(ttot)
        ]
        time_values = [
            float(time_myr)
            for file_meta in processed_files.values()
            for time_myr in file_meta.get("time_myr", [])
            if pd.notna(time_myr)
        ]
        formation_counts = (
            objects["formation_channel"].value_counts().to_dict() if not objects.empty else {}
        )
        fate_counts = objects["fate_label"].value_counts().to_dict() if not objects.empty else {}
        return {
            "n_objects": int(len(objects)),
            "formation_channel_counts": formation_counts,
            "fate_label_counts": fate_counts,
            "scanned_files": len(processed_files),
            "scanned_snapshots": len(ttot_values),
            "max_ttot": max(ttot_values) if ttot_values else None,
            "max_time_myr": max(time_values) if time_values else None,
        }


class IntermediateMassBlackHoleTask(FeatherMetaCacheMixin):
    """Scan HDF5 files for IMBH candidate snapshots and linked merger events."""

    schema_version = 1
    name = "intermediate_mass_black_hole"
    required_tables: Sequence[str] = ("scalars", "singles", "binaries", "mergers")
    columns_by_table: Mapping[str, Sequence[str] | None] = {
        "scalars": ["TTOT", "Time[Myr]"],
        "singles": None,
        "binaries": None,
        "mergers": None,
    }

    def __init__(self, config_manager: Any, simu_name: str) -> None:
        self.config = config_manager
        self.simu_name = simu_name
        self._merger_cache_df = pd.DataFrame(columns=MERGER_COLUMNS)

    @property
    def cache_path(self) -> Path:
        return self._cache_dir() / "imbh_snapshots.feather"

    @property
    def merger_cache_path(self) -> Path:
        return self._cache_dir() / "imbh_merger_events.feather"

    def read_cache(self) -> pd.DataFrame:
        self._merger_cache_df = self.read_merger_cache()
        cache_df = super().read_cache()
        if cache_df.empty:
            return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
        return cache_df

    def read_merger_cache(self) -> pd.DataFrame:
        if not self.merger_cache_path.exists():
            return pd.DataFrame(columns=MERGER_COLUMNS)
        event_df = pd.read_feather(self.merger_cache_path)
        return self.finalize_merger_cache(event_df)

    def is_file_fresh(self, hdf5_path: str, meta: Dict[str, Any], cache_df: pd.DataFrame) -> bool:
        cached_times = set()
        if "TTOT" in cache_df.columns:
            cached_times.update(float(ttot) for ttot in cache_df["TTOT"].dropna().unique())
        if "TTOT" in self._merger_cache_df.columns:
            cached_times.update(
                float(ttot) for ttot in self._merger_cache_df["TTOT"].dropna().unique()
            )
        return file_is_fresh(hdf5_path, meta, cached_times or None)

    def process_file(
        self,
        hdf5_path: str,
        df_dict: Dict[str, pd.DataFrame],
        meta: Dict[str, Any],
        cache_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        file_meta = self._file_meta(hdf5_path, df_dict)
        snapshots = pd.DataFrame(self._snapshot_rows(hdf5_path, df_dict), columns=SNAPSHOT_COLUMNS)
        known_imbh_names = set()
        if "object_name" in cache_df.columns:
            known_imbh_names.update(_normalize_name(name) for name in cache_df["object_name"])
        if "object_name" in snapshots.columns:
            known_imbh_names.update(_normalize_name(name) for name in snapshots["object_name"])
        merger_events = pd.DataFrame(
            self._merger_rows(hdf5_path, df_dict, known_imbh_names), columns=MERGER_COLUMNS
        )
        return {"rows": snapshots, "merger_rows": merger_events, "file_meta": file_meta}

    def merge_file_result(
        self, cache_df: pd.DataFrame, hdf5_path: str, result: Dict[str, Any]
    ) -> pd.DataFrame:
        ttot_values = result.get("file_meta", {}).get("ttot", [])
        if "TTOT" in cache_df.columns and ttot_values:
            cache_df = cache_df[~cache_df["TTOT"].astype(float).isin(ttot_values)]
        new_df = result.get("rows", pd.DataFrame(columns=SNAPSHOT_COLUMNS))
        if not new_df.empty:
            if cache_df.empty:
                cache_df = new_df
            else:
                cache_df = pd.concat([cache_df, new_df], ignore_index=True, sort=False)

        event_df = self._merger_cache_df
        if "TTOT" in event_df.columns and ttot_values:
            event_df = event_df[~event_df["TTOT"].astype(float).isin(ttot_values)]
        new_events = result.get("merger_rows", pd.DataFrame(columns=MERGER_COLUMNS))
        if not new_events.empty:
            if event_df.empty:
                event_df = new_events
            else:
                event_df = pd.concat([event_df, new_events], ignore_index=True, sort=False)
        self._merger_cache_df = self.finalize_merger_cache(event_df)
        return self.finalize_cache(cache_df)

    def finalize_cache(self, cache_df: pd.DataFrame) -> pd.DataFrame:
        if cache_df.empty:
            return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
        cache_df = self._deduplicate_snapshot_rows(cache_df)
        sort_columns = ["TTOT", "object_name", "state", "component_index"]
        return cache_df.sort_values(sort_columns).reset_index(drop=True)[SNAPSHOT_COLUMNS]

    def finalize_merger_cache(self, event_df: pd.DataFrame) -> pd.DataFrame:
        if event_df.empty:
            return pd.DataFrame(columns=MERGER_COLUMNS)
        sort_columns = ["TTOT", "product_name"]
        return event_df.sort_values(sort_columns).reset_index(drop=True)[MERGER_COLUMNS]

    def write_cache_and_meta(
        self,
        cache_df: pd.DataFrame,
        processed_files: Dict[str, Dict[str, Any]],
        options: HDF5ScanOptions,
    ) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        event_df = self.finalize_merger_cache(self._merger_cache_df)
        tmp_event_path = self.merger_cache_path.with_suffix(self.merger_cache_path.suffix + ".tmp")
        event_df.to_feather(tmp_event_path)
        tmp_event_path.replace(self.merger_cache_path)
        super().write_cache_and_meta(cache_df, processed_files, options)

    def _cache_dir(self) -> Path:
        return analysis_cache_dir(self.config, self.simu_name, INTERMEDIATE_MASS_BLACK_HOLE_FEATURE)

    def _snapshot_rows(
        self, hdf5_path: str, df_dict: Mapping[str, pd.DataFrame]
    ) -> list[dict[str, Any]]:
        rows = []
        singles = df_dict.get("singles", pd.DataFrame())
        if not singles.empty:
            self._require_columns(singles, {"KW", "M", "TTOT"}, "single IMBH extraction")
            single_candidates = singles.loc[_imbh_mask(singles["KW"], singles["M"])]
            for _, row in single_candidates.iterrows():
                rows.append(self._single_row(hdf5_path, row))

        binaries = df_dict.get("binaries", pd.DataFrame())
        if not binaries.empty:
            self._require_columns(
                binaries,
                {"Bin KW1", "Bin KW2", "Bin M1*", "Bin M2*", "TTOT"},
                "binary IMBH extraction",
            )
            for _, row in binaries.iterrows():
                if _is_imbh(row["Bin KW1"], row["Bin M1*"]):
                    rows.append(self._binary_row(hdf5_path, row, component_index=1))
                if _is_imbh(row["Bin KW2"], row["Bin M2*"]):
                    rows.append(self._binary_row(hdf5_path, row, component_index=2))
        return rows

    def _single_row(self, hdf5_path: str, row: pd.Series) -> dict[str, Any]:
        return {
            "simu_name": self.simu_name,
            "object_name": _first_existing(row, ["Name", "NAME"]),
            "TTOT": float(row["TTOT"]),
            "Time[Myr]": _float_or_nan(row.get("Time[Myr]")),
            "state": "single",
            "component_index": pd.NA,
            "mass_msun": float(row["M"]),
            "kw": int(row["KW"]),
            "radius_pc": _float_or_nan(row.get("Distance_to_cluster_center[pc]")),
            "x_pc": _float_or_nan(_first_existing(row, ["X [pc]", "X1"])),
            "y_pc": _float_or_nan(_first_existing(row, ["Y [pc]", "X2"])),
            "z_pc": _float_or_nan(_first_existing(row, ["Z [pc]", "X3"])),
            "speed_kmps": _float_or_nan(row.get("mod_velocity[kmps]")),
            "companion_name": pd.NA,
            "companion_mass_msun": np.nan,
            "companion_kw": pd.NA,
            "binary_pair_key": pd.NA,
            "binary_cm_name": pd.NA,
            "binary_total_mass_msun": np.nan,
            "binary_a_au": np.nan,
            "binary_ecc": np.nan,
            "tau_gw_myr": np.nan,
            "source_hdf5_path": hdf5_path,
        }

    def _binary_row(
        self, hdf5_path: str, row: pd.Series, *, component_index: int
    ) -> dict[str, Any]:
        other_index = 2 if component_index == 1 else 1
        name = row.get(f"Bin Name{component_index}", pd.NA)
        companion_name = row.get(f"Bin Name{other_index}", pd.NA)
        pair_key = _pair_key(name, companion_name)
        x_pc = _first_existing(row, ["Bin cm X [pc]", "Bin cm X1"])
        y_pc = _first_existing(row, ["Bin cm Y [pc]", "Bin cm X2"])
        z_pc = _first_existing(row, ["Bin cm Z [pc]", "Bin cm X3"])
        return {
            "simu_name": self.simu_name,
            "object_name": name,
            "TTOT": float(row["TTOT"]),
            "Time[Myr]": _float_or_nan(row.get("Time[Myr]")),
            "state": "binary",
            "component_index": component_index,
            "mass_msun": float(row[f"Bin M{component_index}*"]),
            "kw": int(row[f"Bin KW{component_index}"]),
            "radius_pc": _float_or_nan(row.get("Distance_to_cluster_center[pc]")),
            "x_pc": _float_or_nan(x_pc),
            "y_pc": _float_or_nan(y_pc),
            "z_pc": _float_or_nan(z_pc),
            "speed_kmps": _binary_speed(row),
            "companion_name": companion_name,
            "companion_mass_msun": _float_or_nan(row.get(f"Bin M{other_index}*")),
            "companion_kw": row.get(f"Bin KW{other_index}", pd.NA),
            "binary_pair_key": pair_key,
            "binary_cm_name": _first_existing(row, ["Bin cm Name", "Bin Label"]),
            "binary_total_mass_msun": _float_or_nan(row.get("total_mass[solar]")),
            "binary_a_au": _float_or_nan(row.get("Bin A[au]")),
            "binary_ecc": _float_or_nan(row.get("Bin ECC")),
            "tau_gw_myr": _float_or_nan(row.get("tau_gw[Myr]")),
            "source_hdf5_path": hdf5_path,
        }

    def _merger_rows(
        self,
        hdf5_path: str,
        df_dict: Mapping[str, pd.DataFrame],
        known_imbh_names: set[Any],
    ) -> list[dict[str, Any]]:
        mergers = df_dict.get("mergers", pd.DataFrame())
        if mergers.empty:
            return []
        rows = []
        for _, row in mergers.iterrows():
            event = self._normalized_merger_row(hdf5_path, row)
            product_key = _normalize_name(event["product_name"])
            parent_keys = {
                _normalize_name(event["parent_name_1"]),
                _normalize_name(event["parent_name_2"]),
                _normalize_name(event["parent_name_3"]),
            }
            is_product = _is_imbh(event["product_kw"], event["product_mass_msun"])
            has_parent = any(
                parent in known_imbh_names for parent in parent_keys if not pd.isna(parent)
            ) or any(
                _is_imbh(event[f"parent_kw_{index}"], event[f"parent_mass_{index}_msun"])
                for index in [1, 2, 3]
            )
            event["is_imbh_product"] = bool(is_product)
            event["has_imbh_parent"] = bool(has_parent)
            if (
                event["is_imbh_product"]
                or event["has_imbh_parent"]
                or product_key in known_imbh_names
            ):
                rows.append(event)
        return rows

    def _normalized_merger_row(self, hdf5_path: str, row: pd.Series) -> dict[str, Any]:
        return {
            "simu_name": self.simu_name,
            "TTOT": _float_or_nan(row.get("TTOT")),
            "Time[Myr]": _float_or_nan(row.get("Time[Myr]")),
            "product_name": row.get("Mer NAMC", pd.NA),
            "parent_name_1": row.get("Mer NAM1", pd.NA),
            "parent_name_2": row.get("Mer NAM2", pd.NA),
            "parent_name_3": row.get("Mer NAM3", pd.NA),
            "product_kw": row.get("Mer KWC", pd.NA),
            "parent_kw_1": row.get("Mer KW1", pd.NA),
            "parent_kw_2": row.get("Mer KW2", pd.NA),
            "parent_kw_3": row.get("Mer KW3", pd.NA),
            "product_mass_msun": _merger_product_mass(row),
            "parent_mass_1_msun": _float_or_nan(row.get("Mer M1")),
            "parent_mass_2_msun": _float_or_nan(row.get("Mer M2")),
            "parent_mass_3_msun": _float_or_nan(row.get("Mer M3")),
            "is_imbh_product": False,
            "has_imbh_parent": False,
            "source_hdf5_path": hdf5_path,
            "Mer NAM1": row.get("Mer NAM1", pd.NA),
            "Mer NAM2": row.get("Mer NAM2", pd.NA),
            "Mer NAM3": row.get("Mer NAM3", pd.NA),
            "Mer NAMC": row.get("Mer NAMC", pd.NA),
            "Mer KW1": row.get("Mer KW1", pd.NA),
            "Mer KW2": row.get("Mer KW2", pd.NA),
            "Mer KW3": row.get("Mer KW3", pd.NA),
            "Mer KWC": row.get("Mer KWC", pd.NA),
            "Mer M1": row.get("Mer M1", np.nan),
            "Mer M2": row.get("Mer M2", np.nan),
            "Mer M3": row.get("Mer M3", np.nan),
        }

    def _file_meta(self, hdf5_path: str, df_dict: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
        file_meta = default_file_meta(hdf5_path, df_dict)
        scalars = df_dict.get("scalars", pd.DataFrame())
        if "Time[Myr]" in scalars.columns:
            file_meta["time_myr"] = [
                float(time_myr) for time_myr in scalars["Time[Myr]"].dropna().tolist()
            ]
        else:
            file_meta["time_myr"] = []
        return file_meta

    def _deduplicate_snapshot_rows(self, cache_df: pd.DataFrame) -> pd.DataFrame:
        if cache_df.empty:
            return cache_df
        cache_df = cache_df.copy()
        cache_df["_state_rank"] = cache_df["state"].map({"single": 0, "binary": 1}).fillna(0)
        cache_df = cache_df.sort_values(["object_name", "TTOT", "_state_rank"])
        cache_df = cache_df.drop_duplicates(["object_name", "TTOT"], keep="last")
        return cache_df.drop(columns=["_state_rank"])

    def _require_columns(self, df: pd.DataFrame, columns: set[str], purpose: str) -> None:
        missing = columns.difference(df.columns)
        if missing:
            raise ValueError(
                f"Table missing required columns for {purpose}: {', '.join(sorted(missing))}"
            )


def _imbh_mask(kw: pd.Series, mass: pd.Series) -> pd.Series:
    return (
        (kw.astype(int) == BH_KW)
        & (mass.astype(float) > IMBH_MIN_MASS_MSUN)
        & (mass.astype(float) < IMBH_MAX_MASS_MSUN)
    )


def _is_imbh(kw: Any, mass: Any) -> bool:
    if pd.isna(kw) or pd.isna(mass):
        return False
    return int(kw) == BH_KW and IMBH_MIN_MASS_MSUN < float(mass) < IMBH_MAX_MASS_MSUN


def _float_or_nan(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    return float(value)


def _first_existing(row: pd.Series, columns: Sequence[str]) -> Any:
    for column in columns:
        if column in row and pd.notna(row[column]):
            return row[column]
    return pd.NA


def _pair_key(name1: Any, name2: Any) -> str:
    ordered = sorted([name1, name2], key=lambda value: str(value))
    return f"{ordered[0]}-{ordered[1]}"


def _binary_speed(row: pd.Series) -> float:
    if "Bin cm V1" in row and "Bin cm V2" in row and "Bin cm V3" in row:
        return float(np.sqrt(row["Bin cm V1"] ** 2 + row["Bin cm V2"] ** 2 + row["Bin cm V3"] ** 2))
    return np.nan


def _merger_product_mass(row: pd.Series) -> float:
    values = [_float_or_nan(row.get(column)) for column in ["Mer M1", "Mer M2", "Mer M3"]]
    return float(np.nansum(values)) if not all(np.isnan(value) for value in values) else np.nan


def _normalize_name(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return value


def _merger_generations(merger_events: pd.DataFrame) -> dict[Any, int]:
    if merger_events.empty:
        return {}
    parents_by_product = {}
    for _, event in merger_events.iterrows():
        product = _normalize_name(event.get("product_name"))
        if pd.isna(product):
            continue
        parents_by_product[product] = [
            _normalize_name(event.get("parent_name_1")),
            _normalize_name(event.get("parent_name_2")),
            _normalize_name(event.get("parent_name_3")),
        ]

    memo: dict[Any, int] = {}

    def generation(name: Any) -> int:
        if name in memo:
            return memo[name]
        parents = [parent for parent in parents_by_product.get(name, []) if not pd.isna(parent)]
        if not parents:
            memo[name] = 0
            return 0
        memo[name] = 1 + max(generation(parent) for parent in parents)
        return memo[name]

    return {product: generation(product) for product in parents_by_product}


def _events_by_parent(merger_events: pd.DataFrame) -> dict[Any, list[pd.Series]]:
    events: dict[Any, list[pd.Series]] = defaultdict(list)
    if merger_events.empty:
        return events
    for _, event in merger_events.iterrows():
        for column in ["parent_name_1", "parent_name_2", "parent_name_3"]:
            parent = _normalize_name(event.get(column))
            if not pd.isna(parent):
                events[parent].append(event)
    return events


def _processed_ttot_bounds(meta: Mapping[str, Any]) -> tuple[float | None, float | None]:
    processed_files = meta.get("processed_files", {}) if isinstance(meta, Mapping) else {}
    values = [
        float(ttot)
        for file_meta in processed_files.values()
        for ttot in file_meta.get("ttot", [])
        if pd.notna(ttot)
    ]
    if not values:
        return None, None
    return min(values), max(values)
