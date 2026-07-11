"""Intermediate-mass black hole population analysis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from nbody_pipeline.analysis.cache_paths import (
    INTERMEDIATE_MASS_BLACK_HOLE_FEATURE,
    analysis_cache_dir,
)
from nbody_pipeline.analysis.hdf5_scan import (
    FeatherMetaCacheMixin,
    HDF5ScanJob,
    ScanBackedAnalysisBase,
    default_file_meta,
    file_is_fresh,
)
from nbody_pipeline.io.collision_reader import Coal24FileProcessor, Coll13FileProcessor

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
    "tidal_radius_pc",
    "escape_radius_pc",
    "is_escape_candidate",
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
    "discarded_parent_name",
    "survivor_parent_index",
    "equal_mass_survivor_fallback",
    "event_source",
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
    "first_escape_ttot",
    "first_escape_time_myr",
]


class IntermediateMassBlackHoleAnalyzer(ScanBackedAnalysisBase):
    """Identify and summarize IMBH candidates across HDF5 snapshots."""

    def __init__(
        self,
        config_manager: Any,
        hdf5_file_processor: Any | None = None,
        coll13_file_processor: Any | None = None,
        coal24_file_processor: Any | None = None,
    ) -> None:
        super().__init__(config_manager, hdf5_file_processor)
        self.coll13_file_processor = coll13_file_processor or Coll13FileProcessor(config_manager)
        self.coal24_file_processor = coal24_file_processor or Coal24FileProcessor(config_manager)

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
            snapshots = self._run_scan_job(job)
            return self._refresh_true_merger_events(job.task, snapshots)
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
        if update:
            merger_events = self._refresh_true_merger_events(job.task, snapshots)
        else:
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
            first_escape = _first_escape_row(group)
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
                    "first_escape_ttot": (
                        float(first_escape["TTOT"]) if first_escape is not None else np.nan
                    ),
                    "first_escape_time_myr": (
                        _float_or_nan(first_escape.get("Time[Myr]"))
                        if first_escape is not None
                        else np.nan
                    ),
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
        later_events = [
            event
            for event in events_by_parent.get(object_name, [])
            if pd.notna(event.get("TTOT")) and float(event["TTOT"]) >= last_ttot
        ]
        if later_events:
            event = sorted(later_events, key=lambda row: float(row["TTOT"]))[0]
            return "merged_into_other", event.get("product_name", pd.NA)
        if _has_escape_candidate(group):
            return "escaped_candidate", pd.NA
        if max_ttot is not None and np.isclose(last_ttot, max_ttot, rtol=0.0, atol=1e-9):
            return "retained_candidate", pd.NA
        if max_ttot is not None and last_ttot < max_ttot:
            return "not_seen_at_scan_end", pd.NA
        return "unknown", pd.NA

    def _refresh_true_merger_events(
        self, task: "IntermediateMassBlackHoleTask", snapshots: pd.DataFrame
    ) -> pd.DataFrame:
        events = task.true_merger_events_from_continuous_files(
            self.coll13_file_processor,
            self.coal24_file_processor,
            snapshots=snapshots,
        )
        task.write_merger_cache(events)
        return events

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
    """Scan HDF5 files for IMBH candidate snapshots."""

    schema_version = 2
    name = "intermediate_mass_black_hole"
    required_tables: Sequence[str] = ("scalars", "singles", "binaries")
    columns_by_table: Mapping[str, Sequence[str] | None] = {
        "scalars": ["TTOT", "Time[Myr]", "RBAR", "TIDAL(1)"],
        "singles": None,
        "binaries": None,
    }

    def __init__(self, config_manager: Any, simu_name: str) -> None:
        self.config = config_manager
        self.simu_name = simu_name

    @property
    def cache_path(self) -> Path:
        return self._cache_dir() / "imbh_snapshots.feather"

    @property
    def merger_cache_path(self) -> Path:
        return self._cache_dir() / "imbh_true_merger_events.feather"

    def read_cache(self) -> pd.DataFrame:
        cache_df = super().read_cache()
        if cache_df.empty:
            return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
        return cache_df

    def read_merger_cache(self) -> pd.DataFrame:
        if not self.merger_cache_path.exists():
            return pd.DataFrame(columns=MERGER_COLUMNS)
        event_df = pd.read_feather(self.merger_cache_path)
        return self.finalize_merger_cache(event_df)

    def write_merger_cache(self, event_df: pd.DataFrame) -> None:
        self.merger_cache_path.parent.mkdir(parents=True, exist_ok=True)
        event_df = self.finalize_merger_cache(event_df)
        tmp_event_path = self.merger_cache_path.with_suffix(self.merger_cache_path.suffix + ".tmp")
        event_df.to_feather(tmp_event_path)
        tmp_event_path.replace(self.merger_cache_path)

    def is_file_fresh(self, hdf5_path: str, meta: Dict[str, Any], cache_df: pd.DataFrame) -> bool:
        cached_times = set()
        if "TTOT" in cache_df.columns:
            cached_times.update(float(ttot) for ttot in cache_df["TTOT"].dropna().unique())
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
        return {"rows": snapshots, "file_meta": file_meta}

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

    def _cache_dir(self) -> Path:
        return analysis_cache_dir(self.config, self.simu_name, INTERMEDIATE_MASS_BLACK_HOLE_FEATURE)

    def _snapshot_rows(
        self, hdf5_path: str, df_dict: Mapping[str, pd.DataFrame]
    ) -> list[dict[str, Any]]:
        rows = []
        scalar_context_by_ttot = _scalar_context_by_ttot(df_dict.get("scalars", pd.DataFrame()))
        singles = df_dict.get("singles", pd.DataFrame())
        if not singles.empty:
            self._require_columns(singles, {"KW", "M", "TTOT"}, "single IMBH extraction")
            single_candidates = singles.loc[_imbh_mask(singles["KW"], singles["M"])]
            for _, row in single_candidates.iterrows():
                rows.append(
                    self._with_escape_context(
                        self._single_row(hdf5_path, row),
                        scalar_context_by_ttot.get(float(row["TTOT"]), {}),
                    )
                )

        binaries = df_dict.get("binaries", pd.DataFrame())
        if not binaries.empty:
            self._require_columns(
                binaries,
                {"Bin KW1", "Bin KW2", "Bin M1*", "Bin M2*", "TTOT"},
                "binary IMBH extraction",
            )
            for _, row in binaries.iterrows():
                if _is_imbh(row["Bin KW1"], row["Bin M1*"]):
                    rows.append(
                        self._with_escape_context(
                            self._binary_row(hdf5_path, row, component_index=1),
                            scalar_context_by_ttot.get(float(row["TTOT"]), {}),
                        )
                    )
                if _is_imbh(row["Bin KW2"], row["Bin M2*"]):
                    rows.append(
                        self._with_escape_context(
                            self._binary_row(hdf5_path, row, component_index=2),
                            scalar_context_by_ttot.get(float(row["TTOT"]), {}),
                        )
                    )
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
            "tidal_radius_pc": np.nan,
            "escape_radius_pc": np.nan,
            "is_escape_candidate": False,
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
            "tidal_radius_pc": np.nan,
            "escape_radius_pc": np.nan,
            "is_escape_candidate": False,
            "source_hdf5_path": hdf5_path,
        }

    def _with_escape_context(
        self, snapshot_row: dict[str, Any], scalar_context: Mapping[str, Any]
    ) -> dict[str, Any]:
        tidal_radius_pc, escape_radius_pc = _escape_radii_pc(scalar_context)
        snapshot_row["tidal_radius_pc"] = tidal_radius_pc
        snapshot_row["escape_radius_pc"] = escape_radius_pc
        radius_pc = _float_or_nan(snapshot_row.get("radius_pc"))
        snapshot_row["is_escape_candidate"] = bool(
            np.isfinite(radius_pc)
            and np.isfinite(escape_radius_pc)
            and radius_pc > escape_radius_pc
        )
        return snapshot_row

    def true_merger_events_from_continuous_files(
        self,
        coll13_file_processor: Any,
        coal24_file_processor: Any,
        *,
        snapshots: pd.DataFrame,
    ) -> pd.DataFrame:
        event_frames = []
        for processor in [coll13_file_processor, coal24_file_processor]:
            if not _continuous_files_exist(self.config, self.simu_name, processor.file_basename):
                continue
            event_frames.append(processor.read_file(self.simu_name))
        if not event_frames:
            return pd.DataFrame(columns=MERGER_COLUMNS)

        known_imbh_names = set()
        if "object_name" in snapshots.columns:
            known_imbh_names.update(_normalize_name(name) for name in snapshots["object_name"])

        raw_events = pd.concat(event_frames, ignore_index=True, sort=False)
        rows = [
            self._normalized_true_merger_row(row, known_imbh_names)
            for _, row in raw_events.iterrows()
        ]
        rows = [
            row
            for row in rows
            if row["is_imbh_product"]
            or row["has_imbh_parent"]
            or _normalize_name(row["product_name"]) in known_imbh_names
        ]
        return self.finalize_merger_cache(pd.DataFrame(rows, columns=MERGER_COLUMNS))

    def _normalized_true_merger_row(
        self, row: pd.Series, known_imbh_names: set[Any]
    ) -> dict[str, Any]:
        parent_name_1 = _first_existing(row, ["I1", "I(1)", "NAME(I1)", "NAM(I1)", "Name(I1)"])
        parent_name_2 = _first_existing(row, ["I2", "I(2)", "NAME(I2)", "NAM(I2)", "Name(I2)"])
        parent_mass_1 = _float_or_nan(row.get("M(I1)[M*]"))
        parent_mass_2 = _float_or_nan(row.get("M(I2)[M*]"))
        parent_kw_1 = row.get("K*(I1)", pd.NA)
        parent_kw_2 = row.get("K*(I2)", pd.NA)
        survivor_index = _survivor_parent_index(parent_mass_1, parent_mass_2)
        product_name = parent_name_1 if survivor_index == 1 else parent_name_2
        product_kw = parent_kw_1 if survivor_index == 1 else parent_kw_2
        discarded_parent_name = parent_name_2 if survivor_index == 1 else parent_name_1
        product_mass = _true_merger_product_mass(parent_mass_1, parent_mass_2)
        parent_keys = {_normalize_name(parent_name_1), _normalize_name(parent_name_2)}
        has_parent = any(
            parent in known_imbh_names for parent in parent_keys if not pd.isna(parent)
        )
        has_parent = (
            has_parent
            or _is_imbh(parent_kw_1, parent_mass_1)
            or _is_imbh(parent_kw_2, parent_mass_2)
        )
        return {
            "simu_name": self.simu_name,
            "TTOT": _float_or_nan(_first_existing(row, ["TTOT", "TIME[NB]"])),
            "Time[Myr]": _float_or_nan(row.get("Time[Myr]")),
            "product_name": product_name,
            "parent_name_1": parent_name_1,
            "parent_name_2": parent_name_2,
            "parent_name_3": pd.NA,
            "product_kw": product_kw,
            "parent_kw_1": parent_kw_1,
            "parent_kw_2": parent_kw_2,
            "parent_kw_3": pd.NA,
            "product_mass_msun": product_mass,
            "parent_mass_1_msun": parent_mass_1,
            "parent_mass_2_msun": parent_mass_2,
            "parent_mass_3_msun": np.nan,
            "is_imbh_product": _is_imbh(product_kw, product_mass),
            "has_imbh_parent": bool(has_parent),
            "discarded_parent_name": discarded_parent_name,
            "survivor_parent_index": survivor_index,
            "equal_mass_survivor_fallback": bool(
                np.isfinite(parent_mass_1)
                and np.isfinite(parent_mass_2)
                and np.isclose(parent_mass_1, parent_mass_2, rtol=0.0, atol=0.0)
            ),
            "event_source": row.get("Merger_type", pd.NA),
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


def _normalize_name(value: Any) -> Any:
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return value


def _continuous_files_exist(config: Any, simu_name: str, file_basename: str) -> bool:
    simu_path = Path(config.pathof[simu_name])
    return any(simu_path.rglob(f"{file_basename}*"))


def _survivor_parent_index(parent_mass_1: float, parent_mass_2: float) -> int:
    if np.isfinite(parent_mass_1) and np.isfinite(parent_mass_2):
        if parent_mass_2 > parent_mass_1:
            return 2
        return 1
    if np.isfinite(parent_mass_2):
        return 2
    return 1


def _true_merger_product_mass(parent_mass_1: float, parent_mass_2: float) -> float:
    values = [parent_mass_1, parent_mass_2]
    return float(np.nansum(values)) if any(np.isfinite(value) for value in values) else np.nan


def _scalar_context_by_ttot(scalars: pd.DataFrame) -> dict[float, dict[str, Any]]:
    if scalars.empty:
        return {}
    if "TTOT" in scalars.columns:
        scalar_rows = scalars
    else:
        scalar_rows = scalars.reset_index().rename(columns={"index": "TTOT"})
    context: dict[float, dict[str, Any]] = {}
    for _, row in scalar_rows.iterrows():
        if "TTOT" not in row or pd.isna(row["TTOT"]):
            continue
        context[float(row["TTOT"])] = {
            "RBAR": row.get("RBAR", np.nan),
            "TIDAL(1)": row.get("TIDAL(1)", np.nan),
        }
    return context


def _escape_radii_pc(scalar_context: Mapping[str, Any]) -> tuple[float, float]:
    tidal_1 = _float_or_nan(scalar_context.get("TIDAL(1)"))
    rbar = _float_or_nan(scalar_context.get("RBAR"))
    if not np.isfinite(tidal_1) or not np.isfinite(rbar) or tidal_1 <= 0:
        return np.nan, np.nan
    tidal_radius_pc = float((0.5 / tidal_1) ** (1.0 / 3.0) * rbar)
    return tidal_radius_pc, 2.0 * tidal_radius_pc


def _has_escape_candidate(group: pd.DataFrame) -> bool:
    if "is_escape_candidate" not in group.columns:
        return False
    return bool(group["is_escape_candidate"].fillna(False).any())


def _first_escape_row(group: pd.DataFrame) -> pd.Series | None:
    if "is_escape_candidate" not in group.columns:
        return None
    escaped = group.loc[group["is_escape_candidate"].fillna(False)]
    if escaped.empty:
        return None
    return escaped.sort_values("TTOT").iloc[0]


def _merger_generations(merger_events: pd.DataFrame) -> dict[Any, int]:
    if merger_events.empty:
        return {}
    generation_by_name: dict[Any, int] = {}
    generation_by_product: dict[Any, int] = {}
    events = merger_events.sort_values("TTOT") if "TTOT" in merger_events.columns else merger_events
    for _, event in events.iterrows():
        product = _normalize_name(event.get("product_name"))
        if pd.isna(product):
            continue
        parents = [
            _normalize_name(event.get("parent_name_1")),
            _normalize_name(event.get("parent_name_2")),
            _normalize_name(event.get("parent_name_3")),
        ]
        parent_generations = [
            generation_by_name.get(parent, 0) for parent in parents if not pd.isna(parent)
        ]
        product_generation = 1 + max(parent_generations, default=0)
        generation_by_name[product] = max(generation_by_name.get(product, 0), product_generation)
        generation_by_product[product] = generation_by_name[product]
    return generation_by_product


def _events_by_parent(merger_events: pd.DataFrame) -> dict[Any, list[pd.Series]]:
    events: dict[Any, list[pd.Series]] = defaultdict(list)
    if merger_events.empty:
        return events
    for _, event in merger_events.iterrows():
        parent = _normalize_name(event.get("discarded_parent_name"))
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
