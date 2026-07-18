"""Cross-snapshot compact binary counting."""

from __future__ import annotations

import logging
from numbers import Number
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from nbody_pipeline.analysis.cache_paths import (
    COMPACT_BINARY_COUNT_FEATURE,
    analysis_cache_dir,
)
from nbody_pipeline.analysis.hdf5_scan import (
    FeatherMetaCacheMixin,
    HDF5ScanJob,
    ScanBackedAnalysisBase,
    default_file_meta,
    file_is_fresh,
)

logger = logging.getLogger(__name__)


DETAIL_COLUMNS = [
    "binary_key",
    "bin_name1",
    "bin_name2",
    "first_ttot",
    "last_ttot",
    "first_time_myr",
    "last_time_myr",
    "first_kw1",
    "first_kw2",
    "last_kw1",
    "last_kw2",
    "first_stellar_type",
    "last_stellar_type",
    "n_snapshots_seen_in_category",
]

CACHE_COLUMNS = [
    "category",
    "bin_name1",
    "bin_name2",
    "TTOT",
    "Time[Myr]",
    "kw1",
    "kw2",
    "stellar_type",
]


class CompactBinaryCounter(ScanBackedAnalysisBase):
    """Count unique compact binary systems seen across HDF5 snapshots."""

    CATEGORIES = ("gw_source", "pulsar", "xray_binary")
    NS_KW = 13
    BH_KW = 14
    WD_KW = frozenset({10, 11, 12})

    def __init__(self, config_manager: Any, hdf5_file_processor: Any | None = None) -> None:
        super().__init__(config_manager, hdf5_file_processor)
        self.compact_object_kw = frozenset(
            int(kw) for kw in getattr(config_manager, "compact_object_KW", [10, 11, 12, 13, 14])
        )

    def summarize_simulation(
        self,
        simu_name: str,
        *,
        update: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Summarize unique compact binary systems seen from available snapshots."""
        job = self.build_scan_job(simu_name, force=force)
        task = job.task
        cache_df = self._load_or_update_scan_job(job, update=update)
        return self._summary_from_cache(cache_df, task.read_meta())

    def build_scan_job(
        self,
        simu_name: str,
        *,
        force: bool = False,
    ) -> HDF5ScanJob:
        """Build a scan job for batched execution by ``HDF5ScanSession``."""
        options = self._scan_options(force=force)
        task = CompactBinaryCountTask(self, simu_name)
        return HDF5ScanJob(simu_name, task, options)

    def _summary_from_cache(self, cache_df: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, Any]:
        records_by_category: Dict[str, Dict[Tuple[Hashable, Hashable], Dict[str, Any]]] = {
            category: {} for category in self.CATEGORIES
        }
        if not cache_df.empty:
            for _, row in cache_df.iterrows():
                category = str(row["category"])
                if category not in records_by_category:
                    continue
                key, bin_name1, bin_name2 = self._binary_key(row["bin_name1"], row["bin_name2"])
                self._update_category_record(
                    records_by_category[category],
                    key,
                    bin_name1,
                    bin_name2,
                    pd.Series(
                        {
                            "TTOT": row["TTOT"],
                            "Time[Myr]": row.get("Time[Myr]", np.nan),
                            "Stellar Type": row.get("stellar_type", None),
                        }
                    ),
                    int(row["kw1"]),
                    int(row["kw2"]),
                )

        details = {
            category: self._records_to_dataframe(records_by_category[category])
            for category in self.CATEGORIES
        }
        summary = {category: len(details[category]) for category in self.CATEGORIES}
        processed_files = meta.get("processed_files", {})
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
        summary.update(
            {
                "scanned_files": len(processed_files),
                "scanned_snapshots": len(ttot_values),
                "max_ttot": max(ttot_values) if ttot_values else None,
                "max_time_myr": max(time_values) if time_values else None,
            }
        )
        return {"summary": summary, "details": details}

    def _binary_snapshot_from_tables(
        self, df_dict: Dict[str, pd.DataFrame], ttot: float
    ) -> pd.DataFrame:
        binaries = df_dict.get("binaries", pd.DataFrame())
        if "TTOT" not in binaries.columns:
            return pd.DataFrame()
        return binaries[binaries["TTOT"] == ttot].copy()

    def _snapshot_times(self, df_dict: Dict[str, pd.DataFrame]) -> Iterable[float]:
        scalars = df_dict.get("scalars", pd.DataFrame())
        if "TTOT" in scalars.columns:
            return sorted(float(t) for t in scalars["TTOT"].dropna().unique())
        return sorted(float(t) for t in scalars.index.dropna().unique())

    def _snapshot_time_myr(
        self, df_dict: Dict[str, pd.DataFrame], binary_df_at_t: pd.DataFrame, ttot: float
    ) -> float:
        if "Time[Myr]" in binary_df_at_t.columns and not binary_df_at_t["Time[Myr]"].empty:
            return float(binary_df_at_t["Time[Myr]"].max())

        scalars = df_dict.get("scalars", pd.DataFrame())
        if "Time[Myr]" not in scalars.columns:
            return np.nan
        try:
            scalar_row = scalars.loc[ttot]
        except KeyError:
            if "TTOT" not in scalars.columns:
                return np.nan
            rows = scalars[scalars["TTOT"] == ttot]
            if rows.empty:
                return np.nan
            scalar_row = rows.iloc[0]
        if isinstance(scalar_row, pd.DataFrame):
            scalar_row = scalar_row.iloc[0]
        return float(scalar_row["Time[Myr]"])

    def _file_max_time_myr(self, df_dict: Dict[str, pd.DataFrame]) -> float:
        scalars = df_dict.get("scalars", pd.DataFrame())
        if "Time[Myr]" not in scalars.columns or scalars.empty:
            return np.nan
        return float(scalars["Time[Myr]"].max())

    def _accumulate_snapshot(
        self,
        records_by_category: Dict[str, Dict[Tuple[Hashable, Hashable], Dict[str, Any]]],
        binary_df_at_t: pd.DataFrame,
    ) -> None:
        required = {"Bin Name1", "Bin Name2", "Bin KW1", "Bin KW2", "TTOT"}
        missing = required.difference(binary_df_at_t.columns)
        if missing:
            raise ValueError(
                "Binary snapshot missing required columns for compact binary counting: "
                + ", ".join(sorted(missing))
            )

        candidate_df = binary_df_at_t.loc[self._candidate_mask(binary_df_at_t)]
        for _, row in candidate_df.iterrows():
            kw1 = int(row["Bin KW1"])
            kw2 = int(row["Bin KW2"])
            categories = self._categories_for(kw1, kw2)
            if not categories:
                continue

            key, bin_name1, bin_name2 = self._binary_key(row["Bin Name1"], row["Bin Name2"])
            for category in categories:
                self._update_category_record(
                    records_by_category[category],
                    key,
                    bin_name1,
                    bin_name2,
                    row,
                    kw1,
                    kw2,
                )

    def _categories_for(self, kw1: int, kw2: int) -> Tuple[str, ...]:
        categories = []
        has_bh = kw1 == self.BH_KW or kw2 == self.BH_KW
        has_ns = kw1 == self.NS_KW or kw2 == self.NS_KW

        if (kw1 == self.BH_KW and kw2 in {self.BH_KW, self.NS_KW}) or (
            kw2 == self.BH_KW and kw1 == self.NS_KW
        ):
            categories.append("gw_source")
        if has_ns and not has_bh:
            categories.append("pulsar")

        compact1 = kw1 in self.compact_object_kw
        compact2 = kw2 in self.compact_object_kw
        if compact1 != compact2:
            categories.append("xray_binary")
        return tuple(categories)

    def _candidate_mask(self, binary_df_at_t: pd.DataFrame) -> pd.Series:
        kw1 = binary_df_at_t["Bin KW1"]
        kw2 = binary_df_at_t["Bin KW2"]
        has_bh = (kw1 == self.BH_KW) | (kw2 == self.BH_KW)
        has_ns = (kw1 == self.NS_KW) | (kw2 == self.NS_KW)
        gw_source = ((kw1 == self.BH_KW) & (kw2.isin([self.BH_KW, self.NS_KW]))) | (
            (kw2 == self.BH_KW) & (kw1 == self.NS_KW)
        )
        pulsar = has_ns & ~has_bh
        compact1 = kw1.isin(self.compact_object_kw)
        compact2 = kw2.isin(self.compact_object_kw)
        xray_binary = compact1 ^ compact2
        return gw_source | pulsar | xray_binary

    def _binary_key(
        self, name1: Hashable, name2: Hashable
    ) -> Tuple[Tuple[Hashable, Hashable], Hashable, Hashable]:
        ordered = sorted((name1, name2), key=self._name_sort_key)
        return (ordered[0], ordered[1]), ordered[0], ordered[1]

    def _name_sort_key(self, value: Hashable) -> Tuple[int, Any]:
        if isinstance(value, Number):
            return (0, float(value))
        return (1, str(value))

    def _update_category_record(
        self,
        records: Dict[Tuple[Hashable, Hashable], Dict[str, Any]],
        key: Tuple[Hashable, Hashable],
        bin_name1: Hashable,
        bin_name2: Hashable,
        row: pd.Series,
        kw1: int,
        kw2: int,
    ) -> None:
        ttot = float(row["TTOT"])
        time_myr = (
            float(row["Time[Myr]"]) if "Time[Myr]" in row and pd.notna(row["Time[Myr]"]) else np.nan
        )
        stellar_type = self._stellar_type(row, kw1, kw2)

        if key not in records:
            records[key] = {
                "binary_key": key,
                "bin_name1": bin_name1,
                "bin_name2": bin_name2,
                "first_ttot": ttot,
                "last_ttot": ttot,
                "first_time_myr": time_myr,
                "last_time_myr": time_myr,
                "first_kw1": kw1,
                "first_kw2": kw2,
                "last_kw1": kw1,
                "last_kw2": kw2,
                "first_stellar_type": stellar_type,
                "last_stellar_type": stellar_type,
                "n_snapshots_seen_in_category": 1,
            }
            return

        record = records[key]
        if ttot < record["first_ttot"]:
            record["first_ttot"] = ttot
            record["first_time_myr"] = time_myr
            record["first_kw1"] = kw1
            record["first_kw2"] = kw2
            record["first_stellar_type"] = stellar_type
        if ttot >= record["last_ttot"]:
            record["last_ttot"] = ttot
            record["last_time_myr"] = time_myr
            record["last_kw1"] = kw1
            record["last_kw2"] = kw2
            record["last_stellar_type"] = stellar_type
        record["n_snapshots_seen_in_category"] += 1

    def _stellar_type(self, row: pd.Series, kw1: int, kw2: int) -> str:
        if "Stellar Type" in row and pd.notna(row["Stellar Type"]):
            return row["Stellar Type"]
        kw_to_type = getattr(self.config, "kw_to_stellar_type", {})
        return f"{kw_to_type.get(kw1, kw1)}-{kw_to_type.get(kw2, kw2)}"

    def _records_to_dataframe(
        self, records: Dict[Tuple[Hashable, Hashable], Dict[str, Any]]
    ) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=DETAIL_COLUMNS)
        df = pd.DataFrame(records.values())
        return df.sort_values(["first_ttot", "bin_name1", "bin_name2"]).reset_index(drop=True)[
            DETAIL_COLUMNS
        ]

    def _nanmax(self, previous: float, value: float) -> float:
        if np.isnan(previous):
            return value
        if np.isnan(value):
            return previous
        return max(previous, value)


class CompactBinaryCountTask(FeatherMetaCacheMixin):
    """Scan task storing compact-binary category hits per snapshot.

    Only ever needs raw ``Bin Name1``/``Bin Name2``/``Bin KW1``/``Bin KW2``/``TTOT``
    (plus ``TSCALE`` to derive ``Time[Myr]`` itself) -- none of ``read_file``'s
    derived columns -- so this reads straight off the source HDF5 file
    (``hdf5_reader_kind = "raw"``) instead of the lake-first/derived-column path.
    """

    schema_version = 1
    hdf5_reader_kind = "raw"
    required_tables: Sequence[str] = ("scalars", "binaries")
    columns_by_table: Mapping[str, Sequence[str] | None] = {
        "scalars": ["TTOT", "TSCALE"],
        "binaries": [
            "Bin Name1",
            "Bin Name2",
            "Bin KW1",
            "Bin KW2",
            "TTOT",
        ],
    }

    def __init__(self, counter: CompactBinaryCounter, simu_name: str) -> None:
        self.counter = counter
        self.config = counter.config
        self.simu_name = simu_name
        self.name = "compact_binary_count"

    @property
    def cache_path(self) -> Path:
        return (
            analysis_cache_dir(self.config, self.simu_name, COMPACT_BINARY_COUNT_FEATURE)
            / "compact_binary_snapshots.feather"
        )

    def read_cache(self) -> pd.DataFrame:
        cache_df = super().read_cache()
        if cache_df.empty:
            return pd.DataFrame(columns=CACHE_COLUMNS)
        return cache_df

    def is_file_fresh(self, hdf5_path: str, meta: Dict[str, Any], cache_df: pd.DataFrame) -> bool:
        return file_is_fresh(hdf5_path, meta)

    def process_file(
        self,
        hdf5_path: str,
        df_dict: Dict[str, pd.DataFrame],
        meta: Dict[str, Any],
        cache_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        file_meta = self._file_meta(hdf5_path, df_dict)
        rows = []
        binaries = df_dict.get("binaries", pd.DataFrame())
        if not binaries.empty:
            missing = {"Bin Name1", "Bin Name2", "Bin KW1", "Bin KW2", "TTOT"}.difference(
                binaries.columns
            )
            if missing:
                raise ValueError(
                    "Binary table missing required columns for compact binary counting: "
                    + ", ".join(sorted(missing))
                )
            candidate_df = binaries.loc[self.counter._candidate_mask(binaries)]
            for _, row in candidate_df.iterrows():
                kw1 = int(row["Bin KW1"])
                kw2 = int(row["Bin KW2"])
                for category in self.counter._categories_for(kw1, kw2):
                    rows.append(
                        {
                            "category": category,
                            "bin_name1": row["Bin Name1"],
                            "bin_name2": row["Bin Name2"],
                            "TTOT": float(row["TTOT"]),
                            "Time[Myr]": self._row_time_myr(row, df_dict),
                            "kw1": kw1,
                            "kw2": kw2,
                            "stellar_type": self.counter._stellar_type(row, kw1, kw2),
                        }
                    )
        return {"rows": pd.DataFrame(rows, columns=CACHE_COLUMNS), "file_meta": file_meta}

    def merge_file_result(
        self, cache_df: pd.DataFrame, hdf5_path: str, result: Dict[str, Any]
    ) -> pd.DataFrame:
        ttot_values = result.get("file_meta", {}).get("ttot", [])
        if "TTOT" in cache_df.columns and ttot_values:
            cache_df = cache_df[~cache_df["TTOT"].astype(float).isin(ttot_values)]
        new_df = result.get("rows", pd.DataFrame(columns=CACHE_COLUMNS))
        if new_df.empty:
            return cache_df.reset_index(drop=True)
        if cache_df.empty:
            return new_df.reset_index(drop=True)
        return pd.concat([cache_df, new_df], ignore_index=True, sort=False)

    def finalize_cache(self, cache_df: pd.DataFrame) -> pd.DataFrame:
        if cache_df.empty:
            return pd.DataFrame(columns=CACHE_COLUMNS)
        sort_columns = [
            col for col in ["TTOT", "category", "bin_name1", "bin_name2"] if col in cache_df
        ]
        if sort_columns:
            cache_df = cache_df.sort_values(sort_columns)
        return cache_df.reset_index(drop=True)

    def _file_meta(self, hdf5_path: str, df_dict: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
        file_meta = default_file_meta(hdf5_path, df_dict)
        scalars = df_dict.get("scalars", pd.DataFrame())
        if "TSCALE" in scalars.columns and "TTOT" in scalars.columns:
            file_meta["time_myr"] = [
                float(ttot) * float(tscale)
                for ttot, tscale in zip(scalars["TTOT"], scalars["TSCALE"])
                if pd.notna(ttot) and pd.notna(tscale)
            ]
        else:
            file_meta["time_myr"] = []
        return file_meta

    def _row_time_myr(self, row: pd.Series, df_dict: Mapping[str, pd.DataFrame]) -> float:
        scalars = df_dict.get("scalars", pd.DataFrame())
        if "TSCALE" not in scalars.columns or "TTOT" not in scalars.columns:
            return np.nan
        ttot = float(row["TTOT"])
        scalar_rows = scalars[scalars["TTOT"].astype(float) == ttot]
        if scalar_rows.empty:
            return np.nan
        return ttot * float(scalar_rows.iloc[0]["TSCALE"])
