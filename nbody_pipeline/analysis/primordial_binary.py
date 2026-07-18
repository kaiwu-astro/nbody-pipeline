"""Identify and cache primordial binaries from the first HDF5 output."""

from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd

from nbody_pipeline.analysis.cache_paths import PRIMORDIAL_BINARY_FEATURE
from nbody_pipeline.analysis.once import SimulationOnceAnalysisBase
from nbody_pipeline.io import HDF5FileProcessor


class PrimordialBinaryIdentifier(SimulationOnceAnalysisBase):
    """Load primordial binaries from the strict ``TTOT == 0.0`` binary snapshot."""

    SCHEMA_VERSION = 1
    REQUIRED_BINARY_COLUMNS = {"Bin Name1", "Bin Name2", "TTOT"}
    TTOT_RULE = "binaries['TTOT'].astype(float) == 0.0"

    def __init__(self, config_manager: Any) -> None:
        super().__init__(
            config_manager,
            feature=PRIMORDIAL_BINARY_FEATURE,
            cache_filename="primordial_binaries.feather",
            meta_filename="primordial_binaries.meta.json",
        )
        self.hdf5_file_processor = HDF5FileProcessor(config_manager)

    def load_primordial_binaries(
        self,
        simu_name: str,
        *,
        update: bool = True,
        wait_age_hour: int | float | None = None,
        exclude_bad_dirname: bool = True,
        force: bool = False,
    ) -> pd.DataFrame:
        """Return the full cached table of primordial binaries for one simulation."""
        return self.load_or_compute(
            simu_name,
            update=update,
            force=force,
            compute=lambda: self._compute_primordial_binaries(
                simu_name,
                wait_age_hour=wait_age_hour,
                exclude_bad_dirname=exclude_bad_dirname,
            ),
        )

    def _compute_primordial_binaries(
        self,
        simu_name: str,
        *,
        wait_age_hour: int | float | None,
        exclude_bad_dirname: bool,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        hdf5_config = getattr(self.config, "hdf5", {}) or {}
        file_selection = hdf5_config.get("file_selection", {})
        if wait_age_hour is None:
            wait_age_hour = file_selection.get("wait_age_hour", 24)

        first_hdf5_path = self._first_hdf5_path(
            simu_name,
            wait_age_hour=wait_age_hour,
            exclude_bad_dirname=exclude_bad_dirname,
        )
        source_mtime = os.path.getmtime(first_hdf5_path)
        # Only ever needs raw "Bin Name1"/"Bin Name2"/"TTOT" (see
        # _identify_primordial_binaries below); read_raw_tables applies none of
        # read_file's derived columns/NS-BH clipping, and is lake-first (a
        # projected reconstruction from the particle lake) with a source-HDF5
        # fallback when this file isn't in the lake yet -- cheap either way for a
        # single file at simulation start. "binaries": None means "all raw binary
        # columns" under the post-lake-migration definition (the retired L1
        # cache's column set, i.e. no force-derivative/integrator columns) when
        # served from the lake; the HDF5 fallback path returns literally every
        # source column instead. Both supersets satisfy REQUIRED_BINARY_COLUMNS
        # below. This result feeds a once-cache with no freshness check
        # (SimulationOnceAnalysisBase), so whichever column set was computed at
        # cache-write time is what persists.
        df_dict = self.hdf5_file_processor.read_raw_tables(
            first_hdf5_path,
            tables=["scalars", "binaries"],
            columns_by_table={"scalars": ["TTOT"], "binaries": None},
            simu_name=simu_name,
        )
        binaries = df_dict.get("binaries", pd.DataFrame())
        scalars = df_dict.get("scalars", pd.DataFrame())
        discovered_ttot_values = self._ttot_values(scalars)
        primordial = self._identify_primordial_binaries(binaries)
        return primordial, {
            "schema_version": self.SCHEMA_VERSION,
            "source_hdf5_path": first_hdf5_path,
            "source_mtime": source_mtime,
            "discovered_ttot_values": discovered_ttot_values,
            "row_count": int(len(primordial)),
            "ttot_rule": self.TTOT_RULE,
        }

    def _first_hdf5_path(
        self,
        simu_name: str,
        *,
        wait_age_hour: int | float | None,
        exclude_bad_dirname: bool,
    ) -> str:
        kwargs: Dict[str, Any] = {
            "sample_every_nb_time": None,
            "exclude_bad_dirname": exclude_bad_dirname,
        }
        if wait_age_hour is not None:
            kwargs["wait_age_hour"] = wait_age_hour
        hdf5_paths = self.hdf5_file_processor.get_all_hdf5_paths(simu_name, **kwargs)
        if not hdf5_paths:
            raise ValueError(f"No HDF5 files found for simulation {simu_name!r}.")
        return str(hdf5_paths[0])

    def _identify_primordial_binaries(self, binaries: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_BINARY_COLUMNS.difference(binaries.columns)
        if missing:
            raise ValueError(
                "Binary table missing required columns for primordial binary identification: "
                + ", ".join(sorted(missing))
            )

        zero_snapshot = binaries.loc[binaries["TTOT"].astype(float) == 0.0].copy()
        if zero_snapshot.empty:
            raise ValueError("First HDF5 file has no binary snapshot with strict TTOT == 0.0.")

        try:
            name1 = zero_snapshot["Bin Name1"].astype(int)
            name2 = zero_snapshot["Bin Name2"].astype(int)
        except (TypeError, ValueError) as exc:
            raise ValueError("Binary name columns must be convertible to integer IDs.") from exc

        mask = (name1 - name2).abs() == 1
        primordial = zero_snapshot.loc[mask].copy()
        if primordial.empty:
            primordial["primordial_name_min"] = pd.Series(dtype="int64")
            primordial["primordial_name_max"] = pd.Series(dtype="int64")
            primordial["primordial_pair_key"] = pd.Series(dtype="object")
            primordial["is_primordial_binary"] = pd.Series(dtype="bool")
            return primordial.reset_index(drop=True)

        name_min = pd.concat([name1.loc[mask], name2.loc[mask]], axis=1).min(axis=1).astype(int)
        name_max = pd.concat([name1.loc[mask], name2.loc[mask]], axis=1).max(axis=1).astype(int)
        primordial["primordial_name_min"] = name_min.to_numpy()
        primordial["primordial_name_max"] = name_max.to_numpy()
        primordial["primordial_pair_key"] = [
            f"{min_id}-{max_id}" for min_id, max_id in zip(name_min, name_max)
        ]
        primordial["is_primordial_binary"] = True
        return primordial.reset_index(drop=True)

    def _ttot_values(self, scalars: pd.DataFrame) -> list[float]:
        if "TTOT" not in scalars.columns:
            return []
        return sorted(float(ttot) for ttot in pd.unique(scalars["TTOT"].astype(float)))
