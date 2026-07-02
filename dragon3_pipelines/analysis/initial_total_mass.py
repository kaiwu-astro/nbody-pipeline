"""Load and cache the initial total cluster mass from lagr.7 data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dragon3_pipelines.analysis.cache_paths import INITIAL_TOTAL_MASS_FEATURE
from dragon3_pipelines.analysis.once import SimulationOnceAnalysisBase
from dragon3_pipelines.io import LagrFileProcessor


class InitialTotalMassAnalyzer(SimulationOnceAnalysisBase):
    """Compute the initial total mass from the 100% Lagrangian shell."""

    SCHEMA_VERSION = 1
    VALUE_COLUMN = "initial_total_mass_msun"

    def __init__(self, config_manager: Any) -> None:
        super().__init__(
            config_manager,
            feature=INITIAL_TOTAL_MASS_FEATURE,
            cache_filename="initial_total_mass.feather",
            meta_filename="initial_total_mass.meta.json",
        )
        self.lagr_file_processor = LagrFileProcessor(config_manager)

    def get_initial_total_mass_msun(
        self, simu_name: str, *, update: bool = True, force: bool = False
    ) -> float:
        """Return the cached initial total mass in solar masses."""
        df = self.load_or_compute(
            simu_name,
            update=update,
            force=force,
            meta={
                "schema_version": self.SCHEMA_VERSION,
                "source": "lagr.7 100% shell avmass * nshell at Time[Myr] == 0.0",
            },
            compute=lambda: self._compute_initial_total_mass_df(simu_name),
        )
        if self.VALUE_COLUMN not in df.columns or len(df) != 1:
            raise ValueError(
                f"Initial total mass cache for simulation {simu_name!r} must contain "
                f"one row with column {self.VALUE_COLUMN!r}."
            )
        return float(df[self.VALUE_COLUMN].iloc[0])

    def _compute_initial_total_mass_df(self, simu_name: str) -> pd.DataFrame:
        l7df_sns = self.lagr_file_processor.load_sns_friendly_data(simu_name)
        total_mass_df = LagrFileProcessor.build_total_mass_df(l7df_sns)
        initial_rows = total_mass_df[total_mass_df["Time[Myr]"] == 0.0]
        if len(initial_rows) != 1:
            raise ValueError(
                "Expected exactly one lagr.7 total mass row at Time[Myr] == 0.0 "
                f"for simulation {simu_name!r}; found {len(initial_rows)}."
            )
        return pd.DataFrame({self.VALUE_COLUMN: [float(initial_rows["total_mass"].iloc[0])]})
