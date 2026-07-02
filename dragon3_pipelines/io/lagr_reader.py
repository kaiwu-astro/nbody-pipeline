"""Lagr file reading and processing"""

import logging

import numpy as np
import pandas as pd

from dragon3_pipelines.io.base import ContinousFileProcessor

logger = logging.getLogger(__name__)


class LagrFileProcessor(ContinousFileProcessor):
    """Read and preprocess lagr.7 files"""

    def __init__(self, config_manager):
        super().__init__(config_manager, file_basename="lagr.7")

    def read_file(self, simu_name: str) -> pd.DataFrame:
        from dragon3_pipelines.io.text_parsers import read_lagr_7, l7df_to_physical_units

        self.concat_file(simu_name)
        logger.debug(f"Loading gathered {self.file_basename} of {simu_name} at {self.file_path}")
        l7df = read_lagr_7(self.file_path)
        l7df = self.clean_data(l7df)
        l7df = l7df_to_physical_units(l7df, self.get_scale_dict_from_stdout(simu_name))
        return l7df

    def clean_data(self, l7df: pd.DataFrame) -> pd.DataFrame:
        """
        1) Drop rows with non-numeric data (should all be int/float)
        2) Handle duplicate 'Time[NB]': keep last occurrence (avoid incomplete rows from interruptions)
        """
        numeric_df = l7df.apply(pd.to_numeric, errors="coerce")
        non_numeric_mask = (numeric_df.isna() & l7df.notna()).any(axis=1)
        if non_numeric_mask.any():
            if "Time[NB]" in l7df.columns:
                bad_times = np.unique(l7df.loc[non_numeric_mask, "Time[NB]"].values)
                logger.warning(
                    f"[lagr.7] Warning: Found non-numeric entries; dropping {non_numeric_mask.sum()} rows at Time[NB]={bad_times}"
                )
            else:
                logger.warning(
                    f"[lagr.7] Warning: Found non-numeric entries; dropping {non_numeric_mask.sum()} rows (no 'Time[NB]' column)"
                )
        l7df = numeric_df.loc[~non_numeric_mask].copy()

        if "Time[NB]" in l7df.columns:
            duplicated_times = l7df["Time[NB]"].duplicated(keep=False)
            if duplicated_times.any():
                dup_vals = np.unique(l7df.loc[duplicated_times, "Time[NB]"].values)
                logger.warning(
                    f"[lagr.7] Warning: Duplicate 'Time[NB]' detected at {dup_vals}; using the last occurrence"
                )
                l7df = l7df.loc[l7df["Time[NB]"].duplicated(keep="last") | ~duplicated_times]
        else:
            logger.warning("[lagr.7] Warning: 'Time[NB]' column not found when de-duplicating.")
        return l7df

    def load_sns_friendly_data(self, simu_name: str) -> pd.DataFrame:
        from dragon3_pipelines.io.text_parsers import transform_l7df_to_sns_friendly

        l7df_sns = transform_l7df_to_sns_friendly(self.read_file(simu_name))
        metrics_to_transform = ["sigma2", "sigma_r2", "sigma_t2"]
        new_rows = []
        for metric_old in metrics_to_transform:
            df_subset = l7df_sns[l7df_sns["Metric"] == metric_old].copy()
            if not df_subset.empty:
                df_subset["Value"] = np.sqrt(df_subset["Value"])
                metric_new = metric_old[:-1]
                df_subset["Metric"] = metric_new
                new_rows.append(df_subset)
        if new_rows:
            l7df_sns = pd.concat(
                [
                    l7df_sns,
                ]
                + new_rows,
                ignore_index=True,
            )

        return l7df_sns

    @staticmethod
    def build_total_mass_df(l7df_sns: pd.DataFrame) -> pd.DataFrame:
        """Build total mass rows from the 100% Lagrangian shell."""
        percent = "100%"
        avmass = l7df_sns[(l7df_sns["Metric"] == "avmass") & (l7df_sns["%"] == percent)]
        nshell = l7df_sns[(l7df_sns["Metric"] == "nshell") & (l7df_sns["%"] == percent)]

        total_mass_df = pd.merge(
            avmass[["Time[Myr]", "Value"]],
            nshell[["Time[Myr]", "Value"]],
            on="Time[Myr]",
            how="inner",
            suffixes=("_avmass", "_nshell"),
        )
        total_mass_df["total_mass"] = total_mass_df["Value_avmass"] * total_mass_df["Value_nshell"]
        return total_mass_df
