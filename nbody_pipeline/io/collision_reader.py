"""Collision and coalescence file reading and processing"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

from nbody_pipeline.io.base import ContinousFileProcessor

logger = logging.getLogger(__name__)


class _Coll_Coal_FileProcessor(ContinousFileProcessor):
    """Read and preprocess coll.13 and coal.24 files"""

    def read_file(
        self, simu_name: str, read_csv_func: Callable[[str], pd.DataFrame]
    ) -> pd.DataFrame:
        self.concat_file(simu_name)
        logger.debug(f"Loading gathered {self.file_basename} of {simu_name} at {self.file_path}")
        df = read_csv_func(self.file_path)
        df = self.clean_data(df)
        df.insert(0, "Time[Myr]", df["TIME[NB]"] * self.get_scale_dict_from_stdout(simu_name)["t"])
        df["primary_mass[solar]"] = np.max(df[["M(I1)[M*]", "M(I2)[M*]"]], axis=1)
        df["secondary_mass[solar]"] = np.min(df[["M(I1)[M*]", "M(I2)[M*]"]], axis=1)
        df["mass_ratio"] = df["secondary_mass[solar]"] / df["primary_mass[solar]"]
        df["primary_stellar_type"] = np.maximum(df["K*(I1)"], df["K*(I2)"])
        df["secondary_stellar_type"] = np.minimum(df["K*(I1)"], df["K*(I2)"])
        df["Stellar Type"] = (
            df["primary_stellar_type"].map(self.config.kw_to_stellar_type)
            + "-"
            + df["secondary_stellar_type"].map(self.config.kw_to_stellar_type)
        )
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().clean_data(df, timecol="TIME[NB]")

    def merge_coll_coal(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge coll.13 and coal.24 dataframes. Missing columns automatically filled with NaN
        """
        return pd.concat([df1, df2], ignore_index=True)


class Coll13FileProcessor(_Coll_Coal_FileProcessor):
    def __init__(self, config_manager):
        super().__init__(config_manager, file_basename="coll.13")

    def read_file(self, simu_name: str) -> pd.DataFrame:
        from nbody_pipeline.io.text_parsers import read_coll_13

        df = super().read_file(simu_name, read_coll_13)
        df["Merger_type"] = "collision"
        return df


class Coal24FileProcessor(_Coll_Coal_FileProcessor):
    def __init__(self, config_manager):
        super().__init__(config_manager, file_basename="coal.24")

    def read_file(self, simu_name: str) -> pd.DataFrame:
        from nbody_pipeline.io.text_parsers import read_coal_24

        df = super().read_file(simu_name, read_coal_24)
        df["Merger_type"] = "coalescence"
        return df
