"""Base classes for file processors"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from nbody_pipeline.utils import get_output

logger = logging.getLogger(__name__)


class ContinousFileProcessor:
    """Base class for processing continuous file outputs from simulations"""

    def __init__(self, config_manager, file_basename: str):
        self.config = config_manager
        self.file_basename = file_basename
        self.file_path = None
        self.firstjobof: Dict[str, str] = {}
        self.scale_dict_of: Dict[str, Dict[str, float]] = {}

    def concat_file(self, simu_name: str) -> None:
        """Concatenate all files with the given basename into a temporary file"""
        gather_file_cmd = (
            f"cd {self.config.pathof[simu_name]};"
            + f"""tmpf=`mktemp --suffix=.{self.file_basename}`; find -L . -name '{self.file_basename}*' | xargs ls | xargs cat > $tmpf; echo $tmpf"""
        )
        self.file_path = get_output(gather_file_cmd)[0]
        logger.debug(f"Gathered {self.file_basename} of {simu_name} files into {self.file_path}")

    def read_file(self, simu_name: str):
        """Read the concatenated file. Must be implemented by subclass."""
        self.concat_file(simu_name)
        logger.debug(f"Loading gathered self.file_basename at {self.file_path}")
        raise NotImplementedError("Subclass must implement this method")

    def clean_data(self, df: pd.DataFrame, timecol: str = "TIME[NB]") -> pd.DataFrame:
        """
        Remove duplicate/backward entries due to simulation reruns.
        In data like [1.0, 2.1, 3.2, 4.3, 5.7, 3.5, 4.6, 5.9, 4.7, 4.8, 7.1],
        remove 3.5, 4.6, 4.7, 4.8
        """
        is_forwarding = np.array([df[timecol][: i + 1].max() == v for i, v in df[timecol].items()])
        if not is_forwarding.all():
            logger.warning(
                f"[{self.file_basename}] Warning: Found {len(is_forwarding) - is_forwarding.sum()} descending entries in {timecol}, removing"
            )
        return df[is_forwarding].reset_index(drop=True)

    def firstjobhere(self, simu_name: str) -> str:
        """Get first job ID (cached)"""
        if simu_name not in self.firstjobof.keys():
            get_firstj_cmd = (
                f"cd {self.config.pathof[simu_name]};"
                + r"""ls | grep -E '^[0-9]+$' | sort -n | head -n 1"""
            )
            self.firstjobof[simu_name] = get_output(get_firstj_cmd)[-1]
        return self.firstjobof[simu_name]

    def get_scale_dict_from_stdout(self, simu_name: str) -> Dict[str, float]:
        """
        Extract scaling dictionary from stdout (cached)
        """
        from glob import glob
        from nbody_pipeline.io.text_parsers import get_scale_dict

        if simu_name not in self.scale_dict_of:
            all_output_files = glob(self.config.pathof[simu_name] + "/**/N*out", recursive=True)
            sorted_output_files = sorted(all_output_files, key=lambda x: int(x.split(".")[-2]))
            first_output_file_path = sorted_output_files[0]
            self.scale_dict_of[simu_name] = get_scale_dict(first_output_file_path)
            print(f"Got {self.scale_dict_of[simu_name]} from {first_output_file_path}")
        return self.scale_dict_of[simu_name]
