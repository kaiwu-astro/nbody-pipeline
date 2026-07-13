"""I/O module for reading and processing N-body simulation files"""

from nbody_pipeline.io.base import ContinousFileProcessor
from nbody_pipeline.io.hdf5_reader import HDF5FileProcessor
from nbody_pipeline.io.lagr_reader import LagrFileProcessor
from nbody_pipeline.io.collision_reader import Coll13FileProcessor, Coal24FileProcessor
from nbody_pipeline.io.text_parsers import (
    get_scale_dict,
    get_scale_dict_from_hdf5_df,
    load_snapshot_data,
    dataframes_from_hdf5_file,
    raw_dataframes_from_hdf5_file,
    read_step_times,
    merge_multiple_hdf5_dataframes,
    decode_bytes_columns_inplace,
    tau_gw,
    load_GWTC_catalog,
    get_valueStr_of_namelist_key,
    read_bwdat,
    read_bdat,
    read_coll_13,
    read_coal_24,
    make_l7header,
    read_lagr_7,
    l7df_to_physical_units,
    transform_l7df_to_sns_friendly,
)

__all__ = [
    "ContinousFileProcessor",
    "HDF5FileProcessor",
    "LagrFileProcessor",
    "Coll13FileProcessor",
    "Coal24FileProcessor",
    "get_scale_dict",
    "get_scale_dict_from_hdf5_df",
    "load_snapshot_data",
    "dataframes_from_hdf5_file",
    "raw_dataframes_from_hdf5_file",
    "read_step_times",
    "merge_multiple_hdf5_dataframes",
    "decode_bytes_columns_inplace",
    "tau_gw",
    "load_GWTC_catalog",
    "get_valueStr_of_namelist_key",
    "read_bwdat",
    "read_bdat",
    "read_coll_13",
    "read_coal_24",
    "make_l7header",
    "read_lagr_7",
    "l7df_to_physical_units",
    "transform_l7df_to_sns_friendly",
]
