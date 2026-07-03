"""Text file parsing utilities for dragon3_pipelines"""

import os
import re
from functools import lru_cache
from typing import Dict, Union

import h5py
import numpy as np
import pandas as pd
import astropy.constants as constants
from astropy.units.quantity import Quantity

from dragon3_pipelines.utils import get_output, can_convert_to_float


def get_scale_dict(stdout_path: str) -> Dict[str, float]:
    """Extract physical scaling dictionary from stdout file"""
    scaling_line = get_output(f'grep "PHYSICAL SCALING" {stdout_path}')
    scaling_splitted = scaling_line[0].split(":")[-1].split()
    scaling_dict = {}
    for i in range(0, len(scaling_splitted), 3):
        scaling_dict[scaling_splitted[i]] = float(scaling_splitted[i + 2])
    scaling_dict["r"] = scaling_dict.pop("R*")
    scaling_dict["v"] = scaling_dict.pop("V*")
    scaling_dict["m"] = scaling_dict.pop("M*")
    scaling_dict["t"] = scaling_dict.pop("T*")
    return scaling_dict


def get_scale_dict_from_hdf5_df(scalar_df: pd.DataFrame) -> Dict[str, float]:
    """Extract scaling dictionary from HDF5 scalar DataFrame"""
    return {
        "r": scalar_df["RBAR"].values[0],
        "v": scalar_df["VSTAR"].values[0],
        "m": scalar_df["ZMBAR"].values[0],
        "t": scalar_df["TSCALE"].values[0],
    }


def load_snapshot_data(hdf5_file_path: str, step_key: str) -> Dict:
    """Load data for a specific timestep from HDF5 file and organize by category"""
    with h5py.File(hdf5_file_path, "r") as f:
        step_group = f[step_key]

        scalar_data = {
            k: step_group["000 Scalars"][i]
            for i, k in enumerate(
                [
                    "TTOT",
                    "NPAIRS",
                    "RBAR",
                    "ZMBAR",
                    "N",
                    "TSTAR",
                    "RDENS(1)",
                    "RDENS(2)",
                    "RDENS(3)",
                    "TTOT/TCR0",
                    "TSCALE",
                    "VSTAR",
                    "RC",
                    "NC",
                    "VC",
                    "RHOM",
                    "CMAX",
                    "RSCALE",
                    "RSMIN",
                    "DMIN1",
                    "RG(1)",
                    "RG(2)",
                    "RG(3)",
                    "VG(1)",
                    "VG(2)",
                    "VG(3)",
                    "TIDAL(1)",
                    "TIDAL(2)",
                    "TIDAL(3)",
                    "TIDAL(4)",
                    "GMG",
                    "OMEGA",
                    "DISK",
                    "A",
                    "B",
                    "ZMET",
                    "ZPARS(1)",
                    "ZPARS(2)",
                    "ZPARS(3)",
                    "ZPARS(4)",
                    "ZPARS(5)",
                    "ZPARS(6)",
                    "ZPARS(7)",
                    "ZPARS(8)",
                    "ZPARS(9)",
                    "ZPARS(10)",
                    "ZPARS(11)",
                    "ZPARS(12)",
                    "ZPARS(13)",
                    "ZPARS(14)",
                    "ZPARS(15)",
                    "ZPARS(16)",
                    "ZPARS(17)",
                    "ZPARS(18)",
                    "ZPARS(19)",
                    "ZPARS(20)",
                    "ETAI",
                    "ETAR",
                    "ETAU",
                    "ECLOSE",
                    "DTMIN",
                    "RMIN",
                    "GMIN",
                    "GMAX",
                    "SMAX",
                    "NNBOPT",
                    "EPOCH0",
                    "N_SINGLE",
                    "N_BINARY",
                    "N_MERGER",
                ]
            )
        }
        time_value = scalar_data["TTOT"]

        single_cols = [
            "001 X1",
            "002 X2",
            "003 X3",
            "004 V1",
            "005 V2",
            "006 V3",
            "007 A1",
            "008 A2",
            "009 A3",
            "010 AD1",
            "011 AD2",
            "012 AD3",
            "013 D21",
            "014 D22",
            "015 D23",
            "016 D31",
            "017 D32",
            "018 D33",
            "019 STEP",
            "020 STEPR",
            "021 T0",
            "022 T0R",
            "023 M",
            "024 NB-Sph",
            "025 POT",
            "026 R*",
            "027 L*",
            "028 Teff*",
            "029 RC*",
            "030 MC*",
            "031 KW",
            "032 Name",
            "033 Type",
            "035 ASPN",
            "036 TEV",
            "037 TEV0",
            "038 EPOCH",
        ]
        single_data = {
            col.split(" ", 1)[1]: np.array(step_group[col])
            for col in single_cols
            if col in step_group
        }

        binary_cols = [
            "101 Bin cm X1",
            "102 Bin cm X2",
            "103 Bin cm X3",
            "104 Bin cm V1",
            "105 Bin cm V2",
            "106 Bin cm V3",
            "107 Bin cm A1",
            "108 Bin cm A2",
            "109 Bin cm A3",
            "110 Bin cm AD1",
            "111 Bin cm AD2",
            "112 Bin cm AD3",
            "113 Bin cm D21",
            "114 Bin cm D22",
            "115 Bin cm D23",
            "116 Bin cm D31",
            "117 Bin cm D32",
            "118 Bin cm D33",
            "119 Bin cm STEP",
            "120 Bin cm STEPR",
            "121 Bin cm T0",
            "122 Bin cm T0R",
            "123 Bin M1*",
            "124 Bin M2*",
            "125 Bin rel X1",
            "126 Bin rel X2",
            "127 Bin rel X3",
            "128 Bin rel V1",
            "129 Bin rel V2",
            "130 Bin rel V3",
            "131 Bin rel A1",
            "132 Bin rel A2",
            "133 Bin rel A3",
            "134 Bin rel AD1",
            "135 Bin rel AD2",
            "136 Bin rel AD3",
            "137 Bin rel D21",
            "138 Bin rel D22",
            "139 Bin rel D23",
            "140 Bin rel D31",
            "141 Bin rel D32",
            "142 Bin rel D33",
            "143 Bin POT",
            "144 Bin RS1*",
            "145 Bin L1*",
            "146 Bin Teff1*",
            "147 Bin RS2*",
            "148 Bin L2*",
            "149 Bin Teff2*",
            "150 Bin RC1*",
            "151 Bin MC1*",
            "152 Bin RC2*",
            "153 Bin MC2*",
            "154 Bin A[au]",
            "155 Bin ECC",
            "156 Bin P[d]",
            "157 Bin G",
            "158 Bin KW1",
            "159 Bin KW2",
            "160 Bin cm KW",
            "161 Bin Name1",
            "162 Bin Name2",
            "163 Bin cm Name",
            "164 ASPN1",
            "165 ASPN2",
            "166 TEV1",
            "167 TEV2",
            "168 TEV01",
            "169 TEV02",
            "170 EPOCH1",
            "171 EPOCH2",
            "176 Bin Label",
            "176 Bin cm Name",
        ]
        binary_data = {
            col.split(" ", 1)[1]: np.array(step_group[col])
            for col in binary_cols
            if col in step_group
        }
        if "176 Bin cm Name" in binary_data:
            binary_data["176 Bin Label"] = binary_data.pop("176 Bin cm Name")

        merger_cols = [
            "201 Mer XC1",
            "202 Mer XC2",
            "203 Mer XC3",
            "204 Mer VC1",
            "205 Mer VC2",
            "206 Mer VC3",
            "207 Mer M1",
            "208 Mer M2",
            "209 Mer M3",
            "210 Mer XR01",
            "211 Mer XR02",
            "212 Mer XR03",
            "213 Mer VR01",
            "214 Mer VR02",
            "215 Mer VR03",
            "216 Mer XR11",
            "217 Mer XR12",
            "218 Mer XR13",
            "219 Mer VR11",
            "220 Mer VR12",
            "221 Mer VR13",
            "222 Mer POT",
            "223 Mer RS1",
            "224 Mer L1",
            "225 Mer TE1",
            "226 Mer RS2",
            "227 Mer L2",
            "228 Mer TE2",
            "229 Mer RS3",
            "230 Mer L3",
            "231 Mer TE3",
            "232 Mer RC1",
            "233 Mer MC1",
            "234 Mer RC2",
            "235 Mer MC2",
            "236 Mer RC3",
            "237 Mer MC3",
            "238 Mer A0[au]",
            "239 Mer ECC0",
            "240 Mer P0[d]",
            "241 Mer A1[au]",
            "242 Mer ECC1",
            "243 Mer P1[d]",
            "244 Mer KW1",
            "245 Mer KW2",
            "246 Mer KW3",
            "247 Mer KWC",
            "248 Mer NAM1",
            "249 Mer NAM2",
            "250 Mer NAM3",
            "251 Mer NAMC",
        ]
        merger_data = {
            col.split(" ", 1)[1]: np.array(step_group[col])
            for col in merger_cols
            if col in step_group
        }

        return {
            "ttot": time_value,
            "scalars": scalar_data,
            "singles": single_data,
            "binaries": binary_data,
            "mergers": merger_data,
        }


def dataframes_from_hdf5_file(hdf5_file_path: str) -> Dict[str, pd.DataFrame]:
    """Build three datasets: time series for singles, binaries, and mergers"""
    with h5py.File(hdf5_file_path, "r") as f:
        step_keys = sorted([k for k in f.keys() if k.startswith("Step#")])

    singles_dataframes = []
    binaries_dataframes = []
    mergers_dataframes = []
    scalar_data = []

    presented_ttots = []

    for step_key in step_keys:
        data = load_snapshot_data(hdf5_file_path, step_key)
        if data["ttot"] in presented_ttots:
            continue
        else:
            presented_ttots.append(data["ttot"])

        scalar_data.append({**{"TTOT": data["ttot"]}, **data["scalars"]})

        if data["singles"]:
            df_single = pd.DataFrame(data["singles"])
            df_single["TTOT"] = data["ttot"]
            singles_dataframes.append(df_single)

        if data["binaries"]:
            df_binary = pd.DataFrame(data["binaries"])
            df_binary["TTOT"] = data["ttot"]
            binaries_dataframes.append(df_binary)

        if data["mergers"]:
            df_merger = pd.DataFrame(data["mergers"])
            df_merger["TTOT"] = data["ttot"]
            mergers_dataframes.append(df_merger)

    df_scalar = pd.DataFrame(scalar_data).set_index("TTOT", drop=False)
    df_scalar.attrs["data_source"] = hdf5_file_path

    if singles_dataframes:
        df_singles = pd.concat(singles_dataframes)
        df_singles.attrs["data_source"] = hdf5_file_path
    else:
        df_singles = None

    if binaries_dataframes:
        df_binaries = pd.concat(binaries_dataframes)
        df_binaries.attrs["data_source"] = hdf5_file_path
    else:
        df_binaries = None

    if mergers_dataframes:
        df_mergers = pd.concat(mergers_dataframes)
        df_mergers.attrs["data_source"] = hdf5_file_path
    else:
        df_mergers = None

    return {
        "scalars": df_scalar,
        "singles": df_singles,
        "binaries": df_binaries,
        "mergers": df_mergers,
    }


def merge_multiple_hdf5_dataframes(
    hdf5_pandas_dataframes_dict_list: list,
) -> Dict[str, pd.DataFrame]:
    """
    Merge multiple datasets processed by dataframes_from_hdf5_file

    Args:
        hdf5_pandas_dataframes_dict_list: List containing multiple dataset dictionaries

    Returns:
        Merged dataset dictionary
    """
    merged_datasets = {"scalars": None, "singles": None, "binaries": None, "mergers": None}

    scalar_datasets = [
        ds["scalars"] for ds in hdf5_pandas_dataframes_dict_list if ds["scalars"] is not None
    ]
    if scalar_datasets:
        merged_datasets["scalars"] = pd.concat(scalar_datasets)

    singles_dfs = [
        ds["singles"] for ds in hdf5_pandas_dataframes_dict_list if ds["singles"] is not None
    ]
    if singles_dfs:
        merged_datasets["singles"] = pd.concat(singles_dfs)

    binaries_dfs = [
        ds["binaries"] for ds in hdf5_pandas_dataframes_dict_list if ds["binaries"] is not None
    ]
    if binaries_dfs:
        merged_datasets["binaries"] = pd.concat(binaries_dfs)

    mergers_dfs = [
        ds["mergers"] for ds in hdf5_pandas_dataframes_dict_list if ds["mergers"] is not None
    ]
    if mergers_dfs:
        merged_datasets["mergers"] = pd.concat(mergers_dfs)

    return merged_datasets


def decode_bytes_columns_inplace(df: pd.DataFrame) -> None:
    """
    Decode byte data in DataFrame columns to strings and strip whitespace

    Args:
        df: DataFrame to process (modified in place)
    """
    _col_decoded = []
    for col in df.columns:
        if isinstance(df[col].iloc[0], bytes):
            df[col] = df[col].str.decode("utf-8").str.strip()
            _col_decoded.append(col)
    print("Decoded columns:", _col_decoded)


def tau_gw(
    a: Union[float, Quantity],
    e: Union[float, Quantity],
    mu: Union[float, Quantity],
    M: Union[float, Quantity],
    G=None,
    c=None,
) -> Union[float, Quantity]:
    """
    Calculate gravitational wave merger timescale for binary black hole systems.
    Sobolenko, Berczik, Spurzem 2021 eqn (3)

    Args:
        a: Semi-major axis
        e: Eccentricity (0 <= e < 1)
        mu: Reduced mass
        M: Total mass
        G: Gravitational constant (optional)
        c: Speed of light (optional)

    Returns:
        Merger timescale τ_gw
    """
    if G is None:
        if isinstance(a, float):
            G = constants.G.value
        elif isinstance(a, Quantity):
            G = constants.G
    if c is None:
        if isinstance(a, float):
            c = constants.c.value
        elif isinstance(a, Quantity):
            c = constants.c

    num = 5.0 * c**5 * a**4
    den = 64.0 * G**3 * mu * M**2
    F_e = (1 - e**2) ** 3.5 / (1 + 73.0 * e**2 / 24.0 + 37.0 * e**4 / 96.0)

    return (num / den) * F_e


def load_GWTC_catalog(
    csvpath: str = "/p/project1/madnuc/wu13/intermediate_data/GWTC_catalog.csv",
    reload: bool = False,
) -> pd.DataFrame:
    """Load GWTC gravitational wave catalog"""
    if not reload and os.path.exists(os.path.splitext(csvpath)[0] + ".pkl"):
        return pd.read_pickle(os.path.splitext(csvpath)[0] + ".pkl")

    gwtc_df = pd.read_csv(csvpath)

    gwtc_df["mass_ratio"] = gwtc_df["mass_2_source"] / gwtc_df["mass_1_source"]

    m1_abs_min = gwtc_df["mass_1_source"] + gwtc_df["mass_1_source_lower"]
    m1_abs_max = gwtc_df["mass_1_source"] + gwtc_df["mass_1_source_upper"]
    m2_abs_min = gwtc_df["mass_2_source"] + gwtc_df["mass_2_source_lower"]
    m2_abs_max = gwtc_df["mass_2_source"] + gwtc_df["mass_2_source_upper"]

    mass_ratio_val_min = m2_abs_min / m1_abs_max
    mass_ratio_val_max = m2_abs_max / m1_abs_min

    gwtc_df["mass_ratio_lower"] = mass_ratio_val_min - gwtc_df["mass_ratio"]
    gwtc_df["mass_ratio_upper"] = mass_ratio_val_max - gwtc_df["mass_ratio"]

    gwtc_df.to_pickle(os.path.splitext(csvpath)[0] + ".pkl")

    return gwtc_df


@lru_cache
def get_valueStr_of_namelist_key(path: str, key: str) -> str:
    """
    Extract parameter value from namelist input file (cached)

    Args:
        path: Path to namelist input file
        key: Namelist key to extract

    Returns:
        Value as string
    """
    with open(path, "r") as f:
        content = f.read()

    pattern = rf"\s*{re.escape(key)}\s*=\s*([^,\s\n]+)"
    match = re.search(pattern, content, re.IGNORECASE)

    if match:
        return match.group(1).strip()
    else:
        raise KeyError(f"Key '{key}' not found in {path}")


def read_bwdat(filename: str) -> pd.DataFrame:
    """Read bwdat file"""
    return pd.read_csv(filename, skiprows=(0,), sep=r"\s+")


def read_bdat(filename: str) -> pd.DataFrame:
    """Read bdat file"""
    return pd.read_csv(filename, sep=r"\s+")


COLL_13_COLUMNS = [
    "TIME[NB]",
    "NAME(I1)",
    "NAME(I2)",
    "K*(I1)",
    "K*(I2)",
    "K*(INEW)",
    "M(I1)[M*]",
    "M(I2)[M*]",
    "M(INEW)[M*]",
    "DM[M*]",
    "RS(I1)[R*]",
    "RS(I2)[R*]",
    "RI[RC]",
    "R12[R*]",
    "ECC",
    "P[days]",
]

COAL_24_COLUMNS = [
    "TIME[NB]",
    "NAME(I1)",
    "NAME(I2)",
    "K*(I1)",
    "K*(I2)",
    "K*(INEW)",
    "IQCOLL",
    "M(I1)[M*]",
    "M(I2)[M*]",
    "M(INEW)[M*]",
    "DM[M*]",
    "RS(I1)[R*]",
    "RS(I2)[R*]",
    "RI[RC]",
    "R12[R*]",
    "ECC",
    "P[days]",
    "RCOLL[R*]",
    "EB[NB]",
    "DP[NB]",
    "VINF[km/s]",
]

COAL_24_COMPACT_COLUMNS = [
    "Coalescence_trigger",
    "IQCOLL",
    "TIME[NB]",
    "NAME(I1)",
    "NAME(I2)",
    "K*(I1)",
    "K*(I2)",
    "K*(INEW)",
    "COAL24_FIELD_08",
    "COAL24_FIELD_09",
    "COAL24_FIELD_10",
    "COAL24_FIELD_11",
    "COAL24_FIELD_12",
    "COAL24_FIELD_13",
    "COAL24_FIELD_14",
    "COAL24_FIELD_15",
    "M(I1)[M*]",
    "M(I2)[M*]",
    "M(INEW)[M*]",
    "DM[M*]",
    "COAL24_FIELD_20",
    "COAL24_FIELD_21",
    "COAL24_FIELD_22",
    "COAL24_FIELD_23",
    "ECC",
    "P[days]",
]


def _numeric_output_df(records: list[dict], columns: list[str]) -> pd.DataFrame:
    """Build a DataFrame and convert numeric-looking columns."""
    df = pd.DataFrame.from_records(records, columns=columns)
    for column in df.columns:
        if column == "Coalescence_trigger":
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _split_data_lines(path: str) -> list[list[str]]:
    """Return whitespace tokens for non-empty data lines, leaving headers to callers."""
    token_lines = []
    with open(path, "r") as f:
        for line in f:
            tokens = line.split()
            if tokens:
                token_lines.append(tokens)
    return token_lines


def read_coll_13(path: str) -> pd.DataFrame:
    """Read coll.13 collision file, accepting both headered and headerless chunks."""
    records = []
    for tokens in _split_data_lines(path):
        if len(tokens) != len(COLL_13_COLUMNS) or not can_convert_to_float(tokens[0]):
            continue
        records.append(dict(zip(COLL_13_COLUMNS, tokens)))
    return _numeric_output_df(records, COLL_13_COLUMNS)


def read_coal_24(path: str) -> pd.DataFrame:
    """Read coal.24 coalescence file, accepting old and compact output formats."""
    records = []
    columns = COAL_24_COLUMNS + [
        column for column in COAL_24_COMPACT_COLUMNS if column not in COAL_24_COLUMNS
    ]

    for tokens in _split_data_lines(path):
        if len(tokens) == len(COAL_24_COLUMNS) and can_convert_to_float(tokens[0]):
            records.append(dict(zip(COAL_24_COLUMNS, tokens)))
            continue

        if (
            len(tokens) == len(COAL_24_COMPACT_COLUMNS)
            and tokens[0] in {"BINARY", "ROCHE"}
            and can_convert_to_float(tokens[2])
        ):
            records.append(dict(zip(COAL_24_COMPACT_COLUMNS, tokens)))

    return _numeric_output_df(records, columns)


def make_l7header() -> list:
    """Generate column names for lagr.7 file"""
    baselist = [
        "1.00E-03",
        "3.00E-03",
        "5.00E-03",
        "1.00E-02",
        "3.00E-02",
        "5.00E-02",
        "1.00E-01",
        "2.00E-01",
        "3.00E-01",
        "4.00E-01",
        "5.00E-01",
        "6.00E-01",
        "7.00E-01",
        "8.00E-01",
        "9.00E-01",
        "9.50E-01",
        "9.90E-01",
        "1.00E+00",
        "<RC",
    ]
    baselist2 = [
        "1.00E-03",
        "3.00E-03",
        "5.00E-03",
        "1.00E-02",
        "3.00E-02",
        "5.00E-02",
        "1.00E-01",
        "2.00E-01",
        "3.00E-01",
        "4.00E-01",
        "5.00E-01",
        "6.00E-01",
        "7.00E-01",
        "8.00E-01",
        "9.00E-01",
        "9.50E-01",
        "9.90E-01",
        "1.00E+00",
    ]
    eachnames = [
        "rlagr",
        "rlagr_s",
        "rlagr_b",
        "avmass",
        "nshell",
        "vx",
        "vy",
        "vz",
        "v",
        "vr",
        "vt",
        "sigma2",
        "sigma_r2",
        "sigma_t2",
        "vrot",
    ]
    l7header = (
        ["Time[NB]"]
        + [eachnames[0] + n2 for n2 in baselist]
        + [n1 + n2 for n1 in eachnames[1:3] for n2 in baselist2]
        + [n1 + n2 for n1 in eachnames[3:] for n2 in baselist]
    )
    return l7header


def read_lagr_7(n6resultdir: str = ".", fname: str = "lagr.7.txt") -> pd.DataFrame:
    """Read lagr.7 file"""
    path = n6resultdir + f"/{fname}" if os.path.isdir(n6resultdir) else n6resultdir
    if not os.path.exists(path):
        raise IOError(f"{path} not found")
    l7header = make_l7header()
    l7file_ncol = int(get_output("tail " + path + " | awk '{print NF; exit}'")[0])
    if len(l7header) != l7file_ncol:
        raise IOError(
            f"{path} has ncolumn={l7file_ncol} != {len(l7header)=}. lagr.7 file in source code may have been changed."
        )

    return pd.read_csv(path, sep=r"\s+", names=l7header, skiprows=(0, 1))


def l7df_to_physical_units(df: pd.DataFrame, scale_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Convert the lagr.7 DataFrame to physical units using the provided scaling factors.

    Args:
        df: lagr.7 DataFrame
        scale_dict: Scaling dictionary {'r': rscale, 'm': mscale, 'v': vscale, 't': tscale}

    Returns:
        Converted DataFrame
    """
    converted_df = df.copy()
    converted_df["Time[Myr]"] = converted_df["Time[NB]"] * scale_dict["t"]
    converted_df = converted_df.drop(columns=["Time[NB]"])

    metric_prefix_categories = {
        "rlagr": scale_dict["r"],
        "rlagr_s": scale_dict["r"],
        "rlagr_b": scale_dict["r"],
        "avmass": scale_dict["m"],
        "nshell": 1,
        "vx": scale_dict["v"],
        "vy": scale_dict["v"],
        "vz": scale_dict["v"],
        "v": scale_dict["v"],
        "vr": scale_dict["v"],
        "vt": scale_dict["v"],
        "sigma2": scale_dict["v"] ** 2,
        "sigma_r2": scale_dict["v"] ** 2,
        "sigma_t2": scale_dict["v"] ** 2,
        "vrot": scale_dict["v"],
    }
    for col in converted_df.columns:
        if col.startswith(tuple(metric_prefix_categories.keys())):
            prefix = col.split(".")[0][:-1].split("<")[0]
            if prefix in metric_prefix_categories:
                converted_df[col] = (
                    converted_df[col].astype(float) * metric_prefix_categories[prefix]
                )
            else:
                raise ValueError(f"Unknown prefix '{prefix}' in column '{col}'.")

    return converted_df


def transform_l7df_to_sns_friendly(df_physical_units: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the lagr.7 DataFrame to a long format suitable for seaborn plotting.

    Args:
        df_physical_units: DataFrame already converted to physical units

    Returns:
        Long-format DataFrame with columns: Time[Myr], Percentage, Metric, Value, %
    """
    if "Time[Myr]" not in df_physical_units.columns:
        raise ValueError(
            "Input DataFrame must be converted to physical units and contain 'Time[Myr]' column."
        )
    df = df_physical_units
    melted_df = pd.melt(df, id_vars=["Time[Myr]"], var_name="variable", value_name="Value")

    eachnames = [
        "rlagr_s",
        "rlagr_b",
        "rlagr",
        "avmass",
        "nshell",
        "vx",
        "vy",
        "vz",
        "vrot",
        "vr",
        "vt",
        "sigma_r2",
        "sigma_t2",
        "sigma2",
        "v",
    ]

    def extract_metric_and_percentage(variable: str):
        for name in eachnames:
            if variable.startswith(name):
                return name, variable[len(name) :]
        return None, None

    metrics_and_percentages = melted_df["variable"].apply(extract_metric_and_percentage)
    melted_df["Metric"] = metrics_and_percentages.apply(lambda x: x[0])
    melted_df["Percentage"] = metrics_and_percentages.apply(lambda x: x[1])

    melted_df = melted_df.drop("variable", axis=1)
    melted_df = melted_df[["Time[Myr]", "Percentage", "Metric", "Value"]]

    def percentage_present(x: str) -> str:
        """
        Convert a float to a percentage string with one decimal place.
        If the value is '<RC', return it as is.
        """
        if x == "<RC":
            return x
        elif can_convert_to_float(x):
            if float(x) < 0.01:
                return f"{float(x):.1%}"
            else:
                return f"{float(x):.0%}"
        else:
            raise ValueError(
                f"Cannot convert {x} to float for percentage calculation. Is there additional columns in the original l7df?"
            )

    melted_df["%"] = melted_df["Percentage"].apply(percentage_present)

    return melted_df
