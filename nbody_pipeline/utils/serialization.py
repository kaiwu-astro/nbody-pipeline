"""
Serialization utilities for saving and loading data
"""

import gzip
import os
import pickle as pk
from typing import Any, List


def save(datadir: str, vars: List[Any], fname: str = "save.pkl") -> None:
    """
    Save Python objects to a pickle file.

    Args:
        datadir: Directory path or full path to pickle file.
                If ends with .pkl or .pkl.gz, fname is ignored.
        vars: List of objects to save
        fname: Filename for the pickle file (default: 'save.pkl')

    Note:
        Supports .gz compression if path ends with .gz
    """
    if datadir.endswith(".pkl") or datadir.endswith(".pkl.gz"):
        path = datadir
    else:
        path = datadir + "/" + fname

    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            pk.dump(vars, f)
    else:
        with open(path, "wb") as f:
            pk.dump(vars, f)

    print(f"saved {path} uses {os.path.getsize(path) / 1024**2:.2f}MB")


def read(datadir: str, fname: str = "save.pkl") -> List[Any]:
    """
    Load Python objects from a pickle file.

    Args:
        datadir: Directory path or full path to pickle file.
                If ends with .pkl or .pkl.gz, fname is ignored.
        fname: Filename for the pickle file (default: 'save.pkl')

    Returns:
        List of loaded objects

    Note:
        Supports .gz compression if path ends with .gz
    """
    if datadir.endswith(".pkl") or datadir.endswith(".pkl.gz"):
        path = datadir
    else:
        path = datadir + "/" + fname

    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pk.load(f)
    else:
        with open(path, "rb") as f:
            return pk.load(f)
