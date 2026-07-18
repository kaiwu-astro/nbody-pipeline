#!/usr/bin/env python3
"""One-off cleanup for the retired L1 reader-table feather cache.

``nbody_pipeline/io/hdf5_reader.py::HDF5FileProcessor`` used to cache its raw
scalars/singles/binaries/mergers tables next to each source ``.h5part`` file
as ``{hdf5_path}.{table}.df.feather`` (existence-only invalidated). That cache
has been retired: ``read_file`` now sources its raw tables from the parquet
particle lake when available, falling back to parsing the HDF5 file directly
otherwise, and never writes anything to disk. This script is the one-time
cleanup of the leftover feather files under ``paths.simulations.*`` directory
trees.

Scans recursively for exactly ``{name}.h5part.{table}.df.feather`` (the
precise old ``_get_feather_path_of`` naming rule -- not a broad ``*.feather``
glob, which would also catch unrelated caches such as particle-history or
L2-feature feather files that are NOT part of this cleanup).

Safety: dry-run by default (lists files + total size, deletes nothing).
Pass --execute to actually delete, only after reviewing the dry-run output.

Usage:
    python scripts/purge_legacy_feather_cache.py --config configs/juwels_madnuc.yaml
    python scripts/purge_legacy_feather_cache.py --config configs/juwels_madnuc.yaml --simu 60sb
    python scripts/purge_legacy_feather_cache.py --config configs/juwels_madnuc.yaml --execute
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from rich.logging import RichHandler

from nbody_pipeline.config import ConfigManager

logger = logging.getLogger(__name__)

_TABLES = ("scalars", "singles", "binaries", "mergers")


def find_legacy_feather_files(simu_root: str) -> List[Path]:
    """Recursively find ``*.h5part.{table}.df.feather`` files under one simulation root."""
    root = Path(simu_root)
    if not root.is_dir():
        logger.warning("Simulation root does not exist, skipping: %s", root)
        return []
    found = []
    for table in _TABLES:
        found.extend(root.rglob(f"*.h5part.{table}.df.feather"))
    return sorted(set(found))


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to user config YAML")
    parser.add_argument(
        "--simu", dest="simu_name", help="Limit to one simulation (default: all configured)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the listed files. Without this flag, only a dry-run "
        "listing + total size is printed and nothing is deleted.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    config = ConfigManager(config_path=args.config)

    simu_names = [args.simu_name] if args.simu_name else list(config.pathof.keys())
    unknown = [name for name in simu_names if name not in config.pathof]
    if unknown:
        parser.error(f"Unknown simulation(s): {unknown}")

    all_files: List[Path] = []
    for simu_name in simu_names:
        simu_root = config.pathof[simu_name]
        files = find_legacy_feather_files(simu_root)
        print(f"{simu_name}: {len(files)} legacy feather file(s) under {simu_root}")
        all_files.extend(files)

    if not all_files:
        print("\nNo legacy L1 feather files found. Nothing to do.")
        return 0

    total_bytes = sum(f.stat().st_size for f in all_files)
    print(f"\n{len(all_files)} file(s) totalling {_human_size(total_bytes)}:")
    for f in all_files:
        print(f"  {f}  ({_human_size(f.stat().st_size)})")

    if not args.execute:
        print(
            "\nDry-run only: no files were deleted. Re-run with --execute after "
            "reviewing this list to actually delete them."
        )
        return 0

    print(f"\n--execute passed: deleting {len(all_files)} file(s)...")
    deleted = 0
    for f in all_files:
        try:
            f.unlink()
            deleted += 1
        except OSError as exc:
            logger.warning("Failed to delete %s: %s", f, exc)
    print(f"Deleted {deleted}/{len(all_files)} file(s).")
    return 0 if deleted == len(all_files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
