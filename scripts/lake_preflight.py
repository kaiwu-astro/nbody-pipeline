#!/usr/bin/env python3
"""Read-only pre-flight check for the full particle-lake rebuild.

Enumerates every HDF5 file that ``sample_every_nb_time: null`` (i.e. the full,
unsampled particle-lake job) would select for each configured simulation, and
reports anything that would make a full rebuild unsafe or wasteful:

- Duplicate filename-times within one simulation (same nominal snapshot time
  produced by more than one run directory).
- Run-directory time ranges that overlap each other.
- Gaps in the filename-time sequence larger than the typical spacing.
- The real overlapping TTOT sets for flagged candidates, read directly from
  each HDF5 file's ``Step#`` group attrs (cheap: no dataset reads).

It never writes anything under the simulation directories; it only reads.

Usage:
    python scripts/lake_preflight.py --config configs/juwels_madnuc.yaml
    python scripts/lake_preflight.py --config configs/juwels_madnuc.yaml --simu 60sb
    python scripts/lake_preflight.py --config configs/juwels_madnuc.yaml --json report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from rich.logging import RichHandler
from rich.progress import Progress

from nbody_pipeline.config import ConfigManager
from nbody_pipeline.io import HDF5FileProcessor

logger = logging.getLogger(__name__)


@dataclass
class StepInfo:
    ttot: float
    n_star: int


@dataclass
class FileReport:
    path: str
    filename_time: float
    run_dir: str
    size_bytes: int


@dataclass
class OverlapCandidate:
    kind: str  # "duplicate_filename_time" | "directory_range_overlap"
    left: str
    right: str
    detail: str
    overlapping_ttot: List[float] = field(default_factory=list)


@dataclass
class SimuReport:
    simu_name: str
    file_count: int
    total_bytes: int
    ttot_min: float | None
    ttot_max: float | None
    estimated_row_count: int
    run_dirs: List[str]
    gap_warnings: List[str]
    overlap_candidates: List[OverlapCandidate]

    @property
    def is_clean(self) -> bool:
        return not self.overlap_candidates


def _read_step_infos(hdf5_path: str) -> List[StepInfo]:
    """Read (ttot, n_star) for every Step# group via attrs only (no dataset reads)."""
    infos: List[StepInfo] = []
    with h5py.File(hdf5_path, "r") as f:
        for key in f.keys():
            if not key.startswith("Step#"):
                continue
            group = f[key]
            try:
                ttot = float(group.attrs["Time"])
                n_star = int(group.attrs["N_STAR"])
            except KeyError:
                logger.warning(
                    "[%s] %s missing Time/N_STAR attrs; skipping in row/TTOT estimate",
                    hdf5_path,
                    key,
                )
                continue
            infos.append(StepInfo(ttot=ttot, n_star=n_star))
    return infos


def _run_dir_of(hdf5_path: str) -> str:
    return os.path.dirname(hdf5_path)


def _detect_duplicate_filename_times(
    files: List[FileReport], tolerance: float = 1e-9
) -> List[OverlapCandidate]:
    """Flag files from different run directories sharing (nearly) the same filename time."""
    by_time: Dict[float, List[FileReport]] = defaultdict(list)
    for file_report in files:
        by_time[round(file_report.filename_time / tolerance) * tolerance].append(file_report)

    candidates: List[OverlapCandidate] = []
    for _, group in by_time.items():
        run_dirs = {fr.run_dir for fr in group}
        if len(run_dirs) < 2:
            continue
        for left, right in combinations(group, 2):
            if left.run_dir == right.run_dir:
                continue
            candidates.append(
                OverlapCandidate(
                    kind="duplicate_filename_time",
                    left=left.path,
                    right=right.path,
                    detail=f"both claim filename time ~{left.filename_time:.6f}",
                )
            )
    return candidates


def _detect_directory_range_overlaps(files: List[FileReport]) -> List[OverlapCandidate]:
    """Flag run-directory pairs whose filename-time ranges overlap."""
    ranges: Dict[str, Tuple[float, float]] = {}
    for file_report in files:
        lo, hi = ranges.get(file_report.run_dir, (float("inf"), float("-inf")))
        ranges[file_report.run_dir] = (
            min(lo, file_report.filename_time),
            max(hi, file_report.filename_time),
        )

    candidates: List[OverlapCandidate] = []
    for (dir_a, (lo_a, hi_a)), (dir_b, (lo_b, hi_b)) in combinations(ranges.items(), 2):
        overlap_lo, overlap_hi = max(lo_a, lo_b), min(hi_a, hi_b)
        if overlap_lo <= overlap_hi:
            candidates.append(
                OverlapCandidate(
                    kind="directory_range_overlap",
                    left=dir_a,
                    right=dir_b,
                    detail=(
                        f"[{dir_a}] [{lo_a:.6f}, {hi_a:.6f}] overlaps "
                        f"[{dir_b}] [{lo_b:.6f}, {hi_b:.6f}] "
                        f"on [{overlap_lo:.6f}, {overlap_hi:.6f}]"
                    ),
                )
            )
    return candidates


def _resolve_real_overlap(candidate: OverlapCandidate) -> OverlapCandidate:
    """Read real TTOT sets for a flagged pair and report the actual overlap (may be empty)."""
    left_ttots = {info.ttot for info in _read_step_infos(candidate.left)}
    right_paths = (
        [candidate.right]
        if candidate.kind == "duplicate_filename_time"
        else _sibling_files_in_dir(candidate.right)
    )
    right_ttots: set[float] = set()
    for path in right_paths:
        right_ttots.update(info.ttot for info in _read_step_infos(path))
    if candidate.kind == "directory_range_overlap":
        left_ttots = set()
        for path in _sibling_files_in_dir(candidate.left):
            left_ttots.update(info.ttot for info in _read_step_infos(path))
    overlap = sorted(left_ttots & right_ttots)
    candidate.overlapping_ttot = overlap
    return candidate


def _sibling_files_in_dir(one_file_in_dir: str) -> List[str]:
    if os.path.isdir(one_file_in_dir):
        directory = one_file_in_dir
    else:
        directory = os.path.dirname(one_file_in_dir)
    return sorted(str(p) for p in Path(directory).glob("*.h5part"))


def _gap_warnings(files: List[FileReport]) -> List[str]:
    """Flag filename-time gaps that are much larger than the typical spacing."""
    times = sorted(fr.filename_time for fr in files)
    if len(times) < 3:
        return []
    diffs = np.diff(times)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return []
    typical = float(np.median(positive_diffs))
    if typical <= 0:
        return []
    warnings = []
    for i, diff in enumerate(diffs):
        if diff > typical * 1.5:
            warnings.append(
                f"gap {diff:.6f} between t={times[i]:.6f} and t={times[i + 1]:.6f} "
                f"(typical spacing {typical:.6f})"
            )
    return warnings


def preflight_simu(
    hdf5_file_processor: HDF5FileProcessor, simu_name: str, *, resolve_real_overlaps: bool = True
) -> SimuReport:
    """Build the full-rebuild pre-flight report for one simulation."""
    paths = hdf5_file_processor.get_all_hdf5_paths(simu_name, sample_every_nb_time=None)
    files = [
        FileReport(
            path=path,
            filename_time=hdf5_file_processor.get_hdf5_file_time_from_filename(path),
            run_dir=_run_dir_of(path),
            size_bytes=os.path.getsize(path),
        )
        for path in paths
    ]

    candidates = _detect_duplicate_filename_times(files)
    candidates += _detect_directory_range_overlaps(files)
    if resolve_real_overlaps:
        resolved = []
        with Progress() as progress:
            description = f"{simu_name}: resolving overlap candidates"
            task = progress.add_task(description, total=len(candidates))
            for candidate in candidates:
                resolved.append(_resolve_real_overlap(candidate))
                progress.advance(task)
        candidates = [c for c in resolved if c.overlapping_ttot]

    estimated_row_count = 0
    ttot_min: float | None = None
    ttot_max: float | None = None
    with Progress() as progress:
        task = progress.add_task(f"{simu_name}: reading Step attrs", total=len(files))
        for file_report in files:
            for info in _read_step_infos(file_report.path):
                estimated_row_count += info.n_star
                ttot_min = info.ttot if ttot_min is None else min(ttot_min, info.ttot)
                ttot_max = info.ttot if ttot_max is None else max(ttot_max, info.ttot)
            progress.advance(task)

    return SimuReport(
        simu_name=simu_name,
        file_count=len(files),
        total_bytes=sum(fr.size_bytes for fr in files),
        ttot_min=ttot_min,
        ttot_max=ttot_max,
        estimated_row_count=estimated_row_count,
        run_dirs=sorted({fr.run_dir for fr in files}),
        gap_warnings=_gap_warnings(files),
        overlap_candidates=candidates,
    )


def _print_report(report: SimuReport) -> None:
    print(f"\n=== {report.simu_name} ===")
    print(f"files:            {report.file_count}")
    print(f"total size:       {report.total_bytes / 1024**4:.3f} TiB")
    if report.ttot_min is not None:
        print(f"TTOT range:       [{report.ttot_min:.6f}, {report.ttot_max:.6f}]")
    print(f"estimated rows:   {report.estimated_row_count:,} (sum of N_STAR over all Steps)")
    print(f"run dirs ({len(report.run_dirs)}):")
    for run_dir in report.run_dirs:
        print(f"  - {run_dir}")
    if report.gap_warnings:
        print(f"gap warnings ({len(report.gap_warnings)}):")
        for warning in report.gap_warnings:
            print(f"  ! {warning}")
    if report.overlap_candidates:
        n_candidates = len(report.overlap_candidates)
        print(f"OVERLAPS FOUND ({n_candidates}) -- not safe for full rebuild yet:")
        for candidate in report.overlap_candidates:
            print(f"  [{candidate.kind}] {candidate.detail}")
            if candidate.overlapping_ttot:
                shown = candidate.overlapping_ttot[:10]
                n_more = len(candidate.overlapping_ttot) - 10
                more = "" if n_more <= 0 else f" (+{n_more} more)"
                print(f"      overlapping TTOT: {shown}{more}")
    else:
        print("no overlaps found: clean for full rebuild")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to user config YAML")
    parser.add_argument(
        "--simu", dest="simu_name", help="Limit to one simulation (default: all configured)"
    )
    parser.add_argument("--json", dest="json_path", help="Also write the full report as JSON")
    parser.add_argument(
        "--skip-real-overlap-check",
        action="store_true",
        help=(
            "Skip opening candidate files to confirm real TTOT overlaps "
            "(filename-time heuristics only)"
        ),
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    config = ConfigManager(config_path=args.config)
    hdf5_file_processor = HDF5FileProcessor(config)

    simu_names = [args.simu_name] if args.simu_name else list(config.pathof.keys())
    unknown = [name for name in simu_names if name not in config.pathof]
    if unknown:
        parser.error(f"Unknown simulation(s): {unknown}")

    reports = []
    for simu_name in simu_names:
        report = preflight_simu(
            hdf5_file_processor, simu_name, resolve_real_overlaps=not args.skip_real_overlap_check
        )
        reports.append(report)
        _print_report(report)

    if args.json_path:
        payload = [asdict(report) for report in reports]
        Path(args.json_path).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON report to {args.json_path}")

    return 0 if all(report.is_clean for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
