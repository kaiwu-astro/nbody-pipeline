"""Post-hoc reconstruction of ``snapshot_binaries.bin_label`` for source HDF5
files that predate the ``"176 Bin Label"``/``"176 Bin cm Name"`` dataset
(introduced NBODY6++GPU commit ``1381398e``, corrected ``b9ac5a0e`` -- see
``schemas/snapshot_binaries.yaml`` and ``particle_lake.py``'s
``_BIN_LABEL_UNKNOWN``). Files from before that dataset existed have every row
sentineled ``-9`` ("unknown") in the lake; this module recomputes a real label
for those rows from fields present in every source version (>= Aug2021),
without touching rows that already carry a real value.

Algorithm (source-confirmed against NBODY6++GPU, dev branch), checked in this
fixed order per row (wide is intercepted before KS specifically to avoid a
possible ``IWBINC``/KS-name-range numeric collision, see ``cm_id``'s column
description in ``schemas/snapshot_binaries.yaml``):

1. ``cm_id < 0`` -> ``-1`` (merger-internal binary; ``merge.f:263`` negates
   the cm name).
2. ``pert_gamma == 0`` and ``cm_kw == -1`` -> ``0`` (wide branch hard-codes
   ``B_G=0.D0``/``NB_KWC=-1``; requiring both guards against an
   unperturbed-but-real KS binary's GAMMA underflowing to 0 in float32).
3. ``cm_id > NZERO`` and ``cm_id - NZERO`` matches ``object_id_1`` or
   ``object_id_2`` -> ``1`` (KS pair; cm name is ``NZERO`` + first-member NAME
   at pairing time, ``ksinit.F:30``).
4. Otherwise, ``pert_gamma == float32(0.1)`` (the ``GALIMIT`` constant,
   ``custom_output.F:415``) -> ``-1`` (hierarchical/outer binary); anything
   still unclassified is left at ``-9``, conservatively.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Mapping

import duckdb
import numpy as np
import pandas as pd

from nbody_pipeline.analysis.particle_lake import _BIN_LABEL_UNKNOWN

if TYPE_CHECKING:
    from nbody_pipeline.analysis.particle_lake import SnapshotBinariesTask

logger = logging.getLogger(__name__)

ALGORITHM_VERSION = 1

# custom_output.F:415's hard-coded GALIMIT threshold for hierarchical/outer
# binaries not otherwise classified as KS or wide.
_GALIMIT_GAMMA = np.float32(0.1)

_REQUIRED_COLUMNS = (
    "cm_id",
    "object_id_1",
    "object_id_2",
    "pert_gamma",
    "cm_kw",
    "bin_label",
)


def reconstruct_bin_label(df: pd.DataFrame, nzero: int) -> np.ndarray:
    """Return a recomputed ``bin_label`` array: real labels untouched, ``-9`` re-derived.

    Pure/vectorized: does not mutate ``df``. Rows the algorithm cannot classify
    (rule 4 falls through) stay ``-9``.
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"reconstruct_bin_label: missing required columns {missing}")

    bin_label = df["bin_label"].to_numpy(dtype="int32")
    unknown = bin_label == _BIN_LABEL_UNKNOWN
    result = bin_label.copy()
    if not unknown.any():
        return result

    cm_id = df["cm_id"].to_numpy(dtype="int64")
    object_id_1 = df["object_id_1"].to_numpy(dtype="int64")
    object_id_2 = df["object_id_2"].to_numpy(dtype="int64")
    pert_gamma = df["pert_gamma"].to_numpy(dtype="float32")
    cm_kw = df["cm_kw"].to_numpy(dtype="int32")

    # Rule 1: merger-internal binary.
    is_merger = unknown & (cm_id < 0)
    result[is_merger] = -1

    # Rule 2: wide/unperturbed binary -- must run before rule 3 (KS), since
    # IWBINC (the wide branch's per-snapshot fake name counter) can numerically
    # collide with the KS cm-name range.
    remaining = unknown & ~is_merger
    is_wide = remaining & (pert_gamma == np.float32(0.0)) & (cm_kw == -1)
    result[is_wide] = 0

    # Rule 3: KS-regularized binary.
    remaining = remaining & ~is_wide
    ks_cm_id = cm_id - nzero
    is_ks = remaining & (cm_id > nzero) & ((ks_cm_id == object_id_1) | (ks_cm_id == object_id_2))
    result[is_ks] = 1

    # Rule 4: hierarchical/outer binary (GALIMIT path); else stays -9.
    remaining = remaining & ~is_ks
    is_galimit = remaining & (pert_gamma == _GALIMIT_GAMMA)
    result[is_galimit] = -1

    return result.astype("int32")


def resolve_nzero(scalars_df: pd.DataFrame, simulation_id: str, override: int | None = None) -> int:
    """Return NZERO for one simulation: an explicit override, or inferred from
    ``snapshot_scalars``' earliest cached row (its ``n`` column at ``ttot`` == 0).

    Raises ``ValueError`` if inference is required (no override given) but the
    earliest cached ``ttot`` for this simulation is not (numerically) zero --
    NZERO is only meaningfully read off the true ``t=0`` row (``ksinit.F``); a
    cache whose tail starts later (e.g. a restart-only incremental cache)
    cannot be trusted to still reflect the original particle count.
    """
    if override is not None:
        return int(override)

    sim_scalars = scalars_df
    if "simulation_id" in scalars_df.columns:
        sim_scalars = scalars_df.loc[scalars_df["simulation_id"] == simulation_id]
    if sim_scalars.empty:
        raise ValueError(
            f"No cached snapshot_scalars rows for simulation {simulation_id!r}; "
            f"pass --nzero {simulation_id}=N explicitly."
        )

    idx = sim_scalars["ttot"].idxmin()
    ttot_min = float(sim_scalars.loc[idx, "ttot"])
    if not np.isclose(ttot_min, 0.0, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"Cannot auto-infer NZERO for {simulation_id!r}: earliest cached ttot is "
            f"{ttot_min!r}, not 0.0 -- pass --nzero {simulation_id}=N explicitly."
        )
    return int(sim_scalars.loc[idx, "n"])


def backfill_parts(
    task: "SnapshotBinariesTask", nzero_map: Mapping[str, int], *, dry_run: bool = False
) -> Dict[str, Any]:
    """Recompute ``bin_label == -9`` rows in every cached Parquet part of one
    ``SnapshotBinariesTask``, in place.

    Iterates the task's manifest ``processed_files`` (not a raw directory
    glob), so each rewrite can reuse ``write_part`` with the exact same source
    ``hdf5_path`` it was originally written from -- part filenames are a
    deterministic hash of that path (``parquet_cache._part_filename``), so the
    atomic overwrite lands on the same file and neither the manifest nor the
    schema is touched (no column added, no ``schema_hash`` change -- this
    deliberately avoids triggering ``prepare_full_rebuild``). A part with no
    ``-9`` rows, or whose recomputed labels are identical to what's already on
    disk, is left untouched (no write) -- this is what makes repeated runs
    idempotent.
    """
    simulation_id = task.simu_name
    if simulation_id not in nzero_map:
        raise KeyError(f"No NZERO provided for simulation {simulation_id!r}")
    nzero = nzero_map[simulation_id]

    manifest = task.read_meta()
    processed_files: Dict[str, Any] = manifest.get("processed_files", {})
    data_dir = task.data_dir
    column_order = list(task.table_schema.column_names())

    report: Dict[str, Any] = {
        "simulation_id": simulation_id,
        "nzero": nzero,
        "dry_run": dry_run,
        "parts": {},
    }

    for hdf5_path, file_meta in processed_files.items():
        part_name = file_meta.get("part")
        if not part_name:
            continue
        part_path = data_dir / part_name
        if not part_path.exists():
            continue

        # Cheap pre-check: read only the bin_label column (parquet is columnar,
        # so this is a fraction of the part's I/O) before paying for a full read
        # of every other column -- the large majority of parts on a real archive
        # have zero -9 rows and can be skipped this way.
        try:
            bin_label_probe = pd.read_parquet(part_path, columns=["bin_label"])["bin_label"]
        except (KeyError, ValueError):
            continue
        unknown_before = int((bin_label_probe.to_numpy(dtype="int32") == _BIN_LABEL_UNKNOWN).sum())
        if unknown_before == 0:
            continue

        rows = pd.read_parquet(part_path)
        original = rows["bin_label"].to_numpy(dtype="int32")
        new_labels = reconstruct_bin_label(rows, nzero)
        changed = not np.array_equal(new_labels, original)
        counts = {
            "unknown_before": unknown_before,
            "to_hard": int(((original == _BIN_LABEL_UNKNOWN) & (new_labels == 1)).sum()),
            "to_wide": int(((original == _BIN_LABEL_UNKNOWN) & (new_labels == 0)).sum()),
            # -1 covers both rule 1 (merger-internal) and rule 4 (GALIMIT
            # hierarchical) -- bin_label itself has no separate code for the two.
            "to_merger_or_hierarchical": int(
                ((original == _BIN_LABEL_UNKNOWN) & (new_labels == -1)).sum()
            ),
            "still_unknown": int(
                ((original == _BIN_LABEL_UNKNOWN) & (new_labels == _BIN_LABEL_UNKNOWN)).sum()
            ),
            "written": bool(changed and not dry_run),
        }
        report["parts"][part_name] = counts

        if changed and not dry_run:
            rows["bin_label"] = new_labels
            task.write_part(hdf5_path, rows[column_order])

    if not dry_run:
        _write_provenance(task, report)
    return report


def _write_provenance(task: "SnapshotBinariesTask", report: Mapping[str, Any]) -> None:
    """Sidecar recording what backfill_parts did, for later audit -- not a schema column."""
    sidecar_path = task.cache_dir / "backfill_bin_label.json"
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "generated_at": time.time(),
        **report,
    }
    task.cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = sidecar_path.with_suffix(sidecar_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp_path, sidecar_path)


_VALIDATE_DEFAULT_SAMPLE_SIZE = 5_000_000

_VALIDATE_EMPTY_RESULT: Dict[str, Any] = {
    "confusion_matrix": {},
    "n_known": 0,
    "accuracy": None,
    "nzero_consistent": None,
    "sampled": False,
}


def validate_reconstruction(
    task: "SnapshotBinariesTask",
    nzero: int,
    *,
    sample_size: int | None = _VALIDATE_DEFAULT_SAMPLE_SIZE,
) -> Dict[str, Any]:
    """Cross-tabulate reconstructed vs. true ``bin_label`` on a simulation that
    already has real (non ``-9``) labels for at least some rows.

    Run this on a simulation with real labels *before* trusting the algorithm
    on files with no ground truth. For every cached row with a real label, the
    true label is hidden (forced to ``-9``) and ``reconstruct_bin_label`` is
    asked to recover it. Also reports whether
    ``min(cm_id where bin_label == 1) > nzero`` holds, a cheap sanity check on
    ``nzero`` itself, independent of the classification rules.

    Real ``snapshot_binaries`` tables are hundreds of GB / billions of rows --
    far too large to materialize whole in a pandas DataFrame (this has been
    measured to exceed a login node's memory budget). The "known" subset is
    therefore pulled via DuckDB with column projection (only the 6 columns
    ``reconstruct_bin_label`` needs) and, by default, reservoir-sampled down to
    ``sample_size`` rows directly during the Parquet scan -- DuckDB does the
    filtering/sampling out-of-core, so only the (bounded) sample is ever
    materialized in Python. Pass ``sample_size=None`` to read every known row
    instead (fine for small tables, e.g. tests).
    """
    data_dir = task.data_dir
    if not data_dir.exists() or not any(data_dir.glob("*.parquet")):
        return dict(_VALIDATE_EMPTY_RESULT)

    glob_path = str(data_dir / "*.parquet")
    query = (
        "SELECT cm_id, object_id_1, object_id_2, pert_gamma, cm_kw, bin_label "
        f"FROM read_parquet('{glob_path}') WHERE bin_label != {_BIN_LABEL_UNKNOWN}"
    )
    if sample_size is not None:
        query += f" USING SAMPLE reservoir({int(sample_size)} ROWS)"
    known = duckdb.sql(query).df()
    if known.empty:
        return dict(_VALIDATE_EMPTY_RESULT)

    truth = known["bin_label"].to_numpy(dtype="int32")
    forced_unknown = known.copy()
    forced_unknown["bin_label"] = _BIN_LABEL_UNKNOWN
    predicted = reconstruct_bin_label(forced_unknown, nzero)

    confusion = pd.crosstab(
        pd.Series(truth, name="true"), pd.Series(predicted, name="predicted")
    ).to_dict()

    hard_cm_ids = known.loc[known["bin_label"] == 1, "cm_id"]
    nzero_consistent = bool(hard_cm_ids.empty or hard_cm_ids.min() > nzero)

    return {
        "confusion_matrix": confusion,
        "n_known": int(len(known)),
        "accuracy": float((truth == predicted).mean()),
        "nzero_consistent": nzero_consistent,
        "sampled": sample_size is not None,
    }


__all__ = [
    "ALGORITHM_VERSION",
    "reconstruct_bin_label",
    "resolve_nzero",
    "backfill_parts",
    "validate_reconstruction",
]
