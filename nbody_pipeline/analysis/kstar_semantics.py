"""Decode raw NBODY6++GPU KSTAR/``cm_kw`` codes into physical binary states.

``cm_kw`` (HDF5 Item 160, "Bin cm KW") is the raw ``KSTAR(I)`` value of a
binary's centre-of-mass particle. Its odd/even semantics are stated
*backwards* in the NBODY6++GPU manual (``Nb6manual.md`` line 1431: "even
number >=10 = subsequent phases *of* mass transfer, odd number >=11 =
subsequent phases *after* mass transfer"). The manual's own text contradicts
the code it documents.

The actual runtime behaviour, confirmed by reading ``roche.f`` and
``expel.f`` in Nbody6PPGPU-beijing/src/Main (2026-07-14 review, see project
memory ``gaia-bh-step2-kstar-semantics``):

- ``roche.f`` ~L246-249: entering an active Roche-lobe-overflow episode
  ("NEW ROCHE") increments ``KSTAR(I)`` from an even value (or <=10) to the
  next odd value.
- ``roche.f`` ~L1786: leaving the overflow episode ("END ROCHE", the star no
  longer fills its Roche lobe) increments ``KSTAR(I)`` from odd to even.
- ``roche.f`` ~L1805: ``IF (MOD(KSTAR(I),2).EQ.1...)`` is explicitly
  commented "Set new c.m. time only during active stage" -- i.e. odd is the
  active state.
- ``custom_output.F`` writes ``NB_KWC(IK) = KSTAR(I)`` verbatim (Item 160 is
  the raw value, no transform).

So: **odd values >=11 mean mass transfer is ongoing right now; even values
>=10 mean the binary is between episodes (already circularized/flagged but
not currently overflowing)**. This module follows the code, not the manual
prose. This matches the existing (and correct) convention already used by
``nbody_pipeline.io.hdf5_reader.HDF5FileProcessor.mark_funny_star_binary``
and documented in ``schemas/snapshot_binaries.yaml`` -- no fix was needed
there.

Member KW codes (Item 158/159, ``kw_1``/``kw_2``) use a separate convention:
values above 100 flag a common-envelope object, with the "real" stellar type
recovered by subtracting 100 (see ``expel.f``/``roche.f``,
e.g. ``IF (KW1.GT.100) KW1 = KW1 - 100``).

``cm_kw == -1`` is a third, unrelated convention: ``chaos.f`` (Mardling 1995
chaotic-tidal-interaction physics, distinct from Roche-lobe mass transfer)
sets ``KSTAR(I) = -1`` on the centre-of-mass particle while a highly
eccentric binary is undergoing chaotic tidal energy dissipation. This is
common in practice -- on real 0sb data, ~11% of target-binary rows sit at
cm_kw==-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

RELATIVISTIC_CM_KW = -25
CHAOTIC_TIDE_CM_KW = -1
COMMON_ENVELOPE_OFFSET = 100

__all__ = [
    "CmKstarState",
    "decode_cm_kstar",
    "decode_member_kw",
    "annotate_binary_states",
]


@dataclass(frozen=True)
class CmKstarState:
    """Decoded physical state of a binary centre-of-mass KSTAR (``cm_kw``) code."""

    raw: int
    is_standard: bool
    mt_ongoing: bool
    mt_past: bool
    mt_phase_index: int
    is_relativistic: bool
    is_chaotic_tide: bool = False


def decode_cm_kstar(cm_kw: int) -> CmKstarState:
    """Decode one raw ``cm_kw`` value into a :class:`CmKstarState`.

    See the module docstring for the odd/even convention (odd = mass
    transfer ongoing, even = between episodes).
    """
    cm_kw = int(cm_kw)

    if cm_kw == RELATIVISTIC_CM_KW:
        return CmKstarState(
            raw=cm_kw,
            is_standard=False,
            mt_ongoing=False,
            mt_past=False,
            mt_phase_index=0,
            is_relativistic=True,
        )

    if cm_kw == CHAOTIC_TIDE_CM_KW:
        return CmKstarState(
            raw=cm_kw,
            is_standard=False,
            mt_ongoing=False,
            mt_past=False,
            mt_phase_index=0,
            is_relativistic=False,
            is_chaotic_tide=True,
        )

    if cm_kw < 10:
        return CmKstarState(
            raw=cm_kw,
            is_standard=(cm_kw == 0),
            mt_ongoing=False,
            mt_past=False,
            mt_phase_index=0,
            is_relativistic=False,
        )

    is_odd = (cm_kw % 2) == 1
    if is_odd:
        return CmKstarState(
            raw=cm_kw,
            is_standard=False,
            mt_ongoing=True,
            mt_past=False,
            mt_phase_index=(cm_kw - 9) // 2,
            is_relativistic=False,
        )
    return CmKstarState(
        raw=cm_kw,
        is_standard=False,
        mt_ongoing=False,
        mt_past=cm_kw > 10,
        mt_phase_index=(cm_kw - 10) // 2,
        is_relativistic=False,
    )


def decode_member_kw(kw: int) -> Tuple[int, bool]:
    """Split a member KW code into ``(base_kw, in_common_envelope)``.

    Values above 100 flag a common-envelope object; the underlying stellar
    type is ``kw - 100``.
    """
    kw = int(kw)
    if kw > COMMON_ENVELOPE_OFFSET:
        return kw - COMMON_ENVELOPE_OFFSET, True
    return kw, False


def annotate_binary_states(
    df: pd.DataFrame,
    *,
    cm_kw_col: str = "cm_kw",
    kw1_col: str = "kw_1",
    kw2_col: str = "kw_2",
) -> pd.DataFrame:
    """Vectorized version of :func:`decode_cm_kstar`/:func:`decode_member_kw`.

    Adds ``mt_ongoing``, ``mt_past``, ``is_relativistic_binary``,
    ``is_chaotic_tide_binary``, ``base_kw_1``, ``base_kw_2``, ``in_ce_1``,
    ``in_ce_2`` columns to a copy of ``df`` and returns it.
    """
    out = df.copy()

    cm_kw = out[cm_kw_col].to_numpy()
    ge10 = cm_kw >= 10
    odd = np.mod(cm_kw, 2) == 1

    out["mt_ongoing"] = ge10 & odd
    out["mt_past"] = ge10 & ~odd & (cm_kw > 10)
    out["is_relativistic_binary"] = cm_kw == RELATIVISTIC_CM_KW
    out["is_chaotic_tide_binary"] = cm_kw == CHAOTIC_TIDE_CM_KW

    for kw_col, base_col, ce_col in (
        (kw1_col, "base_kw_1", "in_ce_1"),
        (kw2_col, "base_kw_2", "in_ce_2"),
    ):
        kw = out[kw_col].to_numpy()
        in_ce = kw > COMMON_ENVELOPE_OFFSET
        out[base_col] = np.where(in_ce, kw - COMMON_ENVELOPE_OFFSET, kw)
        out[ce_col] = in_ce

    return out
