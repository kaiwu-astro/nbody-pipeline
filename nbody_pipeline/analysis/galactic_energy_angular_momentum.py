"""Snapshot-level galactic energy and angular-momentum calculations."""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

E_KIN_GAL_COL = "E_kin_gal[Msun*(km/s)^2]"
E_POT_GAL_COL = "E_pot_gal[Msun*(km/s)^2]"
E_GAL_COL = "E_gal[Msun*(km/s)^2]"
L_Z_GAL_COL = "L_z_gal[Msun*kpc*km/s]"

# Specific (per-unit-mass) counterparts: not multiplied by the star's own
# mass, so a low-mass and a high-mass star at the same galactocentric phase-
# space point land on the same value.
E_KIN_GAL_SPECIFIC_COL = "E_kin_gal_specific[(km/s)^2]"
E_POT_GAL_SPECIFIC_COL = "E_pot_gal_specific[(km/s)^2]"
E_GAL_SPECIFIC_COL = "E_gal_specific[(km/s)^2]"
L_Z_GAL_SPECIFIC_COL = "L_z_gal_specific[kpc*km/s]"

_MW_POTENTIAL: Any | None = None
_HAS_LOGGED_MW_POTENTIAL = False


def _mw_potential() -> Any:
    """Return a process-local physical MWPotential2014 instance."""
    global _MW_POTENTIAL
    if _MW_POTENTIAL is None:
        from galpy.potential import MWPotential2014, turn_physical_on

        _MW_POTENTIAL = deepcopy(MWPotential2014)
        turn_physical_on(_MW_POTENTIAL, ro=8.0, vo=220.0)
    return _MW_POTENTIAL


def _evaluate_mw_potential(
    radius_kpc: np.ndarray, z_kpc: np.ndarray, phi_rad: np.ndarray
) -> np.ndarray:
    """Evaluate MWPotential2014 in physical ``(km/s)^2`` units.

    ``_mw_potential()`` has physical output turned on (``ro=8, vo=220``).
    With physical output on, galpy interprets *bare* floats/arrays as
    dimensionless natural units (i.e. already divided by ``ro``/``vo``), not
    kpc/km-s — passing plain ``radius_kpc`` here would silently evaluate the
    potential at ``radius_kpc * ro`` (8x too far out). Attaching astropy
    units makes galpy do the ro/vo division itself and return a plain
    ``km^2/s^2``-valued array.
    """
    from galpy.potential import evaluatePotentials
    import astropy.units as u

    return np.asarray(
        evaluatePotentials(_mw_potential(), radius_kpc * u.kpc, z_kpc * u.kpc, phi=phi_rad * u.rad)
    )


def _scaled_rg_vg(scalar_row_at_t: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return cluster-COM ``(RG, VG)`` converted from raw N-body units to pc/km-s.

    NBODY6++GPU stores ``RG(1-3)``/``VG(1-3)`` in the HDF5 scalars table in raw
    N-body (Henon) units; it converts kpc/km-s to N-body units on input
    (``xtrnl0.F``) but writes the scalars block back out without re-scaling by
    ``RSCALE_OUT``/``VSCALE_OUT`` the way it does for ``X1``/``V1`` singles
    columns. So this must mirror the ``RDENS(1-3) * RBAR`` treatment already
    used for the cluster density center in ``hdf5_reader.py``.
    """
    rbar_pc = float(scalar_row_at_t["RBAR"])
    vstar_kms = float(scalar_row_at_t["VSTAR"])
    rg_pc = np.array([scalar_row_at_t[f"RG({i})"] for i in (1, 2, 3)], dtype=float) * rbar_pc
    vg_kms = np.array([scalar_row_at_t[f"VG({i})"] for i in (1, 2, 3)], dtype=float) * vstar_kms
    return rg_pc, vg_kms


class GalacticEnergyAngularMomentumProcessor:
    """Compute galactocentric energy and ``L_z`` for stars in one snapshot."""

    def __init__(self, config: Any | None = None) -> None:
        self.config = config

    def compute_snapshot(
        self, single_df_at_t: pd.DataFrame, scalar_row_at_t: pd.Series
    ) -> pd.DataFrame:
        """Return a copy of ``single_df_at_t`` with galactic ``E`` and ``L_z`` columns."""
        global _HAS_LOGGED_MW_POTENTIAL
        if not _HAS_LOGGED_MW_POTENTIAL:
            logger.info(
                "Computing galactic E/Lz with galpy.potential.MWPotential2014; "
                "raw RG/VG are in N-body units and scaled by RBAR/VSTAR to pc/km/s."
            )
            _HAS_LOGGED_MW_POTENTIAL = True

        result = single_df_at_t.copy()
        mass_msun = result["M"].to_numpy(dtype=float)
        rg_pc, vg_kms = _scaled_rg_vg(scalar_row_at_t)

        x_pc = result["X [pc]"].to_numpy(dtype=float) + rg_pc[0]
        y_pc = result["Y [pc]"].to_numpy(dtype=float) + rg_pc[1]
        z_pc = result["Z [pc]"].to_numpy(dtype=float) + rg_pc[2]
        vx_km_s = result["V1"].to_numpy(dtype=float) + vg_kms[0]
        vy_km_s = result["V2"].to_numpy(dtype=float) + vg_kms[1]
        vz_km_s = result["V3"].to_numpy(dtype=float) + vg_kms[2]

        x_kpc = x_pc / 1000.0
        y_kpc = y_pc / 1000.0
        z_kpc = z_pc / 1000.0
        radius_kpc = np.sqrt(x_kpc**2 + y_kpc**2)
        phi_rad = np.arctan2(y_pc, x_pc)

        potential_km2_s2 = _evaluate_mw_potential(radius_kpc, z_kpc, phi_rad)
        kinetic_specific = 0.5 * (vx_km_s**2 + vy_km_s**2 + vz_km_s**2)
        l_z_specific = x_kpc * vy_km_s - y_kpc * vx_km_s

        result[E_KIN_GAL_SPECIFIC_COL] = kinetic_specific
        result[E_POT_GAL_SPECIFIC_COL] = potential_km2_s2
        result[E_GAL_SPECIFIC_COL] = kinetic_specific + potential_km2_s2
        result[L_Z_GAL_SPECIFIC_COL] = l_z_specific

        result[E_KIN_GAL_COL] = mass_msun * kinetic_specific
        result[E_POT_GAL_COL] = mass_msun * potential_km2_s2
        result[E_GAL_COL] = mass_msun * (kinetic_specific + potential_km2_s2)
        result[L_Z_GAL_COL] = mass_msun * l_z_specific
        return result

    def compute_cluster_com(
        self, scalar_row_at_t: pd.Series, representative_mass_msun: float | None = None
    ) -> dict[str, float]:
        """Return the cluster COM point in the same galactic E/Lz frame as ``compute_snapshot``.

        The cluster's own bulk position/velocity (``RG``/``VG``) is a single
        point with no internal spread, so it is reported as specific
        (per-unit-mass) quantities. If ``representative_mass_msun`` is given
        (e.g. the mean stellar mass in the snapshot), mass-weighted
        counterparts are also returned so the point can be overlaid on the
        mass-weighted scatter for a like-for-like visual comparison.
        """
        rg_pc, vg_kms = _scaled_rg_vg(scalar_row_at_t)
        x_kpc, y_kpc, z_kpc = rg_pc / 1000.0
        radius_kpc = np.hypot(x_kpc, y_kpc)
        phi_rad = np.arctan2(rg_pc[1], rg_pc[0])

        potential_km2_s2 = float(
            _evaluate_mw_potential(np.array([radius_kpc]), np.array([z_kpc]), np.array([phi_rad]))[
                0
            ]
        )
        kinetic_specific = 0.5 * float(np.sum(vg_kms**2))
        l_z_specific = float(x_kpc * vg_kms[1] - y_kpc * vg_kms[0])

        result = {
            E_KIN_GAL_SPECIFIC_COL: kinetic_specific,
            E_POT_GAL_SPECIFIC_COL: potential_km2_s2,
            E_GAL_SPECIFIC_COL: kinetic_specific + potential_km2_s2,
            L_Z_GAL_SPECIFIC_COL: l_z_specific,
        }
        if representative_mass_msun is not None:
            m = float(representative_mass_msun)
            result.update(
                {
                    E_KIN_GAL_COL: m * kinetic_specific,
                    E_POT_GAL_COL: m * potential_km2_s2,
                    E_GAL_COL: m * (kinetic_specific + potential_km2_s2),
                    L_Z_GAL_COL: m * l_z_specific,
                }
            )
        return result
