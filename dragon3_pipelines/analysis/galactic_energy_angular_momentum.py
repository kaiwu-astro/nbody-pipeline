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
    """Evaluate MWPotential2014 in physical ``(km/s)^2`` units."""
    from galpy.potential import evaluatePotentials

    return np.asarray(evaluatePotentials(_mw_potential(), radius_kpc, z_kpc, phi=phi_rad))


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
                "raw RG/VG are interpreted as pc/km/s."
            )
            _HAS_LOGGED_MW_POTENTIAL = True

        result = single_df_at_t.copy()
        mass_msun = result["M"].to_numpy(dtype=float)

        x_pc = result["X [pc]"].to_numpy(dtype=float) + float(scalar_row_at_t["RG(1)"])
        y_pc = result["Y [pc]"].to_numpy(dtype=float) + float(scalar_row_at_t["RG(2)"])
        z_pc = result["Z [pc]"].to_numpy(dtype=float) + float(scalar_row_at_t["RG(3)"])
        vx_km_s = result["V1"].to_numpy(dtype=float) + float(scalar_row_at_t["VG(1)"])
        vy_km_s = result["V2"].to_numpy(dtype=float) + float(scalar_row_at_t["VG(2)"])
        vz_km_s = result["V3"].to_numpy(dtype=float) + float(scalar_row_at_t["VG(3)"])

        x_kpc = x_pc / 1000.0
        y_kpc = y_pc / 1000.0
        z_kpc = z_pc / 1000.0
        radius_kpc = np.sqrt(x_kpc**2 + y_kpc**2)
        phi_rad = np.arctan2(y_pc, x_pc)

        potential_km2_s2 = _evaluate_mw_potential(radius_kpc, z_kpc, phi_rad)
        kinetic = 0.5 * mass_msun * (vx_km_s**2 + vy_km_s**2 + vz_km_s**2)
        potential = mass_msun * potential_km2_s2
        l_z = mass_msun * (x_kpc * vy_km_s - y_kpc * vx_km_s)

        result[E_KIN_GAL_COL] = kinetic
        result[E_POT_GAL_COL] = potential
        result[E_GAL_COL] = kinetic + potential
        result[L_Z_GAL_COL] = l_z
        return result
