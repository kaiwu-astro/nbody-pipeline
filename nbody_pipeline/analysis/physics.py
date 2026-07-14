"""Physics calculations and formulae for astrophysical simulations"""

from typing import Tuple, Union
import numpy as np
import astropy.constants as constants
import astropy.units as u

# Re-export tau_gw from io module where it was already migrated
from nbody_pipeline.io.text_parsers import tau_gw

__all__ = [
    "tau_gw",
    "compute_binary_orbit_relative_positions",
    "compute_individual_orbit_params",
    "binding_energy_nb",
    "ebind_over_kt",
]

ArrayLike = Union[float, np.ndarray]

_PC_TO_AU = constants.pc.to(u.AU).value


def binding_energy_nb(
    m1_msun: ArrayLike,
    m2_msun: ArrayLike,
    a_au: ArrayLike,
    *,
    zmbar_msun: ArrayLike,
    rbar_pc: ArrayLike,
) -> ArrayLike:
    """Binary binding-energy magnitude in N-body units.

    Reproduces ``nbody_pipeline.io.hdf5_reader.HDF5FileProcessor``'s
    ``Ebind_abs_NBODY`` column (``hdf5_reader.py`` ~L352-362) exactly, so
    step2 results stay numerically comparable to the existing step1 CSVs:
    ``m1_nb * m2_nb / (2 * a_nb) / (m1_nb + m2_nb)``, i.e. reduced mass over
    twice the semi-major axis (all in N-body units). Note this divides by
    the *binary's own* total mass, not the cluster's -- that is what
    ``hdf5_reader.py`` does, kept as-is here for consistency rather than
    "corrected" against the textbook ``-G*m1*m2/(2a)`` form.
    """
    m1_nb = np.asarray(m1_msun, dtype=float) / zmbar_msun
    m2_nb = np.asarray(m2_msun, dtype=float) / zmbar_msun
    a_nb = np.asarray(a_au, dtype=float) / _PC_TO_AU / rbar_pc
    return m1_nb * m2_nb / (2.0 * a_nb) / (m1_nb + m2_nb)


def ebind_over_kt(ebind_nb: ArrayLike, eclose_nb: ArrayLike) -> ArrayLike:
    """Binding energy in units of the hard/soft threshold ``eclose_nb``.

    Plain division -- reusable for either the true per-snapshot ``ECLOSE``
    (``snapshot_scalars.eclose_nb`` in the particle lake, which varies over
    the run because ``adjust.F`` can re-tune it) or a fixed constant.

    step1 (``hdf5_reader.py:363``) always divides by the fixed config
    constant ``config.ECLOSE_INPUT`` (default 1.0), *not* the real
    per-snapshot ``eclose_nb`` -- even though ``eclose_nb`` varies
    substantially within a single simulation (e.g. 0.003-1.0 in 20sb). This
    is a known normalization issue in step1, out of scope to fix here; for
    numerical continuity with step1's ``Ebind/kT``/``is_hard_binary``
    columns, step2 callers should pass ``config.ECLOSE_INPUT`` (i.e. 1.0),
    not the true ``eclose_nb``, and label results accordingly (see
    ``examples/gaia_bh_formation/step2/findings.md``). A proper fix (using
    the real time-varying threshold) is tracked as a separate future
    project.
    """
    return np.asarray(ebind_nb, dtype=float) / eclose_nb


def compute_binary_orbit_relative_positions(
    m1: float, m2: float, rel_x: float, rel_y: float, rel_z: float
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute the positions of two stars relative to their common center of mass.

    Given the relative position vector (r_rel = r_2 - r_1), this function computes
    the positions of each star relative to the center of mass:
        r_1 = -M_2 / (M_1 + M_2) * r_rel
        r_2 = M_1 / (M_1 + M_2) * r_rel

    Args:
        m1: Mass of the primary star [any unit, same as m2]
        m2: Mass of the secondary star [any unit, same as m1]
        rel_x: X component of relative position vector (r_2 - r_1)
        rel_y: Y component of relative position vector (r_2 - r_1)
        rel_z: Z component of relative position vector (r_2 - r_1)

    Returns:
        Tuple of two tuples:
            - (x1, y1, z1): Position of primary star relative to center of mass
            - (x2, y2, z2): Position of secondary star relative to center of mass
    """
    total_mass = m1 + m2
    if total_mass == 0:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    # Primary star position relative to center of mass
    frac1 = -m2 / total_mass
    x1 = frac1 * rel_x
    y1 = frac1 * rel_y
    z1 = frac1 * rel_z

    # Secondary star position relative to center of mass
    frac2 = m1 / total_mass
    x2 = frac2 * rel_x
    y2 = frac2 * rel_y
    z2 = frac2 * rel_z

    return (x1, y1, z1), (x2, y2, z2)


def compute_individual_orbit_params(
    a_bin: float, ecc_bin: float, m1: float, m2: float
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute the orbital parameters for each star around the center of mass.

    For a binary system with semi-major axis a and eccentricity e, each star
    orbits the center of mass with:
        a_1 = a * M_2 / (M_1 + M_2)
        a_2 = a * M_1 / (M_1 + M_2)
        e_1 = e_2 = e  (eccentricity is the same for both orbits)

    Args:
        a_bin: Semi-major axis of the binary [any unit]
        ecc_bin: Eccentricity of the binary orbit
        m1: Mass of the primary star [any unit, same as m2]
        m2: Mass of the secondary star [any unit, same as m1]

    Returns:
        Tuple of two tuples:
            - (a1, e1): Semi-major axis and eccentricity for primary star
            - (a2, e2): Semi-major axis and eccentricity for secondary star
    """
    total_mass = m1 + m2
    if total_mass == 0:
        return (0.0, ecc_bin), (0.0, ecc_bin)

    a1 = a_bin * m2 / total_mass
    a2 = a_bin * m1 / total_mass

    return (a1, ecc_bin), (a2, ecc_bin)


def orbit_elements_from_state(
    r: np.ndarray, v: np.ndarray, mu: float
) -> Tuple[float, float, float, float, float]:
    """
    Compute Keplerian orbital elements from state vector (r, v).

    Args:
        r: Position vector [3-element array]
        v: Velocity vector [3-element array]
        mu: Standard gravitational parameter (G * total_mass)

    Returns:
        Tuple containing (a, e, i, Omega, omega):
            - a: Semi-major axis
            - e: Eccentricity
            - i: Inclination [radians]
            - Omega: Longitude of ascending node [radians]
            - omega: Argument of periapsis [radians]
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rnorm = np.linalg.norm(r)
    vnorm = np.linalg.norm(v)

    h = np.cross(r, v)
    hnorm = np.linalg.norm(h)

    # eccentricity vector
    e_vec = (np.cross(v, h) / mu) - (r / rnorm)
    e = np.linalg.norm(e_vec)

    # semi-major axis from vis-viva
    a = 1.0 / (2.0 / rnorm - (vnorm * vnorm) / mu)

    # inclination
    i = np.arccos(np.clip(h[2] / hnorm, -1.0, 1.0))

    # node vector
    k = np.array([0.0, 0.0, 1.0])
    n = np.cross(k, h)
    nnorm = np.linalg.norm(n)

    # RAAN Omega
    if nnorm < 1e-14:
        Omega = 0.0
    else:
        Omega = np.arctan2(n[1], n[0]) % (2 * np.pi)

    # argument of periapsis omega
    if e < 1e-14 or nnorm < 1e-14:
        omega = 0.0
    else:
        # omega = angle from n to e_vec in orbital plane
        cosw = np.clip(np.dot(n, e_vec) / (nnorm * e), -1.0, 1.0)
        omega = np.arccos(cosw)
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega

    return a, e, i, Omega, omega


def rot_matrix(Omega: float, inc: float, omega: float) -> np.ndarray:
    """
    Compute the rotation matrix from perifocal (PQW) to inertial coordinates.

    Args:
        Omega: Longitude of ascending node [radians]
        inc: Inclination [radians]
        omega: Argument of periapsis [radians]

    Returns:
        3x3 rotation matrix (PQW -> inertial)
    """
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(omega), np.sin(omega)

    RzO = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])
    Rxi = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    Rzo = np.array([[co, -so, 0], [so, co, 0], [0, 0, 1]])
    return RzO @ Rxi @ Rzo  # PQW -> inertial


def sample_relative_orbit_xy(
    a_rel: float, e: float, i: float, Omega: float, omega: float, n: int = 600
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample the relative orbit in inertial coordinates and return its xy projection.

    Args:
        a_rel: Semi-major axis of the relative orbit
        e: Eccentricity
        i: Inclination [radians]
        Omega: Longitude of ascending node [radians]
        omega: Argument of periapsis [radians]
        n: Number of points to sample

    Returns:
        Tuple of (x_points, y_points) representing the projected orbit
    """
    nu = np.linspace(0, 2 * np.pi, n)
    p = a_rel * (1 - e * e)
    r = p / (1 + e * np.cos(nu))

    # Perifocal (PQW) coordinates
    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)
    z_p = np.zeros_like(x_p)
    rpqw = np.vstack([x_p, y_p, z_p])  # (3,n)

    R = rot_matrix(Omega, i, omega)
    rI = R @ rpqw  # inertial (3,n)
    return rI[0], rI[1]  # xy projection
