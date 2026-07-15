"""Physics calculations and formulae for astrophysical simulations"""

from typing import Tuple, Union
import numpy as np
import pandas as pd
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
    "TEMPORARY_EBIND_FACTOR",
    "BINARY_CLASS_HARD",
    "BINARY_CLASS_SOFT",
    "BINARY_CLASS_TEMPORARY",
    "BINARY_CLASS_ORDER",
    "mean_core_interparticle_distance_au",
    "classify_binaries",
    "add_binary_energetics_and_class",
    "drop_temporary_binaries",
]

ArrayLike = Union[float, np.ndarray]

_PC_TO_AU = constants.pc.to(u.AU).value

# 2026-07 会议（examples/soft-hard-temp/meeting.md）定死的科学定义：暂时双星判据用
# Ebind < TEMPORARY_EBIND_FACTOR * ECLOSE。不做成 config 项 -- ConfigManager._merge_user_config
# 不合并用户 `physics:` 段，做了也只会静默用默认值；且分母/分类/图注三处需严格一致。
TEMPORARY_EBIND_FACTOR: float = 1.0e-3

BINARY_CLASS_HARD = "hard"
BINARY_CLASS_SOFT = "soft"
BINARY_CLASS_TEMPORARY = "temporary"
BINARY_CLASS_ORDER = (BINARY_CLASS_HARD, BINARY_CLASS_SOFT, BINARY_CLASS_TEMPORARY)


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
    """Binding energy in units of a hard/soft threshold.

    Plain division -- the caller decides what threshold to divide by.

    As of the 2026-07 soft/hard/temporary reclassification (see
    ``examples/soft-hard-temp/meeting.md``), ``hdf5_reader.py``'s ``Ebind/kT``
    column divides by ``TEMPORARY_EBIND_FACTOR * eclose_nb`` (the true
    per-snapshot ``ECLOSE``, ``snapshot_scalars.eclose_nb`` in the particle
    lake, which varies substantially within a run -- e.g. 0.003-1.0 in 20sb
    -- because ``adjust.F`` re-tunes it), replacing the old fixed
    ``config.ECLOSE_INPUT`` denominator. Older step2 scripts (e.g.
    ``examples/gaia_bh_formation/step2/11_extract_lake_timelines.py``) still
    pass ``config.ECLOSE_INPUT`` here for continuity with archived results;
    new callers should pass ``TEMPORARY_EBIND_FACTOR * eclose_nb`` (see
    ``classify_binaries``/``add_binary_energetics_and_class`` below) unless
    they specifically need the legacy normalization.
    """
    return np.asarray(ebind_nb, dtype=float) / eclose_nb


def mean_core_interparticle_distance_au(
    nc: ArrayLike,
    rc_nb: ArrayLike,
    *,
    rbar_pc: ArrayLike,
) -> np.ndarray:
    """Mean inter-particle spacing near the cluster core, from NC/RC geometry.

    ``n_c = 3*NC / (4*pi*RC^3)`` (core number density, N-body units), then
    ``d = n_c^(-1/3)`` (N-body length), converted to AU via ``RBAR``. Used as
    the "temporary binary" distance threshold: a binary wider than this is
    presumably not a real bound pair, just two unrelated stars passing close
    in the dense core.

    ``NC <= 0``, ``RC <= 0``, or NaN inputs return NaN (e.g. at ``t=0``
    before ``adjust.F`` has defined a core) rather than raising or dividing
    by zero.
    """
    nc_arr = np.asarray(nc, dtype=float)
    rc_arr = np.asarray(rc_nb, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d_nb = np.cbrt(4.0 * np.pi * rc_arr**3 / (3.0 * nc_arr))
    invalid = ~(nc_arr > 0) | ~(rc_arr > 0)
    d_nb = np.where(invalid, np.nan, d_nb)
    return d_nb * np.asarray(rbar_pc, dtype=float) * _PC_TO_AU


def classify_binaries(
    bin_label: ArrayLike,
    a_au: ArrayLike,
    ebind_nb: ArrayLike,
    *,
    eclose_nb: ArrayLike,
    mean_core_distance_au: ArrayLike,
    temporary_ebind_factor: float = TEMPORARY_EBIND_FACTOR,
) -> np.ndarray:
    """Classify binaries as hard / temporary / soft (checked in that priority order).

    - **hard**: ``bin_label == 1`` (binary-table "KS binary" label), unconditionally.
    - **temporary**: not hard, *and* wider than the core spacing
      (``a_au > mean_core_distance_au``), *and* weakly bound
      (``ebind_nb < temporary_ebind_factor * eclose_nb``). These are binary-table
      false positives -- two stars caught transiently close together -- expected
      to be disrupted almost immediately.
    - **soft**: everything else.

    Never raises; every edge case degrades to a safe classification via plain
    NaN-comparison semantics (comparisons against NaN are always False):
    ``bin_label`` in ``{-9, 0, -1}`` (unknown/wide/merger-internal) is simply
    not hard and falls through to the energy/distance test; ``eclose_nb <= 0``
    or NaN, ``mean_core_distance_au`` NaN (``NC``/``RC`` = 0, e.g. ``t=0``), or
    NaN ``a_au``/``ebind_nb`` all make the temporary test False, leaving the
    row soft (unless it was already hard).
    """
    bin_label_arr = np.asarray(bin_label)
    a_arr = np.asarray(a_au, dtype=float)
    ebind_arr = np.asarray(ebind_nb, dtype=float)
    eclose_arr = np.asarray(eclose_nb, dtype=float)
    distance_arr = np.asarray(mean_core_distance_au, dtype=float)

    is_hard = bin_label_arr == 1
    is_temporary = (
        ~is_hard
        & (eclose_arr > 0)
        & (a_arr > distance_arr)
        & (ebind_arr < temporary_ebind_factor * eclose_arr)
    )
    return np.select(
        [is_hard, is_temporary],
        [BINARY_CLASS_HARD, BINARY_CLASS_TEMPORARY],
        default=BINARY_CLASS_SOFT,
    )


def add_binary_energetics_and_class(
    binaries: pd.DataFrame,
    scalars: pd.DataFrame,
    *,
    temporary_ebind_factor: float = TEMPORARY_EBIND_FACTOR,
) -> pd.DataFrame:
    """Read-time helper for particle-lake consumers: join per-snapshot scalars
    onto a ``snapshot_binaries`` frame and add ``ebind_nb``,
    ``mean_core_distance_au``, ``ebind_over_kt`` (new ECLOSE-normalized
    definition), and ``binary_class``.

    The lake itself stays schema-frozen (raw columns only, see
    ``nbody_pipeline.analysis.particle_lake``); callers that want
    energetics/classification call this after reading instead. Joins
    ``scalars[["simulation_id", "ttot", "zmbar_msun", "rbar_pc", "eclose_nb",
    "rc_nb", "nc"]]`` onto ``binaries`` on ``["simulation_id", "ttot"]`` (or
    just ``["ttot"]`` if either frame lacks ``simulation_id``, e.g. a
    single-simulation subset). Returns a new DataFrame; ``binaries`` and
    ``scalars`` are not mutated.
    """
    join_cols = ["ttot", "zmbar_msun", "rbar_pc", "eclose_nb", "rc_nb", "nc"]
    join_keys = ["ttot"]
    if "simulation_id" in binaries.columns and "simulation_id" in scalars.columns:
        join_cols = ["simulation_id"] + join_cols
        join_keys = ["simulation_id"] + join_keys

    merged = binaries.merge(scalars[join_cols], on=join_keys, how="left")

    merged["ebind_nb"] = binding_energy_nb(
        merged["mass_1_msun"].to_numpy(dtype=float),
        merged["mass_2_msun"].to_numpy(dtype=float),
        merged["semi_major_axis_au"].to_numpy(dtype=float),
        zmbar_msun=merged["zmbar_msun"].to_numpy(dtype=float),
        rbar_pc=merged["rbar_pc"].to_numpy(dtype=float),
    )
    merged["mean_core_distance_au"] = mean_core_interparticle_distance_au(
        merged["nc"].to_numpy(dtype=float),
        merged["rc_nb"].to_numpy(dtype=float),
        rbar_pc=merged["rbar_pc"].to_numpy(dtype=float),
    )
    merged["ebind_over_kt"] = ebind_over_kt(
        merged["ebind_nb"].to_numpy(dtype=float),
        temporary_ebind_factor * merged["eclose_nb"].to_numpy(dtype=float),
    )
    merged["binary_class"] = classify_binaries(
        merged["bin_label"].to_numpy(),
        merged["semi_major_axis_au"].to_numpy(dtype=float),
        merged["ebind_nb"].to_numpy(dtype=float),
        eclose_nb=merged["eclose_nb"].to_numpy(dtype=float),
        mean_core_distance_au=merged["mean_core_distance_au"].to_numpy(dtype=float),
        temporary_ebind_factor=temporary_ebind_factor,
    )
    return merged


def drop_temporary_binaries(
    binary_df: pd.DataFrame, *, class_col: str = "binary_class"
) -> pd.DataFrame:
    """Remove rows classified as ``temporary`` (binary-table false positives).

    Rows with a missing ``class_col`` (column absent) or NaN class are kept
    conservatively -- we don't know they're safe to drop. Returns a copy;
    ``binary_df`` is not mutated.
    """
    if class_col not in binary_df.columns:
        return binary_df.copy()
    is_temporary = binary_df[class_col] == BINARY_CLASS_TEMPORARY
    return binary_df.loc[~is_temporary].copy()


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
