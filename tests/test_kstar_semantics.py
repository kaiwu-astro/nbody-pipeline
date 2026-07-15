"""Tests for nbody_pipeline.analysis.kstar_semantics.

Pins down the odd/even ``cm_kw`` convention determined by reading
``roche.f``/``expel.f`` (Nbody6PPGPU-beijing source) on 2026-07-14: odd
values >=11 mean mass transfer is ongoing; even values >=10 mean the binary
is between episodes. See the module docstring for the full derivation and
why this contradicts the manual's prose.
"""

import numpy as np
import pandas as pd
import pytest

from nbody_pipeline.analysis.kstar_semantics import (
    CmKstarState,
    annotate_binary_states,
    decode_cm_kstar,
    decode_member_kw,
)
from nbody_pipeline.analysis.physics import (
    BINARY_CLASS_HARD,
    BINARY_CLASS_SOFT,
    BINARY_CLASS_TEMPORARY,
    add_binary_energetics_and_class,
    binding_energy_nb,
    classify_binaries,
    drop_temporary_binaries,
    ebind_over_kt,
    mean_core_interparticle_distance_au,
)


class TestDecodeCmKstar:
    def test_standard_zero(self):
        state = decode_cm_kstar(0)
        assert state == CmKstarState(
            raw=0,
            is_standard=True,
            mt_ongoing=False,
            mt_past=False,
            mt_phase_index=0,
            is_relativistic=False,
        )

    def test_ten_is_flagged_but_not_yet_transferring(self):
        # 10 is set right after CE circularization, before the first actual
        # Roche-lobe-overflow episode starts (roche.f L246-249 only fires,
        # incrementing to 11, once overflow is detected).
        state = decode_cm_kstar(10)
        assert state.is_standard is False
        assert state.mt_ongoing is False
        assert state.mt_past is False
        assert state.mt_phase_index == 0

    def test_eleven_is_first_ongoing_episode(self):
        state = decode_cm_kstar(11)
        assert state.mt_ongoing is True
        assert state.mt_past is False
        assert state.mt_phase_index == 1

    def test_twelve_is_after_first_episode(self):
        state = decode_cm_kstar(12)
        assert state.mt_ongoing is False
        assert state.mt_past is True
        assert state.mt_phase_index == 1

    def test_thirteen_is_second_ongoing_episode(self):
        state = decode_cm_kstar(13)
        assert state.mt_ongoing is True
        assert state.mt_past is False
        assert state.mt_phase_index == 2

    def test_minus_25_is_relativistic(self):
        state = decode_cm_kstar(-25)
        assert state.is_relativistic is True
        assert state.mt_ongoing is False
        assert state.mt_past is False

    def test_minus_1_is_chaotic_tide(self):
        # chaos.f (Mardling 1995 chaotic-tidal-interaction physics) sets
        # KSTAR(I) = -1 on the c.m. particle; unrelated to mass transfer.
        state = decode_cm_kstar(-1)
        assert state.is_chaotic_tide is True
        assert state.is_relativistic is False
        assert state.is_standard is False
        assert state.mt_ongoing is False
        assert state.mt_past is False


class TestDecodeMemberKw:
    def test_plain_kw_passthrough(self):
        assert decode_member_kw(14) == (14, False)

    def test_common_envelope_offset_is_stripped(self):
        assert decode_member_kw(114) == (14, True)

    def test_boundary_100_is_not_ce(self):
        assert decode_member_kw(100) == (100, False)

    def test_boundary_101_is_ce(self):
        assert decode_member_kw(101) == (1, True)


class TestAnnotateBinaryStates:
    def test_adds_expected_columns(self):
        df = pd.DataFrame(
            {
                "cm_kw": [0, 10, 11, 12, -25, -1],
                "kw_1": [1, 14, 114, 13, 14, 1],
                "kw_2": [1, 1, 1, 113, 14, 1],
            }
        )
        out = annotate_binary_states(df)

        np.testing.assert_array_equal(
            out["mt_ongoing"].to_numpy(), [False, False, True, False, False, False]
        )
        np.testing.assert_array_equal(
            out["mt_past"].to_numpy(), [False, False, False, True, False, False]
        )
        np.testing.assert_array_equal(
            out["is_relativistic_binary"].to_numpy(),
            [False, False, False, False, True, False],
        )
        np.testing.assert_array_equal(
            out["is_chaotic_tide_binary"].to_numpy(),
            [False, False, False, False, False, True],
        )
        np.testing.assert_array_equal(out["base_kw_1"].to_numpy(), [1, 14, 14, 13, 14, 1])
        np.testing.assert_array_equal(
            out["in_ce_1"].to_numpy(), [False, False, True, False, False, False]
        )
        np.testing.assert_array_equal(out["base_kw_2"].to_numpy(), [1, 1, 1, 13, 14, 1])
        np.testing.assert_array_equal(
            out["in_ce_2"].to_numpy(), [False, False, False, True, False, False]
        )

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"cm_kw": [10], "kw_1": [1], "kw_2": [1]})
        annotate_binary_states(df)
        assert list(df.columns) == ["cm_kw", "kw_1", "kw_2"]


class TestBindingEnergyMatchesHdf5Reader:
    def test_matches_hdf5_reader_formula(self):
        # Same formula as hdf5_reader.py's Ebind_abs_NBODY (~L352-362),
        # computed independently here to check binding_energy_nb agrees.
        m1_msun, m2_msun, a_au = 12.0, 3.0, 5.0
        zmbar_msun, rbar_pc = 0.6, 1.3
        pc_to_au = 206264.80624548031

        m1_nb = m1_msun / zmbar_msun
        m2_nb = m2_msun / zmbar_msun
        a_nb = a_au / pc_to_au / rbar_pc
        expected = m1_nb * m2_nb / (2 * a_nb) / (m1_nb + m2_nb)

        result = binding_energy_nb(m1_msun, m2_msun, a_au, zmbar_msun=zmbar_msun, rbar_pc=rbar_pc)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_ebind_over_kt_is_plain_division(self):
        assert ebind_over_kt(4.0, 2.0) == pytest.approx(2.0)

    def test_ebind_over_kt_vectorized(self):
        result = ebind_over_kt(np.array([1.0, 2.0]), np.array([1.0, 4.0]))
        np.testing.assert_allclose(result, [1.0, 0.5])


class TestMeanCoreInterparticleDistance:
    def test_hand_calc_scalar(self):
        pc_to_au = 206264.80624548031
        nc, rc_nb, rbar_pc = 8.0, 1.0, 2.0
        expected_d_nb = (4 * np.pi * rc_nb**3 / (3 * nc)) ** (1 / 3)
        expected_d_au = expected_d_nb * rbar_pc * pc_to_au

        result = mean_core_interparticle_distance_au(nc, rc_nb, rbar_pc=rbar_pc)
        assert result == pytest.approx(expected_d_au, rel=1e-6)

    def test_vectorized(self):
        result = mean_core_interparticle_distance_au(
            np.array([8.0, 64.0]), np.array([1.0, 2.0]), rbar_pc=np.array([2.0, 2.0])
        )
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_nc_zero_gives_nan(self):
        result = mean_core_interparticle_distance_au(0.0, 1.0, rbar_pc=1.0)
        assert np.isnan(result)

    def test_rc_zero_gives_nan(self):
        result = mean_core_interparticle_distance_au(10.0, 0.0, rbar_pc=1.0)
        assert np.isnan(result)

    def test_nan_input_gives_nan(self):
        result = mean_core_interparticle_distance_au(np.nan, 1.0, rbar_pc=1.0)
        assert np.isnan(result)

    def test_negative_gives_nan(self):
        result = mean_core_interparticle_distance_au(-1.0, 1.0, rbar_pc=1.0)
        assert np.isnan(result)


class TestClassifyBinaries:
    def test_hard_priority_over_wide_and_weak(self):
        # bin_label == 1 is hard unconditionally, even though a/ebind look
        # like a textbook "temporary" candidate.
        result = classify_binaries(
            np.array([1]),
            np.array([1000.0]),
            np.array([1e-9]),
            eclose_nb=np.array([1.0]),
            mean_core_distance_au=np.array([10.0]),
        )
        assert result[0] == BINARY_CLASS_HARD

    def test_temporary_needs_both_conditions(self):
        bin_label = np.array([-9, -9, -9, -9])
        # [wide+weak, wide only, weak only, neither]
        a_au = np.array([100.0, 100.0, 1.0, 1.0])
        ebind_nb = np.array([1e-9, 1.0, 1e-9, 1.0])
        eclose_nb = np.full(4, 1.0)
        mean_core_distance_au = np.full(4, 10.0)

        result = classify_binaries(
            bin_label,
            a_au,
            ebind_nb,
            eclose_nb=eclose_nb,
            mean_core_distance_au=mean_core_distance_au,
        )
        np.testing.assert_array_equal(
            result,
            [
                BINARY_CLASS_TEMPORARY,
                BINARY_CLASS_SOFT,
                BINARY_CLASS_SOFT,
                BINARY_CLASS_SOFT,
            ],
        )

    def test_unknown_wide_merger_labels_not_hard(self):
        # -9 (unknown), 0 (wide), -1 (merger-internal) all fall through to
        # the energy/distance test instead of being hard.
        bin_label = np.array([-9, 0, -1])
        a_au = np.full(3, 1.0)  # inside core spacing -> not temporary either
        ebind_nb = np.full(3, 1e-9)
        eclose_nb = np.full(3, 1.0)
        mean_core_distance_au = np.full(3, 10.0)

        result = classify_binaries(
            bin_label,
            a_au,
            ebind_nb,
            eclose_nb=eclose_nb,
            mean_core_distance_au=mean_core_distance_au,
        )
        assert np.all(result == BINARY_CLASS_SOFT)

    def test_eclose_non_positive_or_nan_never_temporary(self):
        bin_label = np.array([-9, -9, -9])
        a_au = np.full(3, 100.0)
        ebind_nb = np.full(3, 1e-9)
        eclose_nb = np.array([0.0, -1.0, np.nan])
        mean_core_distance_au = np.full(3, 10.0)

        result = classify_binaries(
            bin_label,
            a_au,
            ebind_nb,
            eclose_nb=eclose_nb,
            mean_core_distance_au=mean_core_distance_au,
        )
        assert np.all(result == BINARY_CLASS_SOFT)

    def test_distance_nan_never_temporary(self):
        # NC/RC == 0 at t=0 -> mean_core_distance_au is NaN -> never temporary.
        result = classify_binaries(
            np.array([-9]),
            np.array([100.0]),
            np.array([1e-9]),
            eclose_nb=np.array([1.0]),
            mean_core_distance_au=np.array([np.nan]),
        )
        assert result[0] == BINARY_CLASS_SOFT

    def test_a_or_ebind_nan_falls_back_to_soft_unless_hard(self):
        bin_label = np.array([1, -9, -9])
        a_au = np.array([np.nan, np.nan, 100.0])
        ebind_nb = np.array([1e-9, 1e-9, np.nan])
        eclose_nb = np.full(3, 1.0)
        mean_core_distance_au = np.full(3, 10.0)

        result = classify_binaries(
            bin_label,
            a_au,
            ebind_nb,
            eclose_nb=eclose_nb,
            mean_core_distance_au=mean_core_distance_au,
        )
        np.testing.assert_array_equal(
            result, [BINARY_CLASS_HARD, BINARY_CLASS_SOFT, BINARY_CLASS_SOFT]
        )

    def test_empty_arrays(self):
        result = classify_binaries(
            np.array([]),
            np.array([]),
            np.array([]),
            eclose_nb=np.array([]),
            mean_core_distance_au=np.array([]),
        )
        assert result.shape == (0,)


class TestAddBinaryEnergeticsAndClass:
    def _binaries(self):
        return pd.DataFrame(
            {
                "simulation_id": ["20sb", "20sb", "20sb"],
                "ttot": [1.0, 1.0, 2.0],
                "bin_label": [1, -9, -9],
                "mass_1_msun": [10.0, 1.0, 1.0],
                "mass_2_msun": [5.0, 1.0, 1.0],
                "semi_major_axis_au": [1.0, 500.0, 500.0],
            }
        )

    def _scalars(self):
        return pd.DataFrame(
            {
                "simulation_id": ["20sb", "20sb"],
                "ttot": [1.0, 2.0],
                "zmbar_msun": [1.0, 1.0],
                "rbar_pc": [1.0, 1.0],
                "eclose_nb": [1.0, 1.0],
                "rc_nb": [0.01, 0.01],
                "nc": [1000, 1000],
            }
        )

    def test_join_and_columns_added(self):
        binaries = self._binaries()
        scalars = self._scalars()
        out = add_binary_energetics_and_class(binaries, scalars)

        assert len(out) == len(binaries)
        for col in ["ebind_nb", "mean_core_distance_au", "ebind_over_kt", "binary_class"]:
            assert col in out.columns
        assert out.loc[0, "binary_class"] == BINARY_CLASS_HARD

        # Cross-check row 1 against classify_binaries called independently.
        expected_class = classify_binaries(
            out["bin_label"].to_numpy(),
            out["semi_major_axis_au"].to_numpy(dtype=float),
            out["ebind_nb"].to_numpy(dtype=float),
            eclose_nb=out["eclose_nb"].to_numpy(dtype=float),
            mean_core_distance_au=out["mean_core_distance_au"].to_numpy(dtype=float),
        )
        np.testing.assert_array_equal(out["binary_class"].to_numpy(), expected_class)

    def test_missing_scalars_row_becomes_soft(self):
        binaries = self._binaries()
        binaries.loc[2, "ttot"] = 999.0  # no matching scalars row
        scalars = self._scalars()

        out = add_binary_energetics_and_class(binaries, scalars)
        assert out.loc[2, "binary_class"] == BINARY_CLASS_SOFT
        assert np.isnan(out.loc[2, "ebind_over_kt"])

    def test_does_not_mutate_input(self):
        binaries = self._binaries()
        scalars = self._scalars()
        original_columns = list(binaries.columns)

        add_binary_energetics_and_class(binaries, scalars)
        assert list(binaries.columns) == original_columns


class TestDropTemporaryBinaries:
    def test_drops_temporary_rows(self):
        df = pd.DataFrame(
            {
                "binary_class": [
                    BINARY_CLASS_HARD,
                    BINARY_CLASS_TEMPORARY,
                    BINARY_CLASS_SOFT,
                ]
            }
        )
        out = drop_temporary_binaries(df)
        assert list(out["binary_class"]) == [BINARY_CLASS_HARD, BINARY_CLASS_SOFT]

    def test_keeps_rows_with_nan_class(self):
        df = pd.DataFrame({"binary_class": [BINARY_CLASS_TEMPORARY, np.nan]})
        out = drop_temporary_binaries(df)
        assert len(out) == 1
        assert out["binary_class"].isna().iloc[0]

    def test_missing_column_returns_copy(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        out = drop_temporary_binaries(df)
        assert len(out) == 3
        out["a"] = 0
        assert list(df["a"]) == [1, 2, 3]
