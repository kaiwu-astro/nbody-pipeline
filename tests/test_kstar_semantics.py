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
from nbody_pipeline.analysis.physics import binding_energy_nb, ebind_over_kt


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

    def test_negative_other_than_minus_25_is_not_relativistic(self):
        state = decode_cm_kstar(-1)
        assert state.is_relativistic is False
        assert state.is_standard is False


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
                "cm_kw": [0, 10, 11, 12, -25],
                "kw_1": [1, 14, 114, 13, 14],
                "kw_2": [1, 1, 1, 113, 14],
            }
        )
        out = annotate_binary_states(df)

        np.testing.assert_array_equal(
            out["mt_ongoing"].to_numpy(), [False, False, True, False, False]
        )
        np.testing.assert_array_equal(out["mt_past"].to_numpy(), [False, False, False, True, False])
        np.testing.assert_array_equal(
            out["is_relativistic_binary"].to_numpy(), [False, False, False, False, True]
        )
        np.testing.assert_array_equal(out["base_kw_1"].to_numpy(), [1, 14, 14, 13, 14])
        np.testing.assert_array_equal(out["in_ce_1"].to_numpy(), [False, False, True, False, False])
        np.testing.assert_array_equal(out["base_kw_2"].to_numpy(), [1, 1, 1, 13, 14])
        np.testing.assert_array_equal(out["in_ce_2"].to_numpy(), [False, False, False, True, False])

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
