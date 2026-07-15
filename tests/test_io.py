"""Tests for nbody_pipeline.io module"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from nbody_pipeline.io import (
    get_scale_dict_from_hdf5_df,
    read_bwdat,
    read_bdat,
    read_coll_13,
    read_coal_24,
    make_l7header,
    l7df_to_physical_units,
    transform_l7df_to_sns_friendly,
    get_valueStr_of_namelist_key,
    decode_bytes_columns_inplace,
    tau_gw,
    merge_multiple_hdf5_dataframes,
    HDF5FileProcessor,
    LagrFileProcessor,
    Coll13FileProcessor,
    Coal24FileProcessor,
)


class TestScaleDictFunctions:
    """Test scale dictionary extraction functions"""

    def test_get_scale_dict_from_hdf5_df(self):
        """Test extracting scale dict from HDF5 scalar dataframe"""
        scalar_df = pd.DataFrame({"RBAR": [1.5], "VSTAR": [2.0], "ZMBAR": [0.5], "TSCALE": [10.0]})

        scale_dict = get_scale_dict_from_hdf5_df(scalar_df)

        assert scale_dict["r"] == 1.5
        assert scale_dict["v"] == 2.0
        assert scale_dict["m"] == 0.5
        assert scale_dict["t"] == 10.0


class TestTextParsers:
    """Test text file parsing functions"""

    def test_read_bwdat(self, temp_dir):
        """Test reading bwdat file"""
        bwdat_file = temp_dir / "test.bwdat"
        bwdat_file.write_text("# Header\nA B C\n1 2 3\n4 5 6\n")

        df = read_bwdat(str(bwdat_file))
        assert len(df) == 2
        assert list(df.columns) == ["A", "B", "C"]
        assert df["A"].tolist() == [1, 4]

    def test_read_bdat(self, temp_dir):
        """Test reading bdat file"""
        bdat_file = temp_dir / "test.bdat"
        bdat_file.write_text("A B C\n1 2 3\n4 5 6\n")

        df = read_bdat(str(bdat_file))
        assert len(df) == 2
        assert list(df.columns) == ["A", "B", "C"]

    def test_read_coll_13_mixed_headered_and_headerless_chunks(self, temp_dir):
        """Test coll.13 parser with concatenated headered and headerless chunks"""
        coll_file = temp_dir / "coll.13"
        coll_file.write_text("""
      MODEL:    RBAR = 2.0 <M>[M*] = 0.5

                  TIME[NB]    NAME(I1)    NAME(I2)      K*(I1)      K*(I2)    K*(INEW)                 M(I1)[M*]                 M(I2)[M*]               M(INEW)[M*]                    DM[M*]                RS(I1)[R*]                RS(I2)[R*]                    RI[RC]                   R12[R*]                       ECC                   P[days]

   2.1484375000000000E-002          70          69           0           0           0   8.0011524700507985E-002   8.0011152204026201E-002  0.16002267690453420        0.0000000000000000       0.15339321320082239       0.15339345124407011        1.0702293519063384       0.22556537802705393       0.81202286764368137       0.38044884885241465
   2913.6932184696198            95325      119152          14           1          14   40.500000000000000        2.7098000488923106        41.854900024446152        1.3549000244461553        1.7171999999999999E-004   1.5680612599913319       0.49323879099146639        1.8117363135256588        1.0001790545607305       -17927.064589961425
""")

        df = read_coll_13(str(coll_file))

        assert len(df) == 2
        assert df["TIME[NB]"].tolist() == pytest.approx([0.021484375, 2913.6932184696198])
        assert df["NAME(I1)"].tolist() == [70, 95325]
        assert df["K*(I1)"].tolist() == [0, 14]
        assert df["M(I1)[M*]"].tolist() == pytest.approx([0.08001152470050799, 40.5])

    def test_read_coal_24_old_and_compact_formats(self, temp_dir):
        """Test coal.24 parser with old headered rows and compact BINARY/ROCHE rows"""
        coal_file = temp_dir / "coal.24"
        coal_file.write_text("""
          TIME[NB]           NAME(I1)    NAME(I2)     K*(I1)      K*(I2)       K*1       IQCOLL             M(I1)[M*]                 M(I2)[M*]                M(INEW)[M*]                DM[M*]                    RS(I1)[R*]                RS(I2)[R*]                 RI/RC                     R12[R*]                   ECC                       P[days]                   RCOLL[R*]                  EB[NB]                    DP[NB]                    VINF[km/s]
   2.1484375000000000E-002        2637        2638           0           0           0           3  0.11702739289326970       0.10764211228603181       0.11702740805664708       0.10764209712265442       0.13525384536076443        0.0000000000000000        1.3900859407360744       0.35278321333321200       -1.9974978751822192E-003   5.1030022138430192E-002  0.35278321343986480       -5.1039197813355911E-006  -6.7762635780344027E-021   0.0000000000000000
 BINARY   0 2.8827246094E+03     93127     93128   5   0   5  4.27669E+00  0.00000E+00  3.72164E-06  1.00000E-03  3.72537E-06 -8.38777E-07  5.42101E-20  3.81237E+02  3.71806E+00  5.68591E-01  4.27669E+00  9.95515E-03  2.55601E+02  0.00000E+00  3.59179E+02  0.00000E+00  3.33137E+00  2.69375E-01
""")

        df = read_coal_24(str(coal_file))

        assert len(df) == 2
        assert df["TIME[NB]"].tolist() == pytest.approx([0.021484375, 2882.7246094])
        assert df["NAME(I1)"].tolist() == [2637, 93127]
        assert df["K*(I1)"].tolist() == [0, 5]
        assert df["M(I1)[M*]"].tolist() == pytest.approx([0.1170273928932697, 3.71806])
        assert pd.isna(df["Coalescence_trigger"].iloc[0])
        assert df["Coalescence_trigger"].iloc[1] == "BINARY"

    def test_make_l7header(self):
        """Test lagr.7 header generation"""
        header = make_l7header()
        assert isinstance(header, list)
        assert header[0] == "Time[NB]"
        assert len(header) == 284

    def test_get_valueStr_of_namelist_key(self, temp_dir):
        """Test extracting value from namelist file"""
        namelist_file = temp_dir / "test.inp"
        namelist_file.write_text("""
        &INPUT
        N = 1000000
        ALPHA = 0.5
        BETA = 2.3e-4
        &END
        """)

        n_value = get_valueStr_of_namelist_key(str(namelist_file), "N")
        assert n_value == "1000000"

        alpha_value = get_valueStr_of_namelist_key(str(namelist_file), "ALPHA")
        assert alpha_value == "0.5"

        beta_value = get_valueStr_of_namelist_key(str(namelist_file), "BETA")
        assert beta_value == "2.3e-4"

        # Test key not found
        with pytest.raises(KeyError):
            get_valueStr_of_namelist_key(str(namelist_file), "NOTEXIST")

    def test_decode_bytes_columns_inplace(self):
        """Test decoding bytes columns in DataFrame"""
        df = pd.DataFrame({"name": [b"star1  ", b"star2  "], "value": [1.0, 2.0]})

        decode_bytes_columns_inplace(df)

        assert df["name"].tolist() == ["star1", "star2"]
        assert df["value"].tolist() == [1.0, 2.0]


class TestLagrFunctions:
    """Test lagr.7 file processing functions"""

    def test_l7df_to_physical_units(self):
        """Test converting lagr.7 dataframe to physical units"""
        df = pd.DataFrame(
            {
                "Time[NB]": [0.0, 1.0, 2.0],
                "rlagr1.00E-03": [0.1, 0.2, 0.3],
                "vx1.00E-03": [1.0, 2.0, 3.0],
                "sigma21.00E-03": [0.5, 1.0, 1.5],
                "nshell1.00E-03": [100, 200, 300],
            }
        )

        scale_dict = {"r": 2.0, "v": 3.0, "t": 10.0, "m": 0.5}

        converted_df = l7df_to_physical_units(df, scale_dict)

        assert "Time[Myr]" in converted_df.columns
        assert "Time[NB]" not in converted_df.columns
        assert converted_df["Time[Myr]"].tolist() == [0.0, 10.0, 20.0]
        assert converted_df["rlagr1.00E-03"].tolist() == [0.2, 0.4, 0.6]
        assert converted_df["vx1.00E-03"].tolist() == [3.0, 6.0, 9.0]
        assert converted_df["sigma21.00E-03"].tolist() == [4.5, 9.0, 13.5]
        assert converted_df["nshell1.00E-03"].tolist() == [100, 200, 300]

    def test_transform_l7df_to_sns_friendly(self):
        """Test transforming lagr.7 dataframe to seaborn-friendly format"""
        df = pd.DataFrame(
            {
                "Time[Myr]": [0.0, 1.0],
                "rlagr1.00E-03": [0.1, 0.2],
                "rlagr3.00E-03": [0.15, 0.25],
                "vx1.00E-03": [1.0, 2.0],
            }
        )

        sns_df = transform_l7df_to_sns_friendly(df)

        assert "Time[Myr]" in sns_df.columns
        assert "Percentage" in sns_df.columns
        assert "Metric" in sns_df.columns
        assert "Value" in sns_df.columns
        assert "%" in sns_df.columns

        assert len(sns_df) == 6

        assert set(sns_df["Metric"].unique()) == {"rlagr", "vx"}
        assert "0.1%" in sns_df["%"].values
        assert "0.3%" in sns_df["%"].values

    def test_build_total_mass_df_uses_100_percent_and_keeps_zero_time(self):
        """Total mass uses only the 100% shell and keeps Time[Myr] == 0.0."""
        lagr_df = pd.DataFrame(
            [
                {"Time[Myr]": 0.0, "%": "100%", "Metric": "avmass", "Value": 2.0},
                {"Time[Myr]": 1.0, "%": "100%", "Metric": "avmass", "Value": 3.0},
                {"Time[Myr]": 0.0, "%": "100%", "Metric": "nshell", "Value": 10.0},
                {"Time[Myr]": 1.0, "%": "100%", "Metric": "nshell", "Value": 20.0},
                {"Time[Myr]": 0.0, "%": "90%", "Metric": "avmass", "Value": 100.0},
                {"Time[Myr]": 0.0, "%": "90%", "Metric": "nshell", "Value": 100.0},
            ]
        )

        total_mass_df = LagrFileProcessor.build_total_mass_df(lagr_df)

        assert total_mass_df["Time[Myr]"].tolist() == [0.0, 1.0]
        assert total_mass_df["Value_avmass"].tolist() == [2.0, 3.0]
        assert total_mass_df["Value_nshell"].tolist() == [10.0, 20.0]
        assert total_mass_df["total_mass"].tolist() == [20.0, 60.0]


class TestTauGW:
    """Test gravitational wave merger timescale function"""

    def test_tau_gw_float_inputs(self):
        """Test tau_gw with float inputs"""

        a = 1e9
        e = 0.5
        mu = 1e30
        M = 2e30

        tau = tau_gw(a, e, mu, M)

        assert tau > 0
        assert isinstance(tau, float)

    def test_tau_gw_astropy_units(self):
        """Test tau_gw with astropy Quantity inputs"""
        import astropy.units as u

        a = 1.0 * u.au
        e = 0.1
        mu = 30.0 * u.solMass
        M = 60.0 * u.solMass

        tau = tau_gw(a, e, mu, M)

        assert hasattr(tau, "unit")
        assert tau.value > 0


class TestHDF5Functions:
    """Test HDF5 file processing functions"""

    def test_merge_multiple_hdf5_dataframes(self):
        """Test merging multiple HDF5 dataframes"""
        df1_dict = {
            "scalars": pd.DataFrame({"TTOT": [0.0, 1.0], "N": [1000, 999]}),
            "singles": pd.DataFrame({"TTOT": [0.0], "M": [1.0]}),
            "binaries": None,
            "mergers": None,
        }

        df2_dict = {
            "scalars": pd.DataFrame({"TTOT": [2.0, 3.0], "N": [998, 997]}),
            "singles": pd.DataFrame({"TTOT": [2.0], "M": [1.1]}),
            "binaries": pd.DataFrame({"TTOT": [2.0], "Bin M1*": [2.0]}),
            "mergers": None,
        }

        merged = merge_multiple_hdf5_dataframes([df1_dict, df2_dict])

        assert merged["scalars"] is not None
        assert len(merged["scalars"]) == 4
        assert merged["singles"] is not None
        assert len(merged["singles"]) == 2
        assert merged["binaries"] is not None
        assert len(merged["binaries"]) == 1
        assert merged["mergers"] is None


class TestHDF5FileProcessor:
    """Test HDF5FileProcessor class"""

    def test_get_hdf5_file_time_from_filename(self):
        """Test extracting time from HDF5 filename"""
        config_mock = Mock()
        processor = HDF5FileProcessor(config_mock)

        path = "/path/to/snap.40_1.234.h5part"
        time = processor.get_hdf5_file_time_from_filename(path)

        assert time == 1.234

    def test_get_compact_object_mask_singles(self):
        """Test getting compact object mask for singles"""
        config_mock = Mock()
        config_mock.compact_object_KW = [13, 14]
        processor = HDF5FileProcessor(config_mock)

        df = pd.DataFrame({"KW": [0, 1, 13, 14, 5]})

        mask = processor.get_compact_object_mask(df)

        assert mask.tolist() == [False, False, True, True, False]

    def test_get_compact_object_mask_binaries(self):
        """Test getting compact object mask for binaries"""
        config_mock = Mock()
        config_mock.compact_object_KW = [13, 14]
        processor = HDF5FileProcessor(config_mock)

        df = pd.DataFrame({"Bin KW1": [0, 13, 1], "Bin KW2": [1, 2, 14]})

        mask = processor.get_compact_object_mask(df)

        assert mask.tolist() == [False, True, True]

    def _make_scalars_mock(self):
        return pd.DataFrame(
            {
                "TTOT": [1.0],
                "TTOT/TCR0": [1.0],
                "RBAR": [1.0],
                "ZMBAR": [1.0],
                "TSCALE": [1.0],
                "VSTAR": [1.0],
                "RDENS(1)": [0.0],
                "RDENS(2)": [0.0],
                "RDENS(3)": [0.0],
                "ECLOSE": [1.0],
                "NC": [1000],
                "RC": [0.01],
            }
        ).set_index("TTOT", drop=False)

    def _make_binaries_mock_three_classes(self):
        # One row per class, hand-picked to be unambiguous under
        # TEMPORARY_EBIND_FACTOR=1e-3 and the RC=0.01/NC=1000/RBAR=1.0 mean
        # core spacing above (~332.5 au):
        # hard: Bin Label==1 wins regardless of a/Ebind.
        # soft: Bin Label==0 (wide), a=1.0 au < core spacing -> never temporary.
        # temporary: Bin Label==-9 (unknown), a=1000 au > core spacing, and
        # tiny masses push Ebind well below 1e-3*ECLOSE.
        return pd.DataFrame(
            {
                "TTOT": [1.0, 1.0, 1.0],
                "Bin cm X1": [0.0, 0.0, 0.0],
                "Bin cm X2": [0.0, 0.0, 0.0],
                "Bin cm X3": [0.0, 0.0, 0.0],
                "Bin M1*": [10.0, 1.0, 1e-5],
                "Bin M2*": [5.0, 1.0, 1e-5],
                "Bin KW1": [0, 0, 0],
                "Bin KW2": [0, 0, 0],
                "Bin A[au]": [1.0, 1.0, 1000.0],
                "Bin ECC": [0.1, 0.1, 0.1],
                "Bin RS1*": [0.0, 0.0, 0.0],
                "Bin RS2*": [0.0, 0.0, 0.0],
                "Bin Label": [1, 0, -9],
            }
        )

    def test_read_file_can_read_cache_without_writing_missing_cache(self):
        """use_cache=True can check Feather cache without creating it."""
        config_mock = Mock()
        config_mock.input_file_path_of = {"test_simu": "/fake/input"}
        config_mock.kw_to_stellar_type_verbose = {0: "0:MS"}
        config_mock.kw_to_stellar_type = {0: "MS"}
        config_mock.limits = {"L*": [1.0e-6, 1.0e6], "Teff*": [1000.0, 1.0e6]}
        config_mock.universe_age_myr = 13800.0
        processor = HDF5FileProcessor(config_mock)
        df_dict = {
            "scalars": self._make_scalars_mock(),
            "singles": pd.DataFrame(
                {
                    "TTOT": [1.0],
                    "X1": [0.0],
                    "X2": [0.0],
                    "X3": [0.0],
                    "V1": [0.0],
                    "V2": [0.0],
                    "V3": [0.0],
                    "M": [1.0],
                    "KW": [0],
                    "L*": [1.0],
                    "Teff*": [5000.0],
                }
            ),
            "binaries": self._make_binaries_mock_three_classes(),
            "mergers": pd.DataFrame(),
        }

        with (
            patch.object(processor, "_cache_is_complete", return_value=False),
            patch.object(processor, "_write_df_dict_to_cache") as mock_write,
            patch(
                "nbody_pipeline.io.text_parsers.dataframes_from_hdf5_file",
                return_value=df_dict,
            ),
            patch(
                "nbody_pipeline.io.text_parsers.get_valueStr_of_namelist_key",
                return_value="10",
            ),
        ):
            result = processor.read_file(
                "/fake/snap.40_1.h5part",
                "test_simu",
                use_cache=True,
                write_cache=False,
            )

        assert result["scalars"]["Time[Myr]"].tolist() == [1.0]
        mock_write.assert_not_called()

        binaries = result["binaries"]
        assert list(binaries["binary_class"]) == ["hard", "soft", "temporary"]
        assert list(binaries["is_hard_binary"]) == [True, False, False]
        # New denominator: Ebind_abs_NBODY / (1e-3 * ECLOSE), ECLOSE=1.0 here.
        expected_ebind_over_kt = binaries["Ebind_abs_NBODY"] / 1.0e-3
        pd.testing.assert_series_equal(
            binaries["Ebind/kT"], expected_ebind_over_kt, check_names=False
        )
        assert "mean_core_interparticle_distance[au]" in binaries.columns
        assert (binaries["mean_core_interparticle_distance[au]"] > 0).all()
        assert (binaries["eclose_nb"] == 1.0).all()

    def test_read_file_missing_scalar_columns_falls_back_to_nan(self):
        """ECLOSE/NC/RC missing from scalars (old archived files/minimal mocks)
        -> binary_class/Ebind/kT columns degrade to NaN/soft instead of raising."""
        config_mock = Mock()
        config_mock.input_file_path_of = {"test_simu": "/fake/input"}
        config_mock.kw_to_stellar_type_verbose = {0: "0:MS"}
        config_mock.kw_to_stellar_type = {0: "MS"}
        config_mock.limits = {"L*": [1.0e-6, 1.0e6], "Teff*": [1000.0, 1.0e6]}
        config_mock.universe_age_myr = 13800.0
        processor = HDF5FileProcessor(config_mock)
        scalars = self._make_scalars_mock().drop(columns=["ECLOSE", "NC", "RC"])
        binaries = self._make_binaries_mock_three_classes().drop(columns=["Bin Label"])
        df_dict = {
            "scalars": scalars,
            "singles": pd.DataFrame(
                {
                    "TTOT": [1.0],
                    "X1": [0.0],
                    "X2": [0.0],
                    "X3": [0.0],
                    "V1": [0.0],
                    "V2": [0.0],
                    "V3": [0.0],
                    "M": [1.0],
                    "KW": [0],
                    "L*": [1.0],
                    "Teff*": [5000.0],
                }
            ),
            "binaries": binaries,
            "mergers": pd.DataFrame(),
        }

        with (
            patch.object(processor, "_cache_is_complete", return_value=False),
            patch(
                "nbody_pipeline.io.text_parsers.dataframes_from_hdf5_file",
                return_value=df_dict,
            ),
            patch(
                "nbody_pipeline.io.text_parsers.get_valueStr_of_namelist_key",
                return_value="10",
            ),
        ):
            result = processor.read_file(
                "/fake/snap.40_1.h5part", "test_simu", use_cache=False, write_cache=False
            )

        binaries_out = result["binaries"]
        # No "Bin Label" column -> sentinel -9 for every row -> never hard.
        assert (binaries_out["binary_class"] != "hard").all()
        assert binaries_out["Ebind/kT"].isna().all()
        assert binaries_out["mean_core_interparticle_distance[au]"].isna().all()
        assert binaries_out["eclose_nb"].isna().all()

    def test_binaries_cache_is_current_true_when_binary_class_present(self, temp_dir):
        processor = HDF5FileProcessor(Mock())
        path = str(temp_dir / "current.binaries.df.feather")
        pd.DataFrame({"TTOT": [1.0], "binary_class": ["hard"]}).to_feather(path)
        assert processor._binaries_cache_is_current(path) is True

    def test_binaries_cache_is_current_false_for_stale_nonempty(self, temp_dir):
        processor = HDF5FileProcessor(Mock())
        path = str(temp_dir / "stale.binaries.df.feather")
        pd.DataFrame({"TTOT": [1.0], "Ebind/kT": [2.0]}).to_feather(path)
        assert processor._binaries_cache_is_current(path) is False

    def test_binaries_cache_is_current_true_for_empty_placeholder(self, temp_dir):
        """An old zero-row placeholder cache has nothing stale, so counts as current."""
        processor = HDF5FileProcessor(Mock())
        path = str(temp_dir / "empty.binaries.df.feather")
        pd.DataFrame().to_feather(path)
        assert processor._binaries_cache_is_current(path) is True

    def test_read_file_self_heals_stale_binaries_cache(self, temp_dir):
        """use_cache=True re-reads HDF5 and rewrites the cache when the cached
        binaries feather predates binary_class, instead of returning stale data."""
        config_mock = Mock()
        config_mock.input_file_path_of = {"test_simu": "/fake/input"}
        config_mock.kw_to_stellar_type_verbose = {0: "0:MS"}
        config_mock.kw_to_stellar_type = {0: "MS"}
        config_mock.limits = {"L*": [1.0e-6, 1.0e6], "Teff*": [1000.0, 1.0e6]}
        config_mock.universe_age_myr = 13800.0
        processor = HDF5FileProcessor(config_mock)

        hdf5_path = str(temp_dir / "snap.40_1.h5part")
        feather_path_of = processor._get_feather_path_of(hdf5_path)
        self._make_scalars_mock().reset_index(drop=True).to_feather(feather_path_of["scalars"])
        pd.DataFrame({"TTOT": [1.0], "X1": [0.0]}).to_feather(feather_path_of["singles"])
        # Stale: pre-reclassification schema, no binary_class column.
        pd.DataFrame({"TTOT": [1.0], "Ebind/kT": [2.0]}).to_feather(feather_path_of["binaries"])
        pd.DataFrame().to_feather(feather_path_of["mergers"])

        df_dict = {
            "scalars": self._make_scalars_mock(),
            "singles": pd.DataFrame(
                {
                    "TTOT": [1.0],
                    "X1": [0.0],
                    "X2": [0.0],
                    "X3": [0.0],
                    "V1": [0.0],
                    "V2": [0.0],
                    "V3": [0.0],
                    "M": [1.0],
                    "KW": [0],
                    "L*": [1.0],
                    "Teff*": [5000.0],
                }
            ),
            "binaries": self._make_binaries_mock_three_classes(),
            "mergers": pd.DataFrame(),
        }

        with (
            patch(
                "nbody_pipeline.io.text_parsers.dataframes_from_hdf5_file",
                return_value=df_dict,
            ),
            patch(
                "nbody_pipeline.io.text_parsers.get_valueStr_of_namelist_key",
                return_value="10",
            ),
        ):
            result = processor.read_file(hdf5_path, "test_simu", use_cache=True, write_cache=True)

        assert "binary_class" in result["binaries"].columns
        # Self-healed: the cache on disk now has the new schema too.
        assert processor._binaries_cache_is_current(feather_path_of["binaries"]) is True

    def test_read_tables_falls_back_when_binaries_cache_stale(self, temp_dir):
        """Column-projected read_tables reads never request binary_class directly,
        so staleness must be probed independent of columns_by_table."""
        processor = HDF5FileProcessor(Mock())
        hdf5_path = str(temp_dir / "snap.40_2.h5part")
        feather_path_of = processor._get_feather_path_of(hdf5_path)
        pd.DataFrame({"TTOT": [1.0], "Ebind/kT": [2.0]}).to_feather(feather_path_of["binaries"])

        fresh_binaries = pd.DataFrame({"TTOT": [1.0], "Bin A[au]": [1.0], "binary_class": ["hard"]})
        with patch.object(
            processor, "read_file", return_value={"binaries": fresh_binaries}
        ) as mock_read_file:
            result = processor.read_tables(
                hdf5_path,
                "test_simu",
                tables=("binaries",),
                columns_by_table={"binaries": ["Bin A[au]"]},
                use_cache=True,
            )

        mock_read_file.assert_called_once()
        assert "Bin A[au]" in result["binaries"].columns

    def test_get_all_hdf5_paths_dedups_same_index_keeps_larger_file(self, temp_dir, monkeypatch):
        """Two different physical files can share the same filename-derived index
        (e.g. a stale archived copy re-generated later under a different directory
        with the same "snap.40_N.h5part" name, confirmed for 20sb's snap.40_0.h5part).
        Without dedup both stay in the list with a tied sort key, so which one
        downstream code picks depends on arbitrary glob() order. The larger file
        should always win."""
        small_dir = temp_dir / "archive"
        large_dir = temp_dir / "snap.40"
        small_dir.mkdir()
        large_dir.mkdir()
        small_file = small_dir / "snap.40_0.h5part"
        large_file = large_dir / "snap.40_0.h5part"
        small_file.write_bytes(b"0" * 10)
        large_file.write_bytes(b"0" * 100)
        other_file = temp_dir / "snap.40_1.h5part"
        other_file.write_bytes(b"0" * 50)

        monkeypatch.setattr("nbody_pipeline.io.hdf5_reader.os.path.getmtime", lambda p: 0.0)
        monkeypatch.setattr("nbody_pipeline.io.hdf5_reader.time.time", lambda: 1e12)

        config_mock = Mock()
        config_mock.pathof = {"test_simu": str(temp_dir)}
        processor = HDF5FileProcessor(config_mock)

        paths = processor.get_all_hdf5_paths(
            "test_simu", wait_age_hour=0, sample_every_nb_time=None, exclude_bad_dirname=False
        )

        assert str(large_file) in paths
        assert str(small_file) not in paths
        assert str(other_file) in paths
        assert len(paths) == 2


class TestLagrFileProcessor:
    """Test LagrFileProcessor class"""

    def test_initialization(self):
        """Test LagrFileProcessor initialization"""
        config_mock = Mock()
        processor = LagrFileProcessor(config_mock)

        assert processor.file_basename == "lagr.7"
        assert processor.config == config_mock


class TestCollisionProcessors:
    """Test Coll13FileProcessor and Coal24FileProcessor"""

    def test_coll13_initialization(self):
        """Test Coll13FileProcessor initialization"""
        config_mock = Mock()
        processor = Coll13FileProcessor(config_mock)

        assert processor.file_basename == "coll.13"

    def test_coal24_initialization(self):
        """Test Coal24FileProcessor initialization"""
        config_mock = Mock()
        processor = Coal24FileProcessor(config_mock)

        assert processor.file_basename == "coal.24"

    def test_merge_coll_coal(self):
        """Test merging collision and coalescence dataframes"""
        config_mock = Mock()
        processor = Coll13FileProcessor(config_mock)

        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df2 = pd.DataFrame({"A": [5, 6], "C": [7, 8]})

        merged = processor.merge_coll_coal(df1, df2)

        assert len(merged) == 4
        assert set(merged.columns) == {"A", "B", "C"}
        assert merged["A"].tolist() == [1, 2, 5, 6]


class TestIntegration:
    """Integration tests for I/O module"""

    def test_import_all_exports(self):
        """Test that all exports can be imported"""
        from nbody_pipeline.io import (
            HDF5FileProcessor,
            LagrFileProcessor,
            Coll13FileProcessor,
            Coal24FileProcessor,
        )

        assert HDF5FileProcessor is not None
        assert LagrFileProcessor is not None
        assert Coll13FileProcessor is not None
        assert Coal24FileProcessor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
