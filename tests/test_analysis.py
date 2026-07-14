"""Tests for analysis module"""

import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from nbody_pipeline.__main__ import SimulationPlotter
from nbody_pipeline.analysis import (
    CompactBinaryCounter,
    CurrentMassLagrangianProcessor,
    GalacticEnergyAngularMomentumProcessor,
    GalacticOrbitProcessor,
    ParticleTracker,
    tau_gw,
)
from nbody_pipeline.analysis.galactic_energy_angular_momentum import (
    E_GAL_COL,
    E_GAL_SPECIFIC_COL,
    E_KIN_GAL_COL,
    E_KIN_GAL_SPECIFIC_COL,
    E_POT_GAL_COL,
    E_POT_GAL_SPECIFIC_COL,
    L_Z_GAL_COL,
    L_Z_GAL_SPECIFIC_COL,
)


class TestParticleTracker:
    """Tests for ParticleTracker class"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration manager"""
        config = Mock()
        config.particle_df_cache_dir_of = {"test_simu": tempfile.mkdtemp()}
        config.pathof = {"test_simu": "/path/to/test"}
        config.processes_count = 2
        config.tasks_per_child = 10
        config.mem_cap_bytes = 40 * 1024**3 // 2  # 20 GB
        config.inode_limit = 2000000
        return config

    @pytest.fixture
    def particle_tracker(self, mock_config):
        """Create a ParticleTracker instance"""
        return ParticleTracker(mock_config)

    @pytest.fixture
    def sample_df_dict(self):
        """Create sample data for testing"""
        singles_df = pd.DataFrame(
            {
                "Name": [1000, 1000, 2000],
                "TTOT": [0.0, 1.0, 0.0],
                "Time[Myr]": [0.0, 10.0, 0.0],
                "M": [10.0, 9.5, 15.0],
                "KW": [1, 1, 2],
                "X1": [0.1, 0.2, 0.3],
                "X2": [0.1, 0.2, 0.3],
                "X3": [0.1, 0.2, 0.3],
            }
        )

        binaries_df = pd.DataFrame(
            {
                "Bin Name1": [1000, 3000],
                "Bin Name2": [2000, 4000],
                "TTOT": [2.0, 0.0],
                "Time[Myr]": [20.0, 0.0],
                "Bin M1*": [9.0, 20.0],
                "Bin M2*": [14.5, 25.0],
            }
        )

        scalars_df = pd.DataFrame(
            {
                "TTOT": [0.0, 1.0, 2.0],
                "RBAR": [1.0, 1.1, 1.2],
            }
        )

        return {
            "singles": singles_df,
            "binaries": binaries_df,
            "scalars": scalars_df,
        }

    def test_init(self, mock_config):
        """Test ParticleTracker initialization"""
        tracker = ParticleTracker(mock_config)
        assert tracker.config == mock_config
        assert tracker.hdf5_file_processor is not None

    def test_get_particle_df_from_hdf5_file_single_only(self, particle_tracker, sample_df_dict):
        """Test tracking a particle that remains single"""
        result = particle_tracker.get_particle_df_from_hdf5_file(sample_df_dict, 2000)

        # Particle 2000 appears in singles at TTOT=0.0 and as Bin Name2 at TTOT=2.0
        # So we get 2 rows after merge
        assert len(result) == 2
        # The single record should have state 'single' where Name is not NaN
        single_row = result[result["Name"].notna()]
        assert len(single_row) == 1
        assert single_row["Name"].iloc[0] == 2000

    def test_get_particle_df_from_hdf5_file_single_and_binary(
        self, particle_tracker, sample_df_dict
    ):
        """Test tracking a particle that goes from single to binary"""
        result = particle_tracker.get_particle_df_from_hdf5_file(sample_df_dict, 1000)

        # Should have 3 rows: TTOT 0.0, 1.0, 2.0
        assert len(result) == 3
        assert sorted(result["TTOT"].tolist()) == [0.0, 1.0, 2.0]

        # Check binary state (TTOT=2.0)
        binary_rows = result[result["state"] == "binary"]
        assert len(binary_rows) == 1
        assert binary_rows["companion_name"].iloc[0] == 2000

    def test_get_particle_df_from_hdf5_file_not_found(self, particle_tracker, sample_df_dict):
        """Test tracking a non-existent particle"""
        result = particle_tracker.get_particle_df_from_hdf5_file(sample_df_dict, 99999)

        assert result.empty

    def test_get_particle_df_from_hdf5_file_both_binary_members(self, particle_tracker):
        """Test graceful handling and deduplication when particle appears as both binary members"""
        # Create data where particle is both Name1 and Name2
        test_df_dict = {
            "singles": pd.DataFrame(
                {
                    "Name": [1000],
                    "TTOT": [0.0],
                }
            ),
            "binaries": pd.DataFrame(
                {
                    "Bin Name1": [1000, 1000],
                    "Bin Name2": [2000, 1000],  # 1000 appears as Name2 here too!
                    "TTOT": [0.0, 1.0],
                }
            ),
            "scalars": pd.DataFrame({"TTOT": [0.0, 1.0]}),
        }

        # Should not raise error, but should log warning and deduplicate
        result = particle_tracker.get_particle_df_from_hdf5_file(test_df_dict, 1000)

        # Should have deduplicated the TTOT=1.0 entry
        assert not result.empty
        assert result["TTOT"].is_unique

    def test_get_particle_summary_empty(self, particle_tracker):
        """Test summary with empty DataFrame"""
        result = particle_tracker.get_particle_summary(pd.DataFrame())
        assert result == {}

    def test_get_particle_summary_complete(self, particle_tracker):
        """Test summary with complete particle history"""
        particle_df = pd.DataFrame(
            {
                "Name": [1000, 1000, 1000],
                "TTOT": [0.0, 1.0, 2.0],
                "Time[Myr]": [0.0, 10.0, 20.0],
                "M": [10.0, 9.5, 9.0],
                "KW": [1, 1, 2],
                "state": ["single", "single", "binary"],
            }
        )

        result = particle_tracker.get_particle_summary(particle_df)

        assert result["particle_name"] == 1000
        assert result["total_snapshots"] == 3
        assert result["time_range_myr"] == (0.0, 20.0)
        assert result["single_count"] == 2
        assert result["binary_count"] == 1
        assert result["initial_mass"] == 10.0
        assert result["final_mass"] == 9.0
        assert result["stellar_types"] == [1, 2]

    @patch("nbody_pipeline.analysis.particle_tracker.glob")
    @patch("os.path.exists")
    def test_update_one_particle_history_df_no_cache(
        self, mock_exists, mock_glob, particle_tracker, mock_config
    ):
        """Test getting particle data with no existing cache"""
        mock_exists.return_value = False
        mock_glob.return_value = []

        result = particle_tracker.update_one_particle_history_df("test_simu", 1000, update=False)

        assert result.empty

    @patch("nbody_pipeline.analysis.particle_tracker.glob")
    @patch("pandas.read_feather")
    @patch("os.path.exists")
    def test_update_one_particle_history_df_with_cache(
        self, mock_exists, mock_read_feather, mock_glob, particle_tracker
    ):
        """Test getting particle data with existing cache"""
        cached_df = pd.DataFrame(
            {
                "Name": [1000],
                "TTOT": [5.0],
                "M": [10.0],
            }
        )

        mock_exists.return_value = True
        mock_read_feather.return_value = cached_df
        # Mock glob to return a merged cache file path
        mock_glob.return_value = ["/fake/cache/1000/1000_history_until_5.00.df.feather"]

        result = particle_tracker.update_one_particle_history_df("test_simu", 1000, update=False)

        assert len(result) == 1
        assert result["Name"].iloc[0] == 1000

    def test_get_particle_df_from_hdf5_file_all_particles(self, particle_tracker, sample_df_dict):
        """Test processing all particles from HDF5 file"""
        # Mock the HDF5 file processor to return sample data
        with patch.object(
            particle_tracker.hdf5_file_processor,
            "read_file",
            return_value=sample_df_dict,
        ):
            result = particle_tracker.get_particle_df_from_hdf5_file(sample_df_dict, "all")

        # Should return a dict
        assert isinstance(result, dict)
        # Should have data for particles 1000 and 2000
        assert 1000 in result
        assert 2000 in result
        # Each should be a DataFrame
        assert isinstance(result[1000], pd.DataFrame)
        assert isinstance(result[2000], pd.DataFrame)

    def test_update_one_particle_reads_merged_cache(self, particle_tracker, mock_config, tmp_path):
        """Test that update_one_particle_history_df prioritizes merged cache format"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        # Create merged cache file
        particle_dir = tmp_path / "3000"
        particle_dir.mkdir()

        cached_df = pd.DataFrame(
            {
                "Name": [3000, 3000],
                "TTOT": [1.0, 2.0],
                "M": [20.0, 19.5],
            }
        )
        cached_df.to_feather(particle_dir / "3000_history_until_2.00.df.feather")

        # Read with update=False
        result = particle_tracker.update_one_particle_history_df("test_simu", 3000, update=False)

        assert len(result) == 2
        assert list(result["TTOT"]) == [1.0, 2.0]
        assert list(result["Name"]) == [3000, 3000]

    def test_build_progress_dict_no_files(self, particle_tracker, mock_config, tmp_path):
        """Test _build_progress_dict with no existing files"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        result = particle_tracker._build_progress_dict("test_simu", [1000, 2000, 3000])

        assert result == {1000: -1.0, 2000: -1.0, 3000: -1.0}

    def test_build_progress_dict_with_history_files(self, particle_tracker, mock_config, tmp_path):
        """Test _build_progress_dict with existing history files"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        # Create history files for particles
        particle_1_dir = tmp_path / "1000"
        particle_1_dir.mkdir()
        (particle_1_dir / "1000_history_until_5.00.df.feather").touch()

        particle_2_dir = tmp_path / "2000"
        particle_2_dir.mkdir()
        (particle_2_dir / "2000_history_until_10.50.df.feather").touch()

        result = particle_tracker._build_progress_dict("test_simu", [1000, 2000, 3000])

        assert result[1000] == 5.0
        assert result[2000] == 10.5
        assert result[3000] == -1.0  # No file for this particle

    def test_build_progress_dict_multiple_files_uses_max(
        self, particle_tracker, mock_config, tmp_path
    ):
        """Test _build_progress_dict with multiple history files takes max timestamp"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        # Create multiple history files for one particle
        particle_dir = tmp_path / "1000"
        particle_dir.mkdir()
        (particle_dir / "1000_history_until_3.00.df.feather").touch()
        (particle_dir / "1000_history_until_7.50.df.feather").touch()
        (particle_dir / "1000_history_until_5.00.df.feather").touch()

        result = particle_tracker._build_progress_dict("test_simu", [1000])

        # Should use max timestamp (7.50)
        assert result[1000] == 7.5

    def test_read_history_with_direct_path(self, particle_tracker, tmp_path):
        """Test read_history with direct feather path"""
        # Create a test feather file
        test_df = pd.DataFrame(
            {
                "Name": [1000, 1000],
                "TTOT": [1.0, 2.0],
                "M": [10.0, 9.5],
            }
        )
        feather_path = tmp_path / "test_history.feather"
        test_df.to_feather(feather_path)

        result = particle_tracker.read_history(feather_path=str(feather_path))

        assert len(result) == 2
        assert list(result["Name"]) == [1000, 1000]
        assert list(result["TTOT"]) == [1.0, 2.0]

    def test_read_history_with_simu_and_particle_name(
        self, particle_tracker, mock_config, tmp_path
    ):
        """Test read_history with simulation name and particle name"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        # Create particle directory and history file
        particle_dir = tmp_path / "5000"
        particle_dir.mkdir()

        test_df = pd.DataFrame(
            {
                "Name": [5000, 5000, 5000],
                "TTOT": [0.0, 1.0, 2.0],
                "M": [15.0, 14.5, 14.0],
            }
        )
        test_df.to_feather(particle_dir / "5000_history_until_2.00.df.feather")

        result = particle_tracker.read_history(simu_name="test_simu", particle_name=5000)

        assert len(result) == 3
        assert result["Name"].iloc[0] == 5000
        assert list(result["TTOT"]) == [0.0, 1.0, 2.0]

    def test_read_history_missing_file(self, particle_tracker):
        """Test read_history returns empty DataFrame for missing file"""
        result = particle_tracker.read_history(feather_path="/nonexistent/path.feather")
        assert result.empty

    def test_read_history_missing_params_raises_error(self, particle_tracker):
        """Test read_history raises ValueError when required params missing"""
        with pytest.raises(ValueError, match="Either feather_path or both simu_name"):
            particle_tracker.read_history()

    def test_read_history_partial_params_raises_error(self, particle_tracker):
        """Test read_history raises ValueError when only simu_name is provided"""
        with pytest.raises(ValueError, match="Either feather_path or both simu_name"):
            particle_tracker.read_history(simu_name="test_simu")

    def test_read_history_picks_latest_file(self, particle_tracker, mock_config, tmp_path):
        """Test read_history picks the latest history file"""
        mock_config.particle_df_cache_dir_of["test_simu"] = str(tmp_path)

        particle_dir = tmp_path / "6000"
        particle_dir.mkdir()

        # Create multiple history files with different timestamps
        df1 = pd.DataFrame({"Name": [6000], "TTOT": [5.0]})
        df2 = pd.DataFrame({"Name": [6000, 6000], "TTOT": [5.0, 10.0]})

        df1.to_feather(particle_dir / "6000_history_until_5.00.df.feather")
        df2.to_feather(particle_dir / "6000_history_until_10.00.df.feather")

        result = particle_tracker.read_history(simu_name="test_simu", particle_name=6000)

        # Should read the file with the higher timestamp (10.00)
        assert len(result) == 2
        assert result["TTOT"].max() == 10.0


class TestHDF5ParticleTask:
    """Tests for HDF5ParticleTask dataclass"""

    def test_dataclass_creation(self):
        """Test HDF5ParticleTask dataclass can be created"""
        from nbody_pipeline.analysis.particle_tracker import HDF5ParticleTask

        task = HDF5ParticleTask(
            hdf5_file_path="/path/to/file.h5part",
            simu_name="test_simu",
            particle_names=[1000, 2000, 3000],
            progress_dict={1000: 5.0, 2000: 10.0, 3000: -1.0},
        )

        assert task.hdf5_file_path == "/path/to/file.h5part"
        assert task.simu_name == "test_simu"
        assert task.particle_names == [1000, 2000, 3000]
        assert task.progress_dict == {1000: 5.0, 2000: 10.0, 3000: -1.0}


class TestPhysics:
    """Tests for physics functions"""

    def test_tau_gw_basic(self):
        """Test tau_gw with basic float inputs"""
        # Using simple values
        a = 1e10  # 10 Gm semi-major axis
        e = 0.0  # circular orbit
        mu = 1e30  # ~1 solar mass
        M = 2e30  # ~2 solar masses

        result = tau_gw(a, e, mu, M)

        # Should return a positive time
        assert result > 0
        assert isinstance(result, float)

    def test_tau_gw_eccentric(self):
        """Test tau_gw with eccentric orbit"""
        a = 1e10
        e = 0.5  # eccentric
        mu = 1e30
        M = 2e30

        result_eccentric = tau_gw(a, e, mu, M)
        result_circular = tau_gw(a, 0.0, mu, M)

        # Eccentric orbit should merge faster (smaller tau)
        assert result_eccentric < result_circular

    def test_tau_gw_with_astropy_units(self):
        """Test tau_gw with astropy Quantity inputs"""
        from astropy import units as u

        a = 1e10 * u.m
        e = 0.0
        mu = 1e30 * u.kg
        M = 2e30 * u.kg

        result = tau_gw(a, e, mu, M)

        # Should return a Quantity
        assert hasattr(result, "unit")
        assert result.value > 0

    def test_tau_gw_high_eccentricity(self):
        """Test tau_gw with high eccentricity"""
        a = 1e10
        e = 0.9  # very eccentric
        mu = 1e30
        M = 2e30

        result = tau_gw(a, e, mu, M)

        # Should still return valid positive time
        assert result > 0
        assert np.isfinite(result)


class TestBinaryOrbitFunctions:
    """Tests for binary orbit calculation functions"""

    def test_compute_binary_orbit_relative_positions_equal_masses(self):
        """Test compute_binary_orbit_relative_positions with equal masses"""
        from nbody_pipeline.analysis import compute_binary_orbit_relative_positions

        m1, m2 = 10.0, 10.0
        rel_x, rel_y, rel_z = 2.0, 0.0, 0.0

        (x1, y1, z1), (x2, y2, z2) = compute_binary_orbit_relative_positions(
            m1, m2, rel_x, rel_y, rel_z
        )

        # Equal masses: positions should be equal and opposite
        assert x1 == pytest.approx(-1.0)
        assert x2 == pytest.approx(1.0)
        assert y1 == 0.0
        assert y2 == 0.0
        assert z1 == 0.0
        assert z2 == 0.0

    def test_compute_binary_orbit_relative_positions_unequal_masses(self):
        """Test compute_binary_orbit_relative_positions with unequal masses"""
        from nbody_pipeline.analysis import compute_binary_orbit_relative_positions

        m1, m2 = 30.0, 10.0  # 3:1 mass ratio
        rel_x, rel_y, rel_z = 4.0, 0.0, 0.0

        (x1, y1, z1), (x2, y2, z2) = compute_binary_orbit_relative_positions(
            m1, m2, rel_x, rel_y, rel_z
        )

        # r1 = -m2/(m1+m2) * rel = -10/40 * 4 = -1
        # r2 = m1/(m1+m2) * rel = 30/40 * 4 = 3
        assert x1 == pytest.approx(-1.0)
        assert x2 == pytest.approx(3.0)

    def test_compute_binary_orbit_relative_positions_zero_mass(self):
        """Test compute_binary_orbit_relative_positions with zero total mass"""
        from nbody_pipeline.analysis import compute_binary_orbit_relative_positions

        m1, m2 = 0.0, 0.0
        rel_x, rel_y, rel_z = 2.0, 1.0, 0.5

        (x1, y1, z1), (x2, y2, z2) = compute_binary_orbit_relative_positions(
            m1, m2, rel_x, rel_y, rel_z
        )

        # Zero mass should return zero positions
        assert (x1, y1, z1) == (0.0, 0.0, 0.0)
        assert (x2, y2, z2) == (0.0, 0.0, 0.0)

    def test_compute_individual_orbit_params_equal_masses(self):
        """Test compute_individual_orbit_params with equal masses"""
        from nbody_pipeline.analysis import compute_individual_orbit_params

        a_bin, ecc_bin = 10.0, 0.5
        m1, m2 = 10.0, 10.0

        (a1, e1), (a2, e2) = compute_individual_orbit_params(a_bin, ecc_bin, m1, m2)

        # Equal masses: each star has half the semi-major axis
        assert a1 == pytest.approx(5.0)
        assert a2 == pytest.approx(5.0)
        assert e1 == pytest.approx(0.5)
        assert e2 == pytest.approx(0.5)

    def test_compute_individual_orbit_params_unequal_masses(self):
        """Test compute_individual_orbit_params with unequal masses"""
        from nbody_pipeline.analysis import compute_individual_orbit_params

        a_bin, ecc_bin = 100.0, 0.3
        m1, m2 = 30.0, 10.0  # 3:1 mass ratio

        (a1, e1), (a2, e2) = compute_individual_orbit_params(a_bin, ecc_bin, m1, m2)

        # a1 = a * m2/(m1+m2) = 100 * 10/40 = 25
        # a2 = a * m1/(m1+m2) = 100 * 30/40 = 75
        assert a1 == pytest.approx(25.0)
        assert a2 == pytest.approx(75.0)
        assert e1 == pytest.approx(0.3)
        assert e2 == pytest.approx(0.3)

    def test_compute_individual_orbit_params_zero_mass(self):
        """Test compute_individual_orbit_params with zero total mass"""
        from nbody_pipeline.analysis import compute_individual_orbit_params

        a_bin, ecc_bin = 10.0, 0.5
        m1, m2 = 0.0, 0.0

        (a1, e1), (a2, e2) = compute_individual_orbit_params(a_bin, ecc_bin, m1, m2)

        # Zero mass should return zero semi-major axes
        assert a1 == 0.0
        assert a2 == 0.0
        assert e1 == pytest.approx(0.5)
        assert e2 == pytest.approx(0.5)


class TestCurrentMassLagrangianProcessor:
    """Tests for current-mass Lagrangian processing."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        config = Mock()
        config.analysis_cache_dir_of = {"test_simu": str(tmp_path / "cache" / "test_simu")}
        config.particle_df_cache_dir_of = {
            "test_simu": str(tmp_path / "cache" / "test_simu" / "particle_df")
        }
        config.pathof = {"test_simu": str(tmp_path)}
        config.current_lagrangian = {
            "enabled": True,
            "cache_filename": "current_mass_lagr.feather",
        }
        config.hdf5 = {
            "file_selection": {
                "wait_age_hour": 0,
                "sample_every_nb_time": 1.0,
                "exclude_bad_dirname": True,
            },
            "table_cache": {"use_hdf5_cache": True},
            "scan": {"parallel": False, "incremental_from_cache_tail": True},
        }
        return config

    def test_compute_snapshot_uses_current_mass_weighting(self, mock_config):
        processor = CurrentMassLagrangianProcessor(mock_config)
        singles = pd.DataFrame(
            {
                "M": [1.0, 4.0, 5.0],
                "Distance_to_cluster_center[pc]": [1.0, 2.0, 3.0],
                "X [pc]": [1.0, 2.0, 3.0],
                "Y [pc]": [0.0, 0.0, 0.0],
                "Z [pc]": [0.0, 0.0, 0.0],
                "V1": [0.0, 10.0, 0.0],
                "V2": [0.0, 0.0, 0.0],
                "V3": [0.0, 0.0, 0.0],
            }
        )
        scalar = pd.Series({"TTOT": 1.0, "Time[Myr]": 10.0, "RC": 1.0, "RBAR": 2.0})

        row = processor.compute_snapshot(singles, scalar)

        assert row["rlagr5.00E-01"] == pytest.approx(2.0)
        assert row["avmass5.00E-01"] == pytest.approx(2.5)
        assert row["nshell5.00E-01"] == 2
        assert row["rlagr1.00E+00"] == pytest.approx(3.0)
        assert row["avmass1.00E+00"] == pytest.approx(10.0 / 3.0)
        assert row["nshell1.00E+00"] == 3
        assert row["rlagr<RC"] == pytest.approx(2.0)
        assert row["nshell<RC"] == 2
        assert row["vx5.00E-01"] == pytest.approx(4.0)
        assert row["sigma25.00E-01"] == pytest.approx(16.0)

    def test_update_writes_cache_and_skips_fresh_files(self, mock_config, tmp_path):
        processor = CurrentMassLagrangianProcessor(mock_config)
        hdf5_path = tmp_path / "snap.40_1.0.h5part"
        hdf5_path.touch()
        scalars = pd.DataFrame(
            {"TTOT": [1.0], "Time[Myr]": [10.0], "RC": [1.0], "RBAR": [2.0]}
        ).set_index("TTOT", drop=False)
        singles = pd.DataFrame(
            {
                "M": [1.0, 1.0],
                "Distance_to_cluster_center[pc]": [1.0, 2.0],
                "X [pc]": [1.0, 2.0],
                "Y [pc]": [0.0, 0.0],
                "Z [pc]": [0.0, 0.0],
                "V1": [0.0, 0.0],
                "V2": [0.0, 0.0],
                "V3": [0.0, 0.0],
                "TTOT": [1.0, 1.0],
            }
        )
        df_dict = {"scalars": scalars, "singles": singles, "binaries": pd.DataFrame()}

        processor.hdf5_file_processor.get_all_hdf5_paths = Mock(return_value=[str(hdf5_path)])
        processor.hdf5_file_processor.read_file = Mock(return_value=df_dict)
        processor.hdf5_file_processor.get_snapshot_at_t = Mock(
            return_value=(singles, pd.DataFrame({"should_not_be_used": [1]}), True)
        )

        first = processor.update("test_simu")
        second = processor.update("test_simu")

        assert len(first) == 1
        assert len(second) == 1
        assert processor.hdf5_file_processor.read_file.call_count == 1
        processor.hdf5_file_processor.read_file.assert_called_once_with(
            str(hdf5_path),
            "test_simu",
            use_cache=True,
            write_cache=False,
        )
        assert (
            tmp_path / "cache" / "test_simu" / "current_lagrangian" / "current_mass_lagr.feather"
        ).exists()
        assert (
            tmp_path / "cache" / "test_simu" / "current_lagrangian" / "current_mass_lagr.meta.json"
        ).exists()

    def test_update_can_insert_intermediate_time(self, mock_config, tmp_path):
        mock_config.hdf5["file_selection"]["sample_every_nb_time"] = 0.5
        processor = CurrentMassLagrangianProcessor(mock_config)
        paths = [tmp_path / "snap.40_1.0.h5part", tmp_path / "snap.40_0.5.h5part"]
        for path in paths:
            path.touch()

        def make_df_dict(ttot):
            scalars = pd.DataFrame(
                {"TTOT": [ttot], "Time[Myr]": [ttot * 10], "RC": [1.0], "RBAR": [2.0]}
            ).set_index("TTOT", drop=False)
            singles = pd.DataFrame(
                {
                    "M": [1.0],
                    "Distance_to_cluster_center[pc]": [1.0],
                    "X [pc]": [1.0],
                    "Y [pc]": [0.0],
                    "Z [pc]": [0.0],
                    "V1": [0.0],
                    "V2": [0.0],
                    "V3": [0.0],
                    "TTOT": [ttot],
                }
            )
            return {"scalars": scalars, "singles": singles, "binaries": pd.DataFrame()}

        processor.hdf5_file_processor.get_all_hdf5_paths = Mock(return_value=[str(paths[0])])
        processor.hdf5_file_processor.read_file = Mock(return_value=make_df_dict(1.0))
        processor.hdf5_file_processor.get_snapshot_at_t = Mock(
            side_effect=lambda df_dict, ttot: (
                df_dict["singles"],
                pd.DataFrame(),
                True,
            )
        )
        processor.update("test_simu")

        processor.hdf5_file_processor.get_all_hdf5_paths = Mock(
            return_value=[str(paths[1]), str(paths[0])]
        )
        processor.hdf5_file_processor.read_file = Mock(
            side_effect=lambda path, *_args, **_kwargs: make_df_dict(0.5 if "0.5" in path else 1.0)
        )
        updated = processor.update("test_simu")

        assert updated["Time[NB]"].tolist() == [0.5, 1.0]

    def test_load_sns_friendly_drops_time_nb_and_adds_sigma(self, mock_config):
        processor = CurrentMassLagrangianProcessor(mock_config)
        processor.update = Mock(
            return_value=pd.DataFrame(
                {
                    "Time[NB]": [1.0],
                    "Time[Myr]": [10.0],
                    "sigma21.00E+00": [4.0],
                    "rlagr1.00E+00": [2.0],
                }
            )
        )

        result = processor.load_sns_friendly_data("test_simu")

        assert "Time[NB]" not in result.columns
        assert "sigma" in set(result["Metric"])
        assert result.loc[result["Metric"] == "sigma", "Value"].iloc[0] == pytest.approx(2.0)


class TestGalacticEnergyAngularMomentumProcessor:
    """Tests for snapshot-level galactic energy and angular momentum."""

    def test_compute_snapshot_uses_mass_weighted_energy_and_lz(self):
        processor = GalacticEnergyAngularMomentumProcessor()
        singles = pd.DataFrame(
            {
                "M": [2.0, 3.0],
                "X [pc]": [1000.0, 0.0],
                "Y [pc]": [0.0, 2000.0],
                "Z [pc]": [0.0, 0.0],
                "V1": [10.0, 0.0],
                "V2": [0.0, 20.0],
                "V3": [0.0, 0.0],
            }
        )
        scalar = pd.Series(
            {
                "RBAR": 1.0,
                "VSTAR": 1.0,
                "RG(1)": 0.0,
                "RG(2)": 0.0,
                "RG(3)": 0.0,
                "VG(1)": 1.0,
                "VG(2)": 2.0,
                "VG(3)": 3.0,
            }
        )

        with patch(
            "nbody_pipeline.analysis.galactic_energy_angular_momentum._evaluate_mw_potential",
            return_value=np.array([-100.0, -200.0]),
        ):
            result = processor.compute_snapshot(singles, scalar)

        np.testing.assert_allclose(result[E_KIN_GAL_COL], [0.5 * 2.0 * 134.0, 0.5 * 3.0 * 494.0])
        np.testing.assert_allclose(result[E_POT_GAL_COL], [-200.0, -600.0])
        np.testing.assert_allclose(result[E_GAL_COL], [-66.0, 141.0])
        np.testing.assert_allclose(result[L_Z_GAL_COL], [4.0, -6.0])
        np.testing.assert_allclose(result[E_KIN_GAL_SPECIFIC_COL], [0.5 * 134.0, 0.5 * 494.0])
        np.testing.assert_allclose(result[E_POT_GAL_SPECIFIC_COL], [-100.0, -200.0])
        np.testing.assert_allclose(result[E_GAL_SPECIFIC_COL], [-33.0, 47.0])
        np.testing.assert_allclose(result[L_Z_GAL_SPECIFIC_COL], [4.0 / 2.0, -6.0 / 3.0])

    def test_raw_rg_vg_offsets_and_pc_to_kpc_conversion(self):
        """With RBAR=VSTAR=1.0, RG/VG pass through unscaled."""
        processor = GalacticEnergyAngularMomentumProcessor()
        singles = pd.DataFrame(
            {
                "M": [5.0],
                "X [pc]": [200.0],
                "Y [pc]": [300.0],
                "Z [pc]": [400.0],
                "V1": [7.0],
                "V2": [11.0],
                "V3": [13.0],
            }
        )
        scalar = pd.Series(
            {
                "RBAR": 1.0,
                "VSTAR": 1.0,
                "RG(1)": 800.0,
                "RG(2)": 1700.0,
                "RG(3)": 2600.0,
                "VG(1)": 3.0,
                "VG(2)": 5.0,
                "VG(3)": 7.0,
            }
        )
        observed = {}

        def fake_potential(radius_kpc, z_kpc, phi_rad):
            observed["radius_kpc"] = radius_kpc
            observed["z_kpc"] = z_kpc
            observed["phi_rad"] = phi_rad
            return np.array([0.0])

        with patch(
            "nbody_pipeline.analysis.galactic_energy_angular_momentum._evaluate_mw_potential",
            side_effect=fake_potential,
        ):
            result = processor.compute_snapshot(singles, scalar)

        np.testing.assert_allclose(observed["radius_kpc"], [np.sqrt(1.0**2 + 2.0**2)])
        np.testing.assert_allclose(observed["z_kpc"], [3.0])
        np.testing.assert_allclose(observed["phi_rad"], [np.arctan2(2000.0, 1000.0)])
        assert result[L_Z_GAL_COL].iloc[0] == pytest.approx(5.0 * (1.0 * 16.0 - 2.0 * 10.0))

    def test_rg_vg_are_scaled_by_rbar_and_vstar(self):
        """RG/VG in the HDF5 scalars table are raw N-body units; must be scaled by RBAR/VSTAR.

        Regression test for the bug found 2026-07-13: the scalars table's
        RG/VG are written by NBODY6++GPU without re-applying RSCALE_OUT/
        VSCALE_OUT (unlike X1/V1), so they need `* RBAR` / `* VSTAR` before
        being combined with the already-physical per-star X/V columns.
        """
        processor = GalacticEnergyAngularMomentumProcessor()
        singles = pd.DataFrame(
            {
                "M": [1.0],
                "X [pc]": [0.0],
                "Y [pc]": [0.0],
                "Z [pc]": [0.0],
                "V1": [0.0],
                "V2": [0.0],
                "V3": [0.0],
            }
        )
        scalar = pd.Series(
            {
                "RBAR": 2.17673,
                "VSTAR": 33.8996,
                "RG(1)": 2297.02,
                "RG(2)": 0.0,
                "RG(3)": 0.0,
                "VG(1)": 0.0,
                "VG(2)": 7.06852,
                "VG(3)": 12.2429,
            }
        )
        observed = {}

        def fake_potential(radius_kpc, z_kpc, phi_rad):
            observed["radius_kpc"] = radius_kpc
            return np.array([0.0])

        with patch(
            "nbody_pipeline.analysis.galactic_energy_angular_momentum._evaluate_mw_potential",
            side_effect=fake_potential,
        ):
            result = processor.compute_snapshot(singles, scalar)

        np.testing.assert_allclose(observed["radius_kpc"], [5.0], rtol=1e-4)
        expected_vg_kms = np.array([0.0, 7.06852, 12.2429]) * 33.8996
        expected_kin_specific = 0.5 * float(np.sum(expected_vg_kms**2))
        assert result[E_KIN_GAL_SPECIFIC_COL].iloc[0] == pytest.approx(expected_kin_specific)

    def test_evaluate_mw_potential_uses_physical_kpc_not_natural_units(self):
        """Regression test for the 2026-07-13 bug: bare floats passed to a
        physical-units-on galpy potential are read as natural (ro/vo-scaled)
        units, silently evaluating 8x too far out (ro=8). Attaching astropy
        units fixes this; values below are galpy's own MWPotential2014
        physical-unit potential at the solar radius and at 5 kpc.
        """
        import nbody_pipeline.analysis.galactic_energy_angular_momentum as module

        at_solar_radius = module._evaluate_mw_potential(
            np.array([8.0]), np.array([0.0]), np.array([0.0])
        )
        at_5kpc = module._evaluate_mw_potential(np.array([5.0]), np.array([0.0]), np.array([0.0]))

        assert at_solar_radius[0] == pytest.approx(-66470.17, rel=1e-4)
        assert at_5kpc[0] == pytest.approx(-90006.34, rel=1e-4)
        # The pre-fix bug returned a much shallower (less negative) value
        # because it evaluated at 8x the intended radius.
        assert at_solar_radius[0] < -10000.0

    def test_compute_cluster_com_specific_and_mass_weighted(self):
        processor = GalacticEnergyAngularMomentumProcessor()
        scalar = pd.Series(
            {
                "RBAR": 2.0,
                "VSTAR": 10.0,
                "RG(1)": 400.0,
                "RG(2)": 0.0,
                "RG(3)": 0.0,
                "VG(1)": 0.0,
                "VG(2)": 22.0,
                "VG(3)": 0.0,
            }
        )
        with patch(
            "nbody_pipeline.analysis.galactic_energy_angular_momentum._evaluate_mw_potential",
            return_value=np.array([-500.0]),
        ):
            com_no_mass = processor.compute_cluster_com(scalar)
            com_with_mass = processor.compute_cluster_com(scalar, representative_mass_msun=2.0)

        # RG(1)*RBAR = 800 pc -> x_kpc = 0.8; VG(2)*VSTAR = 220 km/s
        assert com_no_mass[E_KIN_GAL_SPECIFIC_COL] == pytest.approx(0.5 * 220.0**2)
        assert com_no_mass[E_POT_GAL_SPECIFIC_COL] == pytest.approx(-500.0)
        assert com_no_mass[E_GAL_SPECIFIC_COL] == pytest.approx(0.5 * 220.0**2 - 500.0)
        assert com_no_mass[L_Z_GAL_SPECIFIC_COL] == pytest.approx(0.8 * 220.0)
        assert E_KIN_GAL_COL not in com_no_mass

        assert com_with_mass[E_GAL_COL] == pytest.approx(2.0 * (0.5 * 220.0**2 - 500.0))
        assert com_with_mass[L_Z_GAL_COL] == pytest.approx(2.0 * 0.8 * 220.0)

    def test_mwpotential_info_log_only_once(self, caplog):
        import nbody_pipeline.analysis.galactic_energy_angular_momentum as module

        module._HAS_LOGGED_MW_POTENTIAL = False
        processor = GalacticEnergyAngularMomentumProcessor()
        singles = pd.DataFrame(
            {
                "M": [1.0],
                "X [pc]": [0.0],
                "Y [pc]": [0.0],
                "Z [pc]": [0.0],
                "V1": [0.0],
                "V2": [0.0],
                "V3": [0.0],
            }
        )
        scalar = pd.Series(
            {
                "RBAR": 1.0,
                "VSTAR": 1.0,
                "RG(1)": 0.0,
                "RG(2)": 0.0,
                "RG(3)": 0.0,
                "VG(1)": 0.0,
                "VG(2)": 0.0,
                "VG(3)": 0.0,
            }
        )

        with (
            patch(
                "nbody_pipeline.analysis.galactic_energy_angular_momentum._evaluate_mw_potential",
                return_value=np.array([0.0]),
            ),
            caplog.at_level("INFO"),
        ):
            processor.compute_snapshot(singles, scalar)
            processor.compute_snapshot(singles, scalar)

        assert caplog.text.count("MWPotential2014") == 1


class TestCompactBinaryCounter:
    """Tests for cross-snapshot compact binary counting."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        config = Mock()
        config.analysis_cache_dir_of = {"test_simu": str(tmp_path / "cache" / "test_simu")}
        config.particle_df_cache_dir_of = {
            "test_simu": str(tmp_path / "cache" / "test_simu" / "particle_df")
        }
        config.pathof = {"test_simu": str(tmp_path)}
        config.input_file_path_of = {"test_simu": str(tmp_path / "input.inp")}
        config.processes_count = 1
        config.tasks_per_child = 1
        config.hdf5 = {
            "file_selection": {
                "wait_age_hour": 3,
                "sample_every_nb_time": 0,
                "exclude_bad_dirname": False,
            },
            "table_cache": {"use_hdf5_cache": False},
            "scan": {"parallel": False, "incremental_from_cache_tail": True},
        }
        config.compact_object_KW = np.array([10, 11, 12, 13, 14])
        config.kw_to_stellar_type = {
            0: "MS",
            1: "MS",
            10: "WD",
            11: "WD",
            12: "WD",
            13: "NS",
            14: "BH",
        }
        return config

    @pytest.fixture
    def counter(self, mock_config):
        return CompactBinaryCounter(mock_config)

    def _binary_df(self, rows):
        return pd.DataFrame(
            rows,
            columns=["Bin Name1", "Bin Name2", "Bin KW1", "Bin KW2", "TTOT", "Time[Myr]"],
        )

    def test_classification_criteria(self, counter):
        binary_df = self._binary_df(
            [
                (1, 2, 14, 14, 1.0, 10.0),
                (3, 4, 14, 13, 1.0, 10.0),
                (5, 6, 13, 1, 1.0, 10.0),
                (7, 8, 13, 10, 1.0, 10.0),
                (9, 10, 10, 1, 1.0, 10.0),
                (11, 12, 14, 1, 1.0, 10.0),
            ]
        )
        records = {category: {} for category in counter.CATEGORIES}

        counter._accumulate_snapshot(records, binary_df)

        assert set(records["gw_source"]) == {(1, 2), (3, 4)}
        assert set(records["pulsar"]) == {(5, 6), (7, 8)}
        assert set(records["xray_binary"]) == {(5, 6), (9, 10), (11, 12)}

    def test_deduplicates_unordered_member_pairs_and_tracks_times(self, counter):
        records = {category: {} for category in counter.CATEGORIES}
        counter._accumulate_snapshot(records, self._binary_df([(100, 200, 13, 1, 1.0, 10.0)]))
        counter._accumulate_snapshot(records, self._binary_df([(200, 100, 1, 13, 2.0, 20.0)]))

        detail = counter._records_to_dataframe(records["pulsar"])

        assert len(detail) == 1
        row = detail.iloc[0]
        assert row["binary_key"] == (100, 200)
        assert row["first_ttot"] == 1.0
        assert row["last_ttot"] == 2.0
        assert row["n_snapshots_seen_in_category"] == 2

    def test_same_binary_can_be_counted_in_multiple_categories_over_time(self, counter):
        records = {category: {} for category in counter.CATEGORIES}
        counter._accumulate_snapshot(records, self._binary_df([(1, 2, 13, 1, 1.0, 10.0)]))
        counter._accumulate_snapshot(records, self._binary_df([(2, 1, 13, 10, 2.0, 20.0)]))

        assert list(records["pulsar"]) == [(1, 2)]
        assert list(records["xray_binary"]) == [(1, 2)]
        assert records["pulsar"][(1, 2)]["n_snapshots_seen_in_category"] == 2
        assert records["xray_binary"][(1, 2)]["n_snapshots_seen_in_category"] == 1

    def test_summarize_simulation_uses_hdf5_scan_parameters(self, counter):
        scalars = pd.DataFrame({"TTOT": [1.0, 2.0], "Time[Myr]": [10.0, 20.0]}).set_index(
            "TTOT", drop=False
        )
        binaries = pd.concat(
            [
                self._binary_df([(1, 2, 14, 14, 1.0, 10.0), (3, 4, 13, 1, 1.0, 10.0)]),
                self._binary_df([(2, 1, 14, 13, 2.0, 20.0), (5, 6, 10, 1, 2.0, 20.0)]),
            ],
            ignore_index=True,
        )

        counter.hdf5_file_processor.get_all_hdf5_paths = Mock(return_value=["/tmp/a.h5part"])
        counter.hdf5_file_processor.read_tables = Mock(
            return_value={"scalars": scalars, "binaries": binaries}
        )

        result = counter.summarize_simulation("test_simu")

        counter.hdf5_file_processor.get_all_hdf5_paths.assert_called_once_with(
            "test_simu",
            sample_every_nb_time=0,
            exclude_bad_dirname=False,
            wait_age_hour=3,
        )
        counter.hdf5_file_processor.read_tables.assert_called_once()
        assert counter.hdf5_file_processor.read_tables.call_args.kwargs["use_cache"] is False
        assert result["summary"] == {
            "gw_source": 1,
            "pulsar": 1,
            "xray_binary": 2,
            "scanned_files": 1,
            "scanned_snapshots": 2,
            "max_ttot": 2.0,
            "max_time_myr": 20.0,
        }
        assert list(result["details"]) == ["gw_source", "pulsar", "xray_binary"]
        assert result["details"]["gw_source"].iloc[0]["binary_key"] == (1, 2)


class TestSimulationPlotterCurrentLagrangian:
    """Integration tests for current-mass Lagrangian plotter wiring."""

    def test_plot_all_simulations_calls_current_lagrangian_when_enabled(self, tmp_path):
        config = Mock()
        config.pathof = {"test_simu": str(tmp_path)}
        config.current_lagrangian = {"enabled": True}
        config.galactic_orbit = {"enabled": False}
        config.processes_count = 1
        config.tasks_per_child = 1
        config.analysis_cache_dir_of = {"test_simu": str(tmp_path / "cache" / "test_simu")}
        config.particle_df_cache_dir_of = {
            "test_simu": str(tmp_path / "cache" / "test_simu" / "particle_df")
        }

        class FakePool:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, iterable):
                return iter(())

        class FakeContext:
            def Pool(self, **_kwargs):
                return FakePool()

        with (
            patch.object(
                SimulationPlotter, "__init__", lambda self, cfg: setattr(self, "config", cfg)
            ),
            patch.object(SimulationPlotter, "update_analysis_store") as update_analysis_store,
            patch.object(SimulationPlotter, "plot_lagr") as plot_lagr,
            patch.object(SimulationPlotter, "plot_current_mass_lagr") as plot_current,
            patch(
                "nbody_pipeline.__main__.multiprocessing.get_context", return_value=FakeContext()
            ),
        ):
            plotter = SimulationPlotter(config)
            plotter.hdf5_file_processor = Mock()
            plotter.hdf5_file_processor.get_all_hdf5_paths.return_value = []
            plotter.plot_all_simulations()

        update_analysis_store.assert_called_once_with("test_simu")
        plot_lagr.assert_called_once_with("test_simu")
        plot_current.assert_called_once_with("test_simu")


class TestGalacticOrbitProcessor:
    """Tests for galactic-orbit HDF5 scan processing."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        config = Mock()
        config.analysis_cache_dir_of = {"test_simu": str(tmp_path / "cache" / "test_simu")}
        config.particle_df_cache_dir_of = {
            "test_simu": str(tmp_path / "cache" / "test_simu" / "particle_df")
        }
        config.pathof = {"test_simu": str(tmp_path)}
        config.processes_count = 1
        config.tasks_per_child = 1
        config.galactic_orbit = {
            "enabled": True,
            "cache_filename": "galactic_orbit.feather",
            "time_color_max_myr": 500.0,
        }
        config.hdf5 = {
            "file_selection": {
                "wait_age_hour": 0,
                "sample_every_nb_time": 1.0,
                "exclude_bad_dirname": True,
            },
            "table_cache": {"use_hdf5_cache": True},
            "scan": {"parallel": False, "incremental_from_cache_tail": True},
        }
        return config

    def _scalar_df(self, ttot_values):
        return pd.DataFrame(
            {
                "TTOT": ttot_values,
                "Time[Myr]": [value * 10.0 for value in ttot_values],
                "RG(1)": [value for value in ttot_values],
                "RG(2)": [value + 1.0 for value in ttot_values],
                "RG(3)": [value + 2.0 for value in ttot_values],
                "VG(1)": [value + 3.0 for value in ttot_values],
                "VG(2)": [value + 4.0 for value in ttot_values],
                "VG(3)": [value + 5.0 for value in ttot_values],
            }
        ).set_index("TTOT", drop=False)

    def test_update_caches_sampled_scalar_rows_and_skips_fresh_files(self, mock_config, tmp_path):
        processor = GalacticOrbitProcessor(mock_config)
        hdf5_path = tmp_path / "snap.40_1.0.h5part"
        hdf5_path.touch()
        processor.hdf5_file_processor.get_all_hdf5_paths = Mock(return_value=[str(hdf5_path)])
        processor.hdf5_file_processor.get_hdf5_file_time_from_filename = Mock(return_value=1.0)
        processor.hdf5_file_processor.read_tables = Mock(
            return_value={"scalars": self._scalar_df([1.0, 1.5, 2.0])}
        )

        first = processor.update("test_simu")
        second = processor.update("test_simu")

        assert len(first) == 2
        assert len(second) == 2
        assert processor.hdf5_file_processor.read_tables.call_count == 1
        assert first["TTOT"].tolist() == [1.0, 2.0]
        assert list(first["source_row_index"]) == [0, 1]
        cache_path = tmp_path / "cache" / "test_simu" / "galactic_orbit" / "galactic_orbit.feather"
        assert cache_path.exists()
        assert cache_path.with_name("galactic_orbit.meta.json").exists()

    def test_duplicate_ttot_warns_and_plot_data_keeps_first(self, mock_config, tmp_path, caplog):
        processor = GalacticOrbitProcessor(mock_config)
        hdf5_paths = [tmp_path / "snap.40_2.0.h5part", tmp_path / "snap.40_1.0.h5part"]
        for path in hdf5_paths:
            path.touch()

        processor.hdf5_file_processor.get_all_hdf5_paths = Mock(
            return_value=[str(hdf5_paths[0]), str(hdf5_paths[1])]
        )
        processor.hdf5_file_processor.get_hdf5_file_time_from_filename = Mock(
            side_effect=lambda path: 2.0 if "2.0" in path else 1.0
        )
        processor.hdf5_file_processor.read_tables = Mock(
            side_effect=lambda path, *_args, **_kwargs: {
                "scalars": self._scalar_df([1.0 if "2.0" in path else 1.0])
            }
        )

        with caplog.at_level("WARNING"):
            plot_df = processor.load_plot_data("test_simu")

        assert len(plot_df) == 1
        assert plot_df["source_file_time"].iloc[0] == pytest.approx(1.0)
        assert "Duplicate galactic-orbit TTOT values detected" in caplog.text

    def test_build_scan_job_requests_only_scalar_orbit_columns(self, mock_config):
        processor = GalacticOrbitProcessor(mock_config)
        task = processor.build_scan_job("test_simu").task

        assert task.required_tables == ("scalars",)
        assert task.columns_by_table == {
            "scalars": [
                "TTOT",
                "Time[Myr]",
                "RG(1)",
                "RG(2)",
                "RG(3)",
                "VG(1)",
                "VG(2)",
                "VG(3)",
            ]
        }


class TestSimulationPlotterGalacticOrbit:
    """Integration tests for galactic-orbit plotter wiring."""

    def test_plot_all_simulations_calls_galactic_orbit_when_enabled(self, tmp_path):
        config = Mock()
        config.pathof = {"test_simu": str(tmp_path)}
        config.current_lagrangian = {"enabled": False}
        config.galactic_orbit = {"enabled": True}
        config.processes_count = 1
        config.tasks_per_child = 1

        class FakePool:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, iterable):
                return iter(())

        class FakeContext:
            def Pool(self, **_kwargs):
                return FakePool()

        with (
            patch.object(
                SimulationPlotter, "__init__", lambda self, cfg: setattr(self, "config", cfg)
            ),
            patch.object(SimulationPlotter, "update_analysis_store") as update_analysis_store,
            patch.object(SimulationPlotter, "plot_lagr") as plot_lagr,
            patch.object(SimulationPlotter, "plot_galactic_orbit") as plot_orbit,
            patch(
                "nbody_pipeline.__main__.multiprocessing.get_context", return_value=FakeContext()
            ),
        ):
            plotter = SimulationPlotter(config)
            plotter.hdf5_file_processor = Mock()
            plotter.hdf5_file_processor.get_all_hdf5_paths.return_value = []
            plotter.plot_all_simulations()

        update_analysis_store.assert_called_once_with("test_simu")
        plot_lagr.assert_called_once_with("test_simu")
        plot_orbit.assert_called_once_with("test_simu")


class TestSimulationPlotterGalacticEnergyAngularMomentum:
    """Integration tests for snapshot galactic E-vs-Lz plotter wiring."""

    def _plotter(self, enabled=True, skip_existing=False, path_exists=False):
        config = Mock()
        config.skip_until_of = {"test_simu": 0.0}
        config.hdf5 = {"file_selection": {"sample_every_nb_time": 1.0}}
        config.skip_existing_plot = skip_existing
        config.galactic_energy_angular_momentum = {"enabled": enabled}

        singles = pd.DataFrame({"TTOT": [1.0], "value": [1], "M": [2.0]})
        scalars = pd.DataFrame(
            {
                "TTOT": [1.0],
                "RBAR": [1.0],
                "VSTAR": [1.0],
                "RG(1)": [0.0],
                "RG(2)": [0.0],
                "RG(3)": [0.0],
                "VG(1)": [0.0],
                "VG(2)": [0.0],
                "VG(3)": [0.0],
            }
        ).set_index("TTOT", drop=False)
        df_dict = {"singles": singles, "binaries": pd.DataFrame(), "scalars": scalars}

        plotter = SimulationPlotter.__new__(SimulationPlotter)
        plotter.config = config
        plotter.hdf5_file_processor = Mock()
        plotter.hdf5_file_processor.get_hdf5_file_time_from_filename.return_value = 1.0
        plotter.hdf5_file_processor.read_file.return_value = df_dict
        plotter.hdf5_file_processor.get_snapshot_at_t.return_value = (
            singles,
            pd.DataFrame(),
            True,
        )
        plotter.hdf5_visualizer = Mock()
        single_visualizer = plotter.hdf5_visualizer.single
        single_visualizer.galactic_energy_angular_momentum_plot_jpg_path.return_value = (
            "/tmp/existing.jpg"
        )
        single_visualizer.galactic_energy_angular_momentum_specific_plot_jpg_path.return_value = (
            "/tmp/existing_specific.jpg"
        )
        single_visualizer.galactic_kinetic_energy_specific_plot_jpg_path.return_value = (
            "/tmp/existing_ke.jpg"
        )
        plotter.galactic_energy_angular_momentum_processor = Mock()
        galactic_df = pd.DataFrame({"TTOT": [1.0], E_GAL_COL: [1.0], L_Z_GAL_COL: [2.0]})
        plotter.galactic_energy_angular_momentum_processor.compute_snapshot.return_value = (
            galactic_df
        )
        com_point = {E_GAL_COL: -1.0, L_Z_GAL_COL: 0.5}
        plotter.galactic_energy_angular_momentum_processor.compute_cluster_com.return_value = (
            com_point
        )
        return plotter, singles, scalars.iloc[0], path_exists, com_point

    def test_plot_hdf5_file_calls_processor_and_visualizer_when_enabled(self):
        plotter, singles, scalar, path_exists, com_point = self._plotter(enabled=True)

        with patch("nbody_pipeline.__main__.os.path.exists", return_value=path_exists):
            plotter.plot_hdf5_file("/tmp/snap.40_1.0.h5part", "test_simu")

        plotter.galactic_energy_angular_momentum_processor.compute_snapshot.assert_called_once()
        call_args = plotter.galactic_energy_angular_momentum_processor.compute_snapshot.call_args
        pd.testing.assert_frame_equal(call_args.args[0], singles)
        pd.testing.assert_series_equal(call_args.args[1], scalar)

        com_call_args = (
            plotter.galactic_energy_angular_momentum_processor.compute_cluster_com.call_args
        )
        pd.testing.assert_series_equal(com_call_args.args[0], scalar)
        assert com_call_args.kwargs["representative_mass_msun"] == pytest.approx(2.0)

        single_visualizer = plotter.hdf5_visualizer.single
        single_visualizer.create_galactic_energy_angular_momentum_plot_jpg.assert_called_once()
        assert (
            single_visualizer.create_galactic_energy_angular_momentum_plot_jpg.call_args.kwargs[
                "com_point"
            ]
            == com_point
        )
        single_visualizer.create_galactic_energy_angular_momentum_specific_plot_jpg.assert_called_once()
        single_visualizer.create_galactic_kinetic_energy_specific_plot_jpg.assert_called_once()

    def test_plot_hdf5_file_skips_when_disabled(self):
        plotter, _, _, path_exists, _ = self._plotter(enabled=False)

        with patch("nbody_pipeline.__main__.os.path.exists", return_value=path_exists):
            plotter.plot_hdf5_file("/tmp/snap.40_1.0.h5part", "test_simu")

        plotter.galactic_energy_angular_momentum_processor.compute_snapshot.assert_not_called()
        plotter.hdf5_visualizer.single.create_galactic_energy_angular_momentum_plot_jpg.assert_not_called()

    def test_plot_hdf5_file_skips_compute_when_existing_jpg_is_reused(self):
        plotter, _, _, path_exists, _ = self._plotter(skip_existing=True, path_exists=True)

        with patch("nbody_pipeline.__main__.os.path.exists", return_value=path_exists):
            plotter.plot_hdf5_file("/tmp/snap.40_1.0.h5part", "test_simu")

        plotter.galactic_energy_angular_momentum_processor.compute_snapshot.assert_not_called()
        plotter.hdf5_visualizer.single.create_galactic_energy_angular_momentum_plot_jpg.assert_not_called()
