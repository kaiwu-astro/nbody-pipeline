"""
Tests for dragon3_pipelines.config module
"""

from pathlib import Path

import yaml
import pytest

from dragon3_pipelines.config import ConfigManager, load_config
import dragon3_pipelines


class TestConfigManager:
    """Tests for ConfigManager class"""

    def test_load_default_config(self):
        """Test loading default configuration"""
        config = ConfigManager()

        # Check basic attributes are loaded
        assert hasattr(config, "pathof")
        assert hasattr(config, "plot_dir")
        assert hasattr(config, "figname_prefix")
        assert hasattr(config, "processes_count")
        assert hasattr(config, "kw_to_stellar_type")
        assert hasattr(config, "galactic_orbit")
        assert hasattr(config, "hdf5")

        # Check specific values
        assert isinstance(config.pathof, dict)
        assert isinstance(config.processes_count, int)
        assert config.processes_count == 40

        # Check new memory and inode parameters
        assert hasattr(config, "mem_max_gb")
        assert hasattr(config, "inode_limit")
        assert config.mem_max_gb == 40.0
        assert config.inode_limit == 2000000
        assert config.galactic_orbit["enabled"] is True
        assert config.galactic_orbit["cache_filename"] == "galactic_orbit.feather"
        assert config.galactic_orbit["time_color_max_myr"] == 500.0
        assert config.hdf5["file_selection"]["wait_age_hour"] == 24
        assert config.hdf5["file_selection"]["sample_every_nb_time"] == 1.0
        assert config.hdf5["file_selection"]["exclude_bad_dirname"] is True
        assert config.hdf5["table_cache"]["use_hdf5_cache"] is True
        assert config.hdf5["scan"]["parallel"] is True
        assert config.hdf5["scan"]["incremental_from_cache_tail"] is True

        # Check stellar types are loaded
        assert 14 in config.kw_to_stellar_type
        assert config.kw_to_stellar_type[14] == "BH"
        assert 13 in config.kw_to_stellar_type
        assert config.kw_to_stellar_type[13] == "NS"

    def test_derived_attributes(self):
        """Test that derived attributes are created correctly"""
        config = ConfigManager()

        # Check reverse mappings
        assert hasattr(config, "stellar_type_to_kw")
        assert "BH" in config.stellar_type_to_kw
        assert config.stellar_type_to_kw["BH"] == 14

        # Check verbose mappings
        assert hasattr(config, "kw_to_stellar_type_verbose")
        assert 14 in config.kw_to_stellar_type_verbose
        assert "14" in config.kw_to_stellar_type_verbose[14]
        assert "BH" in config.kw_to_stellar_type_verbose[14]

        # Check plotting attributes
        assert hasattr(config, "palette_st")
        assert hasattr(config, "marker_fill_list")
        assert len(config.marker_fill_list) > 0

        # Check mem_cap_bytes is calculated
        assert hasattr(config, "mem_cap_bytes")
        assert isinstance(config.mem_cap_bytes, int)
        assert config.mem_cap_bytes > 0
        # Should be at most mem_max_gb * 1024^3 / 2
        max_expected = int(config.mem_max_gb * 1024**3) // 2
        assert config.mem_cap_bytes <= max_expected

    def test_load_config_function(self):
        """Test load_config convenience function"""
        config = load_config()
        assert isinstance(config, ConfigManager)
        assert hasattr(config, "pathof")

    def test_default_analysis_cache_paths(self):
        """Test default analysis cache root and derived feature directories."""
        config = ConfigManager()
        default_config_path = (
            Path(__file__).parents[1] / "dragon3_pipelines" / "config" / "default_config.yaml"
        )
        with open(default_config_path) as f:
            root = yaml.safe_load(f)["paths"]["analysis_cache_dir"]

        assert config.analysis_cache_dir == root
        assert config.analysis_cache_dir_of["0sb"] == f"{root}/0sb"
        assert config.particle_df_cache_dir_of["0sb"] == f"{root}/0sb/particle_df"

    def test_user_config_overrides_analysis_cache_dir(self, temp_dir):
        """Test user config can override the analysis cache root."""
        root = str(temp_dir / "analysis_cache")
        user_config = {"paths": {"analysis_cache_dir": root}}
        user_config_path = temp_dir / "user_cache_config.yaml"
        with open(user_config_path, "w") as f:
            yaml.dump(user_config, f)

        config = ConfigManager(config_path=str(user_config_path))

        assert config.analysis_cache_dir == root
        assert config.analysis_cache_dir_of["0sb"] == f"{root}/0sb"
        assert config.particle_df_cache_dir_of["0sb"] == f"{root}/0sb/particle_df"

    def test_user_config_overrides_galactic_orbit_and_hdf5(self, temp_dir):
        """Test user config can override feature and global HDF5 settings."""
        user_config = {
            "galactic_orbit": {"enabled": False, "time_color_max_myr": 750.0},
            "hdf5": {
                "file_selection": {"sample_every_nb_time": 2.0, "wait_age_hour": 0},
                "scan": {"parallel": False},
            },
        }
        user_config_path = temp_dir / "user_galactic_orbit_config.yaml"
        with open(user_config_path, "w") as f:
            yaml.dump(user_config, f)

        config = ConfigManager(config_path=str(user_config_path))

        assert config.galactic_orbit["enabled"] is False
        assert config.galactic_orbit["time_color_max_myr"] == 750.0
        assert config.galactic_orbit["cache_filename"] == "galactic_orbit.feather"
        assert config.hdf5["file_selection"]["sample_every_nb_time"] == 2.0
        assert config.hdf5["file_selection"]["wait_age_hour"] == 0
        assert config.hdf5["scan"]["parallel"] is False
        assert config.hdf5["table_cache"]["use_hdf5_cache"] is True

    def test_removed_pre_1_config_keys_raise(self, temp_dir):
        user_config = {
            "processing": {"plot_only_int_nbody_time": True},
            "galactic_orbit": {"sample_every_nb_time": 2.0},
        }
        user_config_path = temp_dir / "old_config.yaml"
        with open(user_config_path, "w") as f:
            yaml.dump(user_config, f)

        with pytest.raises(ValueError, match="hdf5.file_selection.sample_every_nb_time"):
            ConfigManager(config_path=str(user_config_path))

    def test_version_is_1_0_0(self):
        assert dragon3_pipelines.__version__ == "1.0.0"

    def test_user_config_merge(self, temp_dir):
        """Test merging user configuration with defaults"""
        # Create a user config file
        user_config = {
            "paths": {"simulations": {"test_sim": "/path/to/test"}, "plot_dir": "/custom/plot/dir"},
            "processing": {"processes_count": 20, "mem_max_gb": 80.0, "inode_limit": 5000000},
        }

        user_config_path = temp_dir / "user_config.yaml"
        with open(user_config_path, "w") as f:
            yaml.dump(user_config, f)

        config = ConfigManager(config_path=str(user_config_path))

        # Check that user config is merged
        assert config.plot_dir == "/custom/plot/dir"
        assert config.processes_count == 20
        assert config.mem_max_gb == 80.0
        assert config.inode_limit == 5000000
        assert "test_sim" in config.pathof

        # Check that defaults are still present
        assert "0sb" in config.pathof  # from default config

        # Check that mem_cap_bytes is recalculated with new value
        max_expected = int(80.0 * 1024**3) // 2
        assert config.mem_cap_bytes <= max_expected

    def test_physics_constants(self):
        """Test physics constants are loaded correctly"""
        config = ConfigManager()

        assert config.ECLOSE_INPUT == 1.0
        assert config.universe_age_myr == 13800.0
        assert isinstance(config.IMBH_mass_range_msun, tuple)
        assert len(config.IMBH_mass_range_msun) == 2
        assert isinstance(config.PISNe_mass_gap, tuple)
        assert len(config.PISNe_mass_gap) == 2

    def test_limits_and_labels(self):
        """Test that limits and labels are loaded"""
        config = ConfigManager()

        assert hasattr(config, "limits")
        assert isinstance(config.limits, dict)
        assert len(config.limits) > 0

        assert hasattr(config, "colname_to_label")
        assert isinstance(config.colname_to_label, dict)
        assert len(config.colname_to_label) > 0

        # Config keys should match DataFrame column names exactly.
        assert "Bin A[au]" in config.limits
        assert "Distance_to_cluster_center[pc]" in config.limits
        assert "mass_ratio" in config.limits
        assert "Bin A[au]" in config.colname_to_label

        # Guard against reintroducing implicit key formatting.
        assert "Bin A [au]" not in config.limits
        assert "Bin A au" not in config.limits
        assert "Distance to cluster center [pc]" not in config.limits
