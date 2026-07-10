"""
Tests for dragon3_pipelines.config module
"""

import yaml
import pytest

from dragon3_pipelines.config import ConfigManager, load_config
import dragon3_pipelines


@pytest.fixture
def base_paths_config(temp_dir):
    """Minimal user config satisfying ConfigManager's required path keys.

    The packaged default config ships no site-specific paths (see
    config.example.yaml), so tests that exercise a "working" ConfigManager
    must supply paths.simulations/plot_dir/analysis_cache_dir themselves.
    """
    config_path = temp_dir / "base_paths_config.yaml"
    user_config = {
        "paths": {
            "simulations": {"0sb": str(temp_dir / "0sb")},
            "plot_dir": str(temp_dir / "plots"),
            "analysis_cache_dir": str(temp_dir / "analysis_cache"),
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(user_config, f)
    return config_path


class TestConfigManager:
    """Tests for ConfigManager class"""

    def test_load_default_config(self, base_paths_config):
        """Test loading default configuration"""
        config = ConfigManager(config_path=str(base_paths_config))

        # Check basic attributes are loaded
        assert hasattr(config, "pathof")
        assert hasattr(config, "plot_dir")
        assert hasattr(config, "figname_prefix")
        assert hasattr(config, "processes_count")
        assert hasattr(config, "kw_to_stellar_type")
        assert hasattr(config, "galactic_orbit")
        assert hasattr(config, "galactic_energy_angular_momentum")
        assert hasattr(config, "hdf5")

        # Check basic types
        assert isinstance(config.pathof, dict)
        assert isinstance(config.processes_count, int)

        # Check memory and inode parameters exist
        assert hasattr(config, "mem_max_gb")
        assert hasattr(config, "inode_limit")
        assert isinstance(config.mem_max_gb, float)
        assert isinstance(config.inode_limit, int)

        # Check nested config sections are loaded
        assert isinstance(config.galactic_orbit, dict)
        assert isinstance(config.galactic_energy_angular_momentum, dict)
        assert config.galactic_energy_angular_momentum["enabled"] is True
        assert config.galactic_energy_angular_momentum["percentile_limits"] == [0.1, 99.9]
        assert isinstance(config.hdf5, dict)
        assert "file_selection" in config.hdf5
        assert "table_cache" in config.hdf5
        assert "scan" in config.hdf5
        assert config.hdf5["scan"]["checkpoint_every_files"] == 100

        # Check stellar types are loaded
        assert len(config.kw_to_stellar_type) > 0
        assert all(isinstance(key, int) for key in config.kw_to_stellar_type)
        assert all(isinstance(value, str) for value in config.kw_to_stellar_type.values())

    def test_derived_attributes(self, base_paths_config):
        """Test that derived attributes are created correctly"""
        config = ConfigManager(config_path=str(base_paths_config))

        # Check reverse mappings
        assert hasattr(config, "stellar_type_to_kw")
        for kw, stellar_type in config.kw_to_stellar_type.items():
            assert config.stellar_type_to_kw[stellar_type] == kw

        # Check verbose mappings
        assert hasattr(config, "kw_to_stellar_type_verbose")
        for kw, stellar_type in config.kw_to_stellar_type.items():
            assert kw in config.kw_to_stellar_type_verbose
            assert str(kw) in config.kw_to_stellar_type_verbose[kw]
            assert stellar_type in config.kw_to_stellar_type_verbose[kw]

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

    def test_load_config_function(self, base_paths_config):
        """Test load_config convenience function"""
        config = load_config(str(base_paths_config))
        assert isinstance(config, ConfigManager)
        assert hasattr(config, "pathof")

    def test_missing_paths_raise_clear_error(self):
        """A fresh install with no user config must fail with actionable guidance."""
        with pytest.raises(ValueError, match="config.example.yaml") as exc_info:
            ConfigManager()
        assert "paths.simulations" in str(exc_info.value)
        assert "paths.plot_dir" in str(exc_info.value)
        assert "paths.analysis_cache_dir" in str(exc_info.value)

    def test_partial_paths_raise_clear_error(self, temp_dir):
        """Missing a single required path key must still be reported explicitly."""
        user_config = {
            "paths": {
                "simulations": {"0sb": str(temp_dir / "0sb")},
                "plot_dir": str(temp_dir / "plots"),
                # analysis_cache_dir intentionally omitted
            }
        }
        user_config_path = temp_dir / "partial_config.yaml"
        with open(user_config_path, "w") as f:
            yaml.dump(user_config, f)

        with pytest.raises(ValueError, match="paths.analysis_cache_dir"):
            ConfigManager(config_path=str(user_config_path))

    def test_default_analysis_cache_paths(self, base_paths_config, temp_dir):
        """Test analysis cache root and derived feature directories."""
        config = ConfigManager(config_path=str(base_paths_config))
        root = str(temp_dir / "analysis_cache")

        assert config.analysis_cache_dir == root
        assert config.analysis_cache_dir_of["0sb"] == f"{root}/0sb"
        assert config.particle_df_cache_dir_of["0sb"] == f"{root}/0sb/particle_df"

    def test_user_config_overrides_analysis_cache_dir(self, temp_dir):
        """Test user config can override the analysis cache root."""
        root = str(temp_dir / "analysis_cache")
        user_config = {
            "paths": {
                "simulations": {"0sb": str(temp_dir / "0sb")},
                "plot_dir": str(temp_dir / "plots"),
                "analysis_cache_dir": root,
            }
        }
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
            "paths": {
                "simulations": {"0sb": str(temp_dir / "0sb")},
                "plot_dir": str(temp_dir / "plots"),
                "analysis_cache_dir": str(temp_dir / "analysis_cache"),
            },
            "galactic_orbit": {"enabled": False, "time_color_max_myr": 750.0},
            "galactic_energy_angular_momentum": {
                "enabled": False,
                "percentile_limits": [1.0, 99.0],
            },
            "hdf5": {
                "file_selection": {"sample_every_nb_time": 2.0, "wait_age_hour": 0},
                "scan": {"parallel": False, "checkpoint_every_files": 25},
            },
        }
        user_config_path = temp_dir / "user_galactic_orbit_config.yaml"
        with open(user_config_path, "w") as f:
            yaml.dump(user_config, f)

        config = ConfigManager(config_path=str(user_config_path))

        assert config.galactic_orbit["enabled"] is False
        assert config.galactic_orbit["time_color_max_myr"] == 750.0
        assert config.galactic_energy_angular_momentum["enabled"] is False
        assert config.galactic_energy_angular_momentum["percentile_limits"] == [1.0, 99.0]
        assert config.hdf5["file_selection"]["sample_every_nb_time"] == 2.0
        assert config.hdf5["file_selection"]["wait_age_hour"] == 0
        assert config.hdf5["scan"]["parallel"] is False
        assert config.hdf5["scan"]["checkpoint_every_files"] == 25

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
            "paths": {
                "simulations": {"test_sim": "/path/to/test"},
                "plot_dir": "/custom/plot/dir",
                "analysis_cache_dir": str(temp_dir / "analysis_cache"),
            },
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

        # Check that mem_cap_bytes is recalculated with new value
        max_expected = int(80.0 * 1024**3) // 2
        assert config.mem_cap_bytes <= max_expected

    def test_physics_constants(self, base_paths_config):
        """Test physics constants are loaded correctly"""
        config = ConfigManager(config_path=str(base_paths_config))

        assert isinstance(config.IMBH_mass_range_msun, tuple)
        assert len(config.IMBH_mass_range_msun) == 2
        assert isinstance(config.PISNe_mass_gap, tuple)
        assert len(config.PISNe_mass_gap) == 2

    def test_limits_and_labels(self, base_paths_config):
        """Test that limits and labels are loaded"""
        config = ConfigManager(config_path=str(base_paths_config))

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
