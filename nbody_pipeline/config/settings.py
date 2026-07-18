"""
Configuration management for nbody_pipeline
"""

import os
import yaml
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from glob import glob
import logging

from nbody_pipeline.utils import can_convert_to_float

logger = logging.getLogger(__name__)

REMOVED_CONFIG_KEYS: Dict[Tuple[str, ...], str] = {
    ("processing", "plot_only_int_nbody_time"): "hdf5.file_selection.sample_every_nb_time",
    ("current_lagrangian", "sample_every_nb_time"): "hdf5.file_selection.sample_every_nb_time",
    ("current_lagrangian", "wait_age_hour"): "hdf5.file_selection.wait_age_hour",
    ("current_lagrangian", "parallel"): "hdf5.scan.parallel",
    ("current_lagrangian", "processes"): "processing.processes_count",
    ("galactic_orbit", "sample_every_nb_time"): "hdf5.file_selection.sample_every_nb_time",
    ("galactic_orbit", "wait_age_hour"): "hdf5.file_selection.wait_age_hour",
    ("galactic_orbit", "parallel"): "hdf5.scan.parallel",
    ("galactic_orbit", "processes"): "processing.processes_count",
    (
        "binary_stellar_type_extraction",
        "sample_every_nb_time",
    ): "hdf5.file_selection.sample_every_nb_time",
    ("binary_stellar_type_extraction", "wait_age_hour"): "hdf5.file_selection.wait_age_hour",
    ("binary_stellar_type_extraction", "parallel"): "hdf5.scan.parallel",
    ("binary_stellar_type_extraction", "processes"): "processing.processes_count",
}


def load_config(config_path: Optional[str] = None) -> "ConfigManager":
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to user config file. If None, uses default config only.

    Returns:
        ConfigManager instance with loaded configuration
    """
    return ConfigManager(config_path=config_path)


class ConfigManager:
    """
    Configuration manager for nbody_pipeline.

    Loads default configuration and optionally merges with user configuration.
    """

    def __init__(self, config_path: Optional[str] = None, opts: List[Tuple[str, str]] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to user configuration file
            opts: Optional list of command-line options as (option, argument) tuples
        """
        self._load_default_config()
        if config_path and os.path.exists(config_path):
            self._merge_user_config(config_path)

        # Initialize derived attributes
        self._setup_derived_attributes()

        # Parse command line arguments if provided
        if opts:
            self._parse_argv(opts)

        # Handle 'last' in skip_until
        self._resolve_skip_until_last()

    def _load_default_config(self) -> None:
        """Load default configuration from package"""
        config_dir = Path(__file__).parent
        default_config_path = config_dir / "default_config.yaml"

        with open(default_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Path configurations. The packaged default ships no site-specific paths
        # (see config.example.yaml); real values come from a user config supplied
        # via --config / NBODY_CONFIG / ./nbody_config.yaml.
        paths = config["paths"]
        self.pathof: Dict[str, str] = paths.get("simulations") or {}
        self.plot_dir: Optional[str] = paths.get("plot_dir")

        self.analysis_cache_dir: Optional[str] = paths.get("analysis_cache_dir")
        self._legacy_cache_dir_suffix: Optional[str] = paths.get("cache_dir_suffix")
        self._input_files_of: Dict[str, str] = paths.get("input_files") or {}

        # Optional second storage root for the full particle-lake feature
        # store (see docs/analysis_architecture.md Roadmap #5). Unset by
        # default; lake-routed features fall back to analysis_cache_dir.
        self.lake_dir: Optional[str] = paths.get("lake_dir")

        # Optional standalone-cache path overrides (fall back to sensible
        # defaults in the consuming code when unset).
        self.teff_rgb_cache: Optional[str] = paths.get("teff_rgb_cache")
        self.gwtc_catalog_csv: Optional[str] = paths.get("gwtc_catalog_csv")

        # input_file_path_of is derived after user config merge (see
        # _setup_derived_attributes) since paths.simulations/input_files can both
        # be extended by the user config.
        self.input_file_path_of: Dict[str, str] = {}

        # Figure name prefixes
        self.figname_prefix: Dict[str, str] = config.get("figure_prefixes") or {}

        # Processing options
        proc = config["processing"]
        self.skip_until_of: Dict[str, Union[float, str]] = proc["skip_until"]
        self.skip_existing_plot: bool = proc["skip_existing_plot"]
        self.close_figure_in_ipython: bool = proc["close_figure_in_ipython"]
        self.processes_count: int = proc["processes_count"]
        self.tasks_per_child: int = proc["tasks_per_child"]
        self.mem_max_gb: float = proc["mem_max_gb"]
        self.inode_limit: int = proc["inode_limit"]
        self.hdf5: Dict[str, Any] = config["hdf5"]

        # Lagrangian radii
        self.selected_lagr_percent: List[str] = config["selected_lagr_percent"]
        self.current_lagrangian: Dict[str, Any] = config["current_lagrangian"]
        self.galactic_orbit: Dict[str, Any] = config["galactic_orbit"]
        self.galactic_energy_angular_momentum: Dict[str, Any] = config[
            "galactic_energy_angular_momentum"
        ]
        self.binary_stellar_type_extraction: Dict[str, Any] = config[
            "binary_stellar_type_extraction"
        ]
        self.compact_object_history: Dict[str, Any] = config["compact_object_history"]
        self.snapshot_summary: Dict[str, Any] = config["snapshot_summary"]
        self.particle_lake: Dict[str, Any] = config["particle_lake"]

        # Physics constants
        phys = config["physics"]
        self.ECLOSE_INPUT: float = phys["ECLOSE_INPUT"]
        self.universe_age_myr: float = phys["universe_age_myr"]
        self.IMBH_mass_range_msun: Tuple[float, float] = tuple(phys["IMBH_mass_range_msun"])
        self.extreme_mass_ratio_upper: float = phys["extreme_mass_ratio_upper"]
        self.high_BH_ecc_lower: float = phys["high_BH_ecc_lower"]
        self.PISNe_mass_gap: Tuple[int, int] = tuple(phys["PISNe_mass_gap"])

        # Stellar types
        self.kw_to_stellar_type: Dict[int, str] = {
            int(k): v for k, v in config["stellar_types"].items()
        }
        self.compact_object_KW: np.ndarray = np.array(config["compact_object_KW"])

        # Binary types
        self.wow_binary_st_list: List[str] = config["wow_binary_st_list"]

        # Limits
        self.limits: Dict[str, Tuple[float, float]] = config["limits"]

        # Column labels
        self.colname_to_label: Dict[str, str] = config["column_labels"]

        # Fixed width font context
        self.fixed_width_font_context: Dict[str, Any] = {"rc": {"font.family": "monospace"}}

    def _merge_user_config(self, config_path: str) -> None:
        """
        Merge user configuration with defaults.

        Args:
            config_path: Path to user YAML configuration file
        """
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        user_config = user_config or {}
        self._raise_for_removed_config_keys(user_config)

        # Simple deep merge - can be extended for more complex merging
        if "paths" in user_config:
            if "simulations" in user_config["paths"]:
                self.pathof.update(user_config["paths"]["simulations"])
            if "input_files" in user_config["paths"]:
                self._input_files_of.update(user_config["paths"]["input_files"])
            if "plot_dir" in user_config["paths"]:
                self.plot_dir = user_config["paths"]["plot_dir"]
            if "analysis_cache_dir" in user_config["paths"]:
                self.analysis_cache_dir = user_config["paths"]["analysis_cache_dir"]
            elif "cache_dir_suffix" in user_config["paths"] and not self.analysis_cache_dir:
                self._legacy_cache_dir_suffix = user_config["paths"]["cache_dir_suffix"]
            if "lake_dir" in user_config["paths"]:
                self.lake_dir = user_config["paths"]["lake_dir"]
            if "teff_rgb_cache" in user_config["paths"]:
                self.teff_rgb_cache = user_config["paths"]["teff_rgb_cache"]
            if "gwtc_catalog_csv" in user_config["paths"]:
                self.gwtc_catalog_csv = user_config["paths"]["gwtc_catalog_csv"]

        if "figure_prefixes" in user_config:
            self.figname_prefix.update(user_config["figure_prefixes"])

        if "processing" in user_config:
            proc = user_config["processing"]
            if "processes_count" in proc:
                self.processes_count = proc["processes_count"]
            if "tasks_per_child" in proc:
                self.tasks_per_child = proc["tasks_per_child"]
            if "skip_until" in proc:
                self.skip_until_of.update(proc["skip_until"])
            if "skip_existing_plot" in proc:
                self.skip_existing_plot = proc["skip_existing_plot"]
            if "close_figure_in_ipython" in proc:
                self.close_figure_in_ipython = proc["close_figure_in_ipython"]
            if "mem_max_gb" in proc:
                self.mem_max_gb = proc["mem_max_gb"]
            if "inode_limit" in proc:
                self.inode_limit = proc["inode_limit"]

        if "hdf5" in user_config:
            self._deep_update(self.hdf5, user_config["hdf5"])

        if "current_lagrangian" in user_config:
            self.current_lagrangian.update(user_config["current_lagrangian"])

        if "galactic_orbit" in user_config:
            self.galactic_orbit.update(user_config["galactic_orbit"])

        if "galactic_energy_angular_momentum" in user_config:
            self.galactic_energy_angular_momentum.update(
                user_config["galactic_energy_angular_momentum"]
            )

        if "binary_stellar_type_extraction" in user_config:
            self.binary_stellar_type_extraction.update(
                user_config["binary_stellar_type_extraction"]
            )

        if "compact_object_history" in user_config:
            self.compact_object_history.update(user_config["compact_object_history"])

        if "snapshot_summary" in user_config:
            self.snapshot_summary.update(user_config["snapshot_summary"])

        if "particle_lake" in user_config:
            self._deep_update(self.particle_lake, user_config["particle_lake"])

    def _raise_for_removed_config_keys(self, user_config: Dict[str, Any]) -> None:
        """Reject pre-1.0 scan configuration keys with migration guidance."""
        removed = []
        for key_path, new_path in REMOVED_CONFIG_KEYS.items():
            cursor: Any = user_config
            for key in key_path:
                if not isinstance(cursor, dict) or key not in cursor:
                    break
                cursor = cursor[key]
            else:
                removed.append((".".join(key_path), new_path))
        if removed:
            details = "; ".join(f"{old} -> {new}" for old, new in removed)
            raise ValueError(
                "Removed pre-1.0 HDF5 scan configuration key(s) found. "
                f"Migrate them to the new global hdf5 schema: {details}"
            )

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Recursively merge nested user configuration into defaults."""
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def _setup_derived_attributes(self) -> None:
        """Set up derived attributes that depend on configuration"""
        missing: List[str] = []
        if not self.pathof:
            missing.append("paths.simulations")
        if not self.plot_dir:
            missing.append("paths.plot_dir")
        if not self.analysis_cache_dir and not self._legacy_cache_dir_suffix:
            missing.append("paths.analysis_cache_dir")
        if missing:
            raise ValueError(
                "Missing required path configuration: "
                + ", ".join(missing)
                + ". Provide a user config (see config.example.yaml for a template) via "
                "--config, the NBODY_CONFIG environment variable, or ./nbody_config.yaml."
            )

        self.input_file_path_of = {
            simu_name: f"{self.pathof[simu_name]}/{self._input_files_of[simu_name]}"
            for simu_name in self.pathof
            if simu_name in self._input_files_of
        }

        if self.analysis_cache_dir:
            analysis_root = Path(self.analysis_cache_dir)
            self.analysis_cache_dir_of = {
                simu_name: str(analysis_root / simu_name) for simu_name in self.pathof
            }
            self.particle_df_cache_dir_of = {
                simu_name: str(Path(cache_dir) / "particle_df")
                for simu_name, cache_dir in self.analysis_cache_dir_of.items()
            }
        else:
            self.analysis_cache_dir_of = {
                simu_name: str(Path(path) / self._legacy_cache_dir_suffix.lstrip("/"))
                for simu_name, path in self.pathof.items()
            }
            self.particle_df_cache_dir_of = dict(self.analysis_cache_dir_of)

        self.lake_dir_of: Dict[str, str] = (
            {simu_name: str(Path(self.lake_dir) / simu_name) for simu_name in self.pathof}
            if self.lake_dir
            else {}
        )

        # Calculate memory capacity
        try:
            total_mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except Exception:
            total_mem_bytes = self.mem_max_gb * 1024**3

        # mem_cap_bytes = min(system memory, mem_max_gb) / 2
        # Division by 2 because data will double when extracting particle_df
        self.mem_cap_bytes: int = min(int(total_mem_bytes), int(self.mem_max_gb * 1024**3)) // 2

        # Reverse mappings
        self.stellar_type_to_kw: Dict[str, int] = {v: k for k, v in self.kw_to_stellar_type.items()}

        self.kw_to_stellar_type_verbose: Dict[int, str] = {
            k: f"{k:2d}:{v}" for k, v in self.kw_to_stellar_type.items()
        }

        self.stellar_type_verbose_to_kw: Dict[str, int] = {
            v: k for k, v in self.kw_to_stellar_type_verbose.items()
        }

        # Plotting styles
        _marker_fill_list = ["d", "v", "^", "<", ">", "h", "8", "s", "p", "H", "D", "o"]
        self.marker_fill_list = [
            "o",
        ] * (17 - len(_marker_fill_list)) + _marker_fill_list
        self.marker_nofill_list = (["1", "+", "3", "x"] * 5)[: len(self.kw_to_stellar_type)]

        self.star_type_verbose_to_marker: Dict[str, str] = dict(
            zip(list(self.kw_to_stellar_type_verbose.values()), self.marker_nofill_list)
        )

        # Color palettes
        self.palette_st = (
            sns.color_palette(n_colors=10)
            + sns.color_palette("husl", 7)[::2]
            + sns.color_palette("husl", 7)[1::2][:-3]
            + [(1, 0, 0), (0, 0, 0), (0.8, 0.8, 0.8)]
        )

        self.st_verbose_to_color: Dict[str, Tuple[float, float, float]] = {
            k: self.palette_st[v + 1] for k, v in self.stellar_type_verbose_to_kw.items()
        }

    def _parse_argv(self, opts: List[Tuple[str, str]]) -> None:
        """
        Parse command line arguments and update configuration.

        Args:
            opts: List of (option, argument) tuples from command line parsing

        Example:
            --skip-until=0: start from t=0 (first file)
            --skip-until=last: read last timestamp from existing plots
        """
        for opt, arg in opts:
            if opt == "--skip-until":
                if can_convert_to_float(arg):
                    arg = float(arg)
                for key in self.skip_until_of:
                    self.skip_until_of[key] = arg

    def _resolve_skip_until_last(self) -> None:
        """
        Resolve 'last' values in skip_until by finding maximum time from existing plots.
        """
        for simu_name in self.pathof.keys():
            if self.skip_until_of.get(simu_name) == "last":
                pattern = f"{self.plot_dir}/jpg/{self.figname_prefix[simu_name]}*ttot_*.jpg"
                all_jpg_plots = glob(pattern)

                if all_jpg_plots:
                    # Extract time from filename
                    def get_time(path: str) -> float:
                        return float(path.split("ttot_")[1].split("_")[0])

                    all_times = np.array([get_time(x) for x in all_jpg_plots])
                    self.skip_until_of[simu_name] = float(all_times.max())
                    logger.info(f"[{simu_name}] Set skip-until={self.skip_until_of[simu_name]}")
                else:
                    self.skip_until_of[simu_name] = 0
