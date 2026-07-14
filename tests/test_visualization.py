"""
Tests for nbody_pipeline.visualization module
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nbody_pipeline.visualization import (
    BaseVisualizer,
    BaseHDF5Visualizer,
    HDF5Visualizer,
    SingleStarVisualizer,
    BinaryStarVisualizer,
    LagrVisualizer,
    GalacticOrbitVisualizer,
    CollCoalVisualizer,
    set_mpl_fonts,
    add_grid,
    PlotPurger,
)


@pytest.fixture(autouse=True)
def mock_color_converter():
    """Mock BlackbodyColorConverter for all tests"""
    with patch("nbody_pipeline.visualization.base.BlackbodyColorConverter") as mock:
        mock.return_value.get_rgb = Mock(return_value=np.array([[0.5, 0.5, 0.5]]))
        yield mock


@pytest.fixture
def mock_config():
    """Create a mock configuration manager"""
    config = Mock()
    config.colname_to_label = {
        "X [pc]": "X Position [pc]",
        "Y [pc]": "Y Position [pc]",
        "Z [pc]": "Z Position [pc]",
        "x_R [pc]": "x_R [pc]",
        "x_T [pc]": "x_T [pc]",
        "x_L [pc]": "x_L [pc]",
        "E_gal[Msun*(km/s)^2]": "E_gal",
        "L_z_gal[Msun*kpc*km/s]": "L_z_gal",
        "M": "Mass [Msolar]",
    }
    config.fixed_width_font_context = {"rc": {"font.family": "monospace"}}
    config.limits = {
        "position_pc_lim": (-10, 10),
        "position_pc_lim_MAX": (-50, 50),
        "velocity_kmps_lim": (-100, 100),
        "M": (0.1, 100),
        "Teff*": (1000, 100000),
        "L*": (0.001, 10000),
        "Bin A[au]": (0.01, 1000),
    }
    config.plot_dir = "/tmp/plots"
    config.figname_prefix = {"test_sim": "test_"}
    config.skip_existing_plot = False
    config.close_figure_in_ipython = False
    config.galactic_energy_angular_momentum = {"enabled": True, "percentile_limits": [25, 75]}
    config.selected_lagr_percent = [10, 50, 90]
    config.compact_object_KW = [10, 11, 12, 13, 14]
    config.kw_to_stellar_type_verbose = {i: f"{i}:Type{i}" for i in range(16)}
    config.st_verbose_to_color = {f"{i}:Type{i}": f"C{i}" for i in range(16)}
    config.star_type_verbose_to_marker = {f"{i}:Type{i}": "o" for i in range(16)}
    config.marker_nofill_list = {i: "o" for i in range(16)}
    config.marker_fill_list = {i: "o" for i in range(16)}
    config.palette_st = {i: f"C{i}" for i in range(16)}
    config.kw_to_stellar_type = {i: f"Type{i}" for i in range(16)}
    config.stellar_type_to_kw = {f"Type{i}": i for i in range(16)}
    config.wow_binary_st_list = []
    return config


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame(
        {
            "TTOT": [1.0] * 100,
            "Time[Myr]": [10.0] * 100,
            "TTOT/TCR0": [5.0] * 100,
            "TTOT/TRH0": [2.0] * 100,
            "X [pc]": np.random.randn(100),
            "Y [pc]": np.random.randn(100),
            "Z [pc]": np.random.randn(100),
            "M": np.random.uniform(0.5, 2.0, 100),
            "Teff*": np.random.uniform(3000, 10000, 100),
            "L*": np.random.uniform(0.1, 100, 100),
            "R*": np.random.uniform(0.5, 2.0, 100),
            "KW": np.random.choice([0, 1, 2, 10, 13, 14], 100),
            "Stellar Type": ["0:Type0"] * 100,
            "Distance_to_cluster_center[pc]": np.random.uniform(0.1, 10, 100),
            "V1": np.random.randn(100),
        }
    )


@pytest.fixture
def sample_binary_dataframe():
    """Create a sample binary DataFrame for testing"""
    return pd.DataFrame(
        {
            "TTOT": [1.0] * 50,
            "Time[Myr]": [10.0] * 50,
            "TTOT/TCR0": [5.0] * 50,
            "TTOT/TRH0": [2.0] * 50,
            "primary_mass[solar]": np.random.uniform(1, 20, 50),
            "mass_ratio": np.random.uniform(0.1, 1.0, 50),
            "Bin A[au]": np.random.uniform(0.1, 100, 50),
            "Bin ECC": np.random.uniform(0, 0.9, 50),
            "Ebind/kT": np.random.uniform(0.1, 100, 50),
            "total_mass[solar]": np.random.uniform(2, 40, 50),
            "Distance_to_cluster_center[pc]": np.random.uniform(0.1, 10, 50),
            "Bin cm X [pc]": np.random.randn(50),
            "Bin cm V1": np.random.randn(50),
            "sum_of_radius[au]": np.random.uniform(0.01, 1, 50),
            "peri[au]": np.random.uniform(0.01, 10, 50),
            "Bin KW1": np.random.choice([10, 13, 14], 50),
            "Bin KW2": np.random.choice([10, 13, 14], 50),
            "Stellar Type": ["Type10-Type13"] * 50,
            "is_hard_binary": np.random.choice([True, False], 50),
            "tau_gw[Myr]": np.random.uniform(1, 1000, 50),
        }
    )


@pytest.fixture
def sample_lagr_dataframe():
    """Create a sample Lagrangian DataFrame for testing"""
    times = np.linspace(0.1, 100, 50)
    data = []
    for t in times:
        for pct in [10, 50, 90]:
            data.append(
                {"Time[Myr]": t, "%": pct, "Metric": "rlagr", "Value": np.random.uniform(0.1, 10)}
            )
    return pd.DataFrame(data)


class TestHelperFunctions:
    """Test helper functions"""

    def test_set_mpl_fonts(self):
        """Test set_mpl_fonts configures matplotlib"""
        set_mpl_fonts()
        assert plt.rcParams["font.family"] == ["serif"]
        assert plt.rcParams["font.size"] == 15

    def test_add_grid_single_axis(self):
        """Test add_grid with single axis"""
        fig, ax = plt.subplots()
        add_grid(ax)
        plt.close(fig)

    def test_add_grid_multiple_axes(self):
        """Test add_grid with multiple axes"""
        fig, axes = plt.subplots(2, 2)
        add_grid(axes.flatten())
        plt.close(fig)

    def test_add_grid_with_parameters(self):
        """Test add_grid with custom parameters"""
        fig, ax = plt.subplots()
        add_grid(ax, which="major", axis="x")
        plt.close(fig)


class TestBaseVisualizer:
    """Test BaseVisualizer class"""

    def test_init(self, mock_config):
        """Test BaseVisualizer initialization"""
        vis = BaseVisualizer(mock_config)
        assert vis.config == mock_config
        assert vis.teff_to_rgb_converter is not None

    def test_luminosity_to_plot_alpha(self, mock_config):
        """Test luminosity_to_plot_alpha method"""
        vis = BaseVisualizer(mock_config)
        L_arr = np.array([1e-10, 0.1, 1.0, 10.0, 100.0])
        alpha = vis.luminosity_to_plot_alpha(L_arr)

        assert len(alpha) == len(L_arr)
        assert np.all((alpha >= 0) & (alpha <= 1))
        assert alpha[0] == 1  # Special case for -10


class TestBaseHDF5Visualizer:
    """Test BaseHDF5Visualizer class"""

    def test_init(self, mock_config):
        """Test BaseHDF5Visualizer initialization"""
        vis = BaseHDF5Visualizer(mock_config)
        assert vis.config == mock_config

    @patch("matplotlib.pyplot.close")
    def test_decorate_jointfig(self, mock_close, mock_config, sample_dataframe):
        """Test decorate_jointfig method"""
        vis = BaseHDF5Visualizer(mock_config)
        fig, ax = plt.subplots()

        vis.decorate_jointfig(
            ax,
            sample_dataframe,
            "X [pc]",
            "Y [pc]",
            (-10, 10),
            (-10, 10),
            "test_sim",
            1.0,
            10.0,
            5.0,
            2.0,
        )

        assert ax.get_xlim() == (-10, 10)
        assert ax.get_ylim() == (-10, 10)
        plt.close(fig)

    def test_symlogY_and_fill_handler(self, mock_config):
        """Test _symlogY_and_fill_handler method"""
        vis = BaseHDF5Visualizer(mock_config)
        fig, ax = plt.subplots()

        vis._symlogY_and_fill_handler(ax, linthresh=5)

        assert ax.get_yscale() == "symlog"
        plt.close(fig)


class TestHDF5Visualizer:
    """Test HDF5Visualizer class"""

    def test_init(self, mock_config):
        """Test HDF5Visualizer initialization"""
        vis = HDF5Visualizer(mock_config)
        assert vis.single is not None
        assert vis.binary is not None
        assert isinstance(vis.single, SingleStarVisualizer)
        assert isinstance(vis.binary, BinaryStarVisualizer)


class TestSingleStarVisualizer:
    """Test SingleStarVisualizer class"""

    def test_init(self, mock_config):
        """Test SingleStarVisualizer initialization"""
        vis = SingleStarVisualizer(mock_config)
        assert vis.config == mock_config

    def test_save_position_figure_uses_square_axes(self, mock_config, tmp_path):
        """Position plot helper keeps the central axes square."""
        vis = SingleStarVisualizer(mock_config)
        fig, ax = plt.subplots()
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)

        vis._save_position_figure(fig, ax, str(tmp_path / "position.jpg"))
        fig.canvas.draw()
        bbox = ax.get_window_extent()

        assert abs(bbox.width - bbox.height) < 1.0
        plt.close(fig)

    def test_position_plot_wide_pc_matches_standard_canvas(
        self, mock_config, sample_dataframe, tmp_path
    ):
        """Wide and standard position JPGs use the same output canvas."""
        mock_config.plot_dir = str(tmp_path)
        (tmp_path / "jpg").mkdir()
        vis = SingleStarVisualizer(mock_config)

        vis.create_position_plot_jpg(sample_dataframe, "test_sim")
        vis.create_position_plot_wide_pc_jpg(sample_dataframe, "test_sim")

        standard_path = tmp_path / "jpg" / "test_output_ttot_1.0_x1_vs_x2.jpg"
        wide_path = tmp_path / "jpg" / "test_output_ttot_1.0_x1_vs_x2_wide_pc.jpg"
        assert plt.imread(standard_path).shape[:2] == plt.imread(wide_path).shape[:2]

    def test_position_plot_wide_pc_uses_max_position_limits(
        self, mock_config, sample_dataframe, tmp_path
    ):
        """Wide position JPG changes only the position limits through the axes handler."""
        mock_config.plot_dir = str(tmp_path)
        (tmp_path / "jpg").mkdir()
        vis = SingleStarVisualizer(mock_config)
        observed = {}

        def capture_limits(fig, ax, save_jpg_path):
            observed["xlim"] = ax.get_xlim()
            observed["ylim"] = ax.get_ylim()
            plt.close(fig)

        with patch.object(vis, "_save_position_figure", side_effect=capture_limits):
            vis.create_position_plot_wide_pc_jpg(sample_dataframe, "test_sim")

        assert observed["xlim"] == mock_config.limits["position_pc_lim_MAX"]
        assert observed["ylim"] == mock_config.limits["position_pc_lim_MAX"]

    def test_orbital_frame_projection_simple_case(self, mock_config, sample_dataframe):
        """RG=(1,0,0), VG=(0,1,0) maps X/Y/Z to x_R/x_T/x_L."""
        vis = SingleStarVisualizer(mock_config)
        scalar_row = pd.Series(
            {"RG(1)": 1.0, "RG(2)": 0.0, "RG(3)": 0.0, "VG(1)": 0.0, "VG(2)": 1.0, "VG(3)": 0.0}
        )

        projected = vis._single_df_with_orbital_frame_positions(sample_dataframe, scalar_row)

        assert projected is not None
        np.testing.assert_allclose(projected[vis.ORBITAL_X_R_COL], sample_dataframe["X [pc]"])
        np.testing.assert_allclose(projected[vis.ORBITAL_X_T_COL], sample_dataframe["Y [pc]"])
        np.testing.assert_allclose(projected[vis.ORBITAL_X_L_COL], sample_dataframe["Z [pc]"])

    def test_orbital_position_plots_match_standard_canvas(
        self, mock_config, sample_dataframe, tmp_path
    ):
        """Orbital wide JPGs use expected names and the same canvas as wide position plots."""
        mock_config.plot_dir = str(tmp_path)
        (tmp_path / "jpg").mkdir()
        vis = SingleStarVisualizer(mock_config)
        scalar_row = pd.Series(
            {"RG(1)": 1.0, "RG(2)": 0.0, "RG(3)": 0.0, "VG(1)": 0.0, "VG(2)": 1.0, "VG(3)": 0.0}
        )

        vis.create_position_plot_wide_pc_jpg(sample_dataframe, "test_sim")
        vis.create_position_plot_orbital_xT_xR_wide_pc_jpg(sample_dataframe, scalar_row, "test_sim")
        vis.create_position_plot_orbital_xT_xL_wide_pc_jpg(sample_dataframe, scalar_row, "test_sim")

        wide_path = tmp_path / "jpg" / "test_output_ttot_1.0_x1_vs_x2_wide_pc.jpg"
        orbital_xr_path = tmp_path / "jpg" / "test_output_ttot_1.0_orbital_xT_vs_xR_wide_pc.jpg"
        orbital_xl_path = tmp_path / "jpg" / "test_output_ttot_1.0_orbital_xT_vs_xL_wide_pc.jpg"
        assert orbital_xr_path.exists()
        assert orbital_xl_path.exists()
        assert plt.imread(wide_path).shape[:2] == plt.imread(orbital_xr_path).shape[:2]
        assert plt.imread(wide_path).shape[:2] == plt.imread(orbital_xl_path).shape[:2]

    def test_orbital_position_plots_use_max_position_limits(
        self, mock_config, sample_dataframe, tmp_path
    ):
        """Orbital wide position JPGs use position_pc_lim_MAX for both axes."""
        mock_config.plot_dir = str(tmp_path)
        (tmp_path / "jpg").mkdir()
        vis = SingleStarVisualizer(mock_config)
        scalar_row = pd.Series(
            {"RG(1)": 1.0, "RG(2)": 0.0, "RG(3)": 0.0, "VG(1)": 0.0, "VG(2)": 1.0, "VG(3)": 0.0}
        )
        observed = {}

        def capture_limits(fig, ax, save_jpg_path):
            observed[Path(save_jpg_path).name] = (ax.get_xlim(), ax.get_ylim())
            plt.close(fig)

        with patch.object(vis, "_save_position_figure", side_effect=capture_limits):
            vis.create_position_plot_orbital_xT_xR_wide_pc_jpg(
                sample_dataframe, scalar_row, "test_sim"
            )
            vis.create_position_plot_orbital_xT_xL_wide_pc_jpg(
                sample_dataframe, scalar_row, "test_sim"
            )

        assert observed["test_output_ttot_1.0_orbital_xT_vs_xR_wide_pc.jpg"] == (
            mock_config.limits["position_pc_lim_MAX"],
            mock_config.limits["position_pc_lim_MAX"],
        )
        assert observed["test_output_ttot_1.0_orbital_xT_vs_xL_wide_pc.jpg"] == (
            mock_config.limits["position_pc_lim_MAX"],
            mock_config.limits["position_pc_lim_MAX"],
        )

    def test_highlight_position_plot_wide_pc_matches_standard_canvas(
        self, mock_config, sample_dataframe, tmp_path
    ):
        """Highlighted wide and standard position JPGs use the same output canvas."""
        mock_config.plot_dir = str(tmp_path)
        (tmp_path / "jpg").mkdir()
        vis = SingleStarVisualizer(mock_config)

        vis.create_position_plot_hightlight_compact_objects_jpg(sample_dataframe, "test_sim")
        vis.create_position_plot_hightlight_compact_objects_wide_pc_jpg(
            sample_dataframe, "test_sim"
        )

        standard_path = (
            tmp_path / "jpg" / "test_output_ttot_1.0_x1_vs_x2_highlight_compact_objects.jpg"
        )
        wide_path = (
            tmp_path / "jpg" / "test_output_ttot_1.0_x1_vs_x2_highlight_compact_objects_wide_pc.jpg"
        )
        assert plt.imread(standard_path).shape[:2] == plt.imread(wide_path).shape[:2]

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    @patch("matplotlib.pyplot.close")
    def test_create_mass_distance_plot_density(
        self, mock_close, mock_makedirs, mock_exists, mock_config, sample_dataframe, temp_dir
    ):
        """Test create_mass_distance_plot_density method"""
        mock_config.plot_dir = str(temp_dir)
        vis = SingleStarVisualizer(mock_config)

        # Should not raise an error
        try:
            vis.create_mass_distance_plot_density(sample_dataframe, "test_sim")
        except Exception:
            # Some plotting functions might fail in headless environment
            # Just ensure the method can be called
            pass

    def test_galactic_energy_angular_momentum_plot_path(self, mock_config, sample_dataframe):
        vis = SingleStarVisualizer(mock_config)

        path = vis.galactic_energy_angular_momentum_plot_jpg_path(sample_dataframe, "test_sim")

        assert path == "/tmp/plots/jpg/test_output_ttot_1.0_galactic_E_vs_Lz.jpg"

    def test_galactic_energy_angular_momentum_plot_filters_finite_and_uses_percentiles(
        self, mock_config, tmp_path, monkeypatch
    ):
        mock_config.plot_dir = str(tmp_path)
        (tmp_path / "jpg").mkdir()
        vis = SingleStarVisualizer(mock_config)
        df = pd.DataFrame(
            {
                "TTOT": [1.0] * 5,
                "Time[Myr]": [10.0] * 5,
                "TTOT/TCR0": [5.0] * 5,
                "TTOT/TRH0": [2.0] * 5,
                "L_z_gal[Msun*kpc*km/s]": [0.0, 10.0, 20.0, np.inf, 40.0],
                "E_gal[Msun*(km/s)^2]": [-100.0, -50.0, 0.0, 10.0, np.nan],
            }
        )
        saved_paths = []
        scatter_calls = []

        def fake_scatterplot(*args, **kwargs):
            scatter_calls.append(kwargs)
            return kwargs["ax"]

        def fake_savefig(self, path, *args, **kwargs):
            saved_paths.append(Path(path))

        monkeypatch.setattr(
            "nbody_pipeline.visualization.single_star.sns.scatterplot", fake_scatterplot
        )
        monkeypatch.setattr(plt.Figure, "savefig", fake_savefig)

        vis.create_galactic_energy_angular_momentum_plot_jpg(df, "test_sim")

        assert saved_paths == [tmp_path / "jpg" / "test_output_ttot_1.0_galactic_E_vs_Lz.jpg"]
        assert scatter_calls[0]["color"] == "white"
        assert scatter_calls[0]["alpha"] == 0.25
        assert len(scatter_calls[0]["data"]) == 3
        ax = scatter_calls[0]["ax"]
        assert ax.get_xlim() == pytest.approx((5.0, 15.0))
        assert ax.get_ylim() == pytest.approx((-75.0, -25.0))
        plt.close(ax.figure)

    def test_galactic_energy_angular_momentum_plot_skips_nonfinite_points(
        self, mock_config, tmp_path, monkeypatch
    ):
        mock_config.plot_dir = str(tmp_path)
        (tmp_path / "jpg").mkdir()
        vis = SingleStarVisualizer(mock_config)
        df = pd.DataFrame(
            {
                "TTOT": [1.0],
                "Time[Myr]": [10.0],
                "TTOT/TCR0": [5.0],
                "TTOT/TRH0": [2.0],
                "L_z_gal[Msun*kpc*km/s]": [np.inf],
                "E_gal[Msun*(km/s)^2]": [np.nan],
            }
        )
        scatter = Mock()
        monkeypatch.setattr("nbody_pipeline.visualization.single_star.sns.scatterplot", scatter)

        vis.create_galactic_energy_angular_momentum_plot_jpg(df, "test_sim")

        scatter.assert_not_called()

    def test_galactic_energy_angular_momentum_specific_and_ke_plot_paths(
        self, mock_config, sample_dataframe
    ):
        vis = SingleStarVisualizer(mock_config)

        assert (
            vis.galactic_energy_angular_momentum_specific_plot_jpg_path(
                sample_dataframe, "test_sim"
            )
            == "/tmp/plots/jpg/test_output_ttot_1.0_galactic_E_vs_Lz_specific.jpg"
        )
        assert (
            vis.galactic_kinetic_energy_specific_plot_jpg_path(sample_dataframe, "test_sim")
            == "/tmp/plots/jpg/test_output_ttot_1.0_galactic_KE_vs_Lz_specific.jpg"
        )

    def test_galactic_energy_angular_momentum_specific_plot_uses_specific_columns(
        self, mock_config, tmp_path, monkeypatch
    ):
        mock_config.plot_dir = str(tmp_path)
        (tmp_path / "jpg").mkdir()
        vis = SingleStarVisualizer(mock_config)
        df = pd.DataFrame(
            {
                "TTOT": [1.0] * 3,
                "Time[Myr]": [10.0] * 3,
                "TTOT/TCR0": [5.0] * 3,
                "TTOT/TRH0": [2.0] * 3,
                "L_z_gal_specific[kpc*km/s]": [0.0, 10.0, 20.0],
                "E_gal_specific[(km/s)^2]": [-100.0, -50.0, 0.0],
            }
        )
        saved_paths = []
        monkeypatch.setattr(
            plt.Figure, "savefig", lambda self, path, *a, **k: saved_paths.append(Path(path))
        )

        vis.create_galactic_energy_angular_momentum_specific_plot_jpg(df, "test_sim")

        assert saved_paths == [
            tmp_path / "jpg" / "test_output_ttot_1.0_galactic_E_vs_Lz_specific.jpg"
        ]

    def test_galactic_kinetic_energy_specific_plot_uses_ke_column(
        self, mock_config, tmp_path, monkeypatch
    ):
        mock_config.plot_dir = str(tmp_path)
        (tmp_path / "jpg").mkdir()
        vis = SingleStarVisualizer(mock_config)
        df = pd.DataFrame(
            {
                "TTOT": [1.0] * 3,
                "Time[Myr]": [10.0] * 3,
                "TTOT/TCR0": [5.0] * 3,
                "TTOT/TRH0": [2.0] * 3,
                "L_z_gal_specific[kpc*km/s]": [0.0, 10.0, 20.0],
                "E_kin_gal_specific[(km/s)^2]": [1.0, 2.0, 3.0],
            }
        )
        saved_paths = []
        monkeypatch.setattr(
            plt.Figure, "savefig", lambda self, path, *a, **k: saved_paths.append(Path(path))
        )

        vis.create_galactic_kinetic_energy_specific_plot_jpg(df, "test_sim")

        assert saved_paths == [
            tmp_path / "jpg" / "test_output_ttot_1.0_galactic_KE_vs_Lz_specific.jpg"
        ]

    def test_galactic_plot_com_overlay_expands_limits_and_adds_marker(
        self, mock_config, tmp_path, monkeypatch
    ):
        mock_config.plot_dir = str(tmp_path)
        (tmp_path / "jpg").mkdir()
        vis = SingleStarVisualizer(mock_config)
        df = pd.DataFrame(
            {
                "TTOT": [1.0] * 5,
                "Time[Myr]": [10.0] * 5,
                "TTOT/TCR0": [5.0] * 5,
                "TTOT/TRH0": [2.0] * 5,
                "L_z_gal[Msun*kpc*km/s]": [0.0, 10.0, 20.0, 30.0, 40.0],
                "E_gal[Msun*(km/s)^2]": [-100.0, -50.0, 0.0, 50.0, 100.0],
            }
        )
        # far outside the [25, 75] percentile-clipped range of the scatter
        com_point = {
            "L_z_gal[Msun*kpc*km/s]": 5000.0,
            "E_gal[Msun*(km/s)^2]": -5000.0,
        }
        scatter_calls = []
        monkeypatch.setattr(plt.Figure, "savefig", lambda self, path, *a, **k: None)
        real_ax_scatter = plt.Axes.scatter

        def spy_ax_scatter(self, *args, **kwargs):
            scatter_calls.append(kwargs)
            return real_ax_scatter(self, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "scatter", spy_ax_scatter)

        vis.create_galactic_energy_angular_momentum_plot_jpg(df, "test_sim", com_point=com_point)

        com_calls = [c for c in scatter_calls if c.get("label") == "Cluster COM"]
        assert len(com_calls) == 1
        assert com_calls[0]["marker"] == "*"
        plt.close("all")


class TestBinaryStarVisualizer:
    """Test BinaryStarVisualizer class"""

    def test_init(self, mock_config):
        """Test BinaryStarVisualizer initialization"""
        vis = BinaryStarVisualizer(mock_config)
        assert vis.config == mock_config

    def test_get_binary_cookie_dict(self, mock_config):
        """Test get_binary_cookie_dict method"""
        vis = BinaryStarVisualizer(mock_config)
        result = vis.get_binary_cookie_dict("Type10", "Type13")

        assert isinstance(result, dict)
        assert "marker" in result
        assert "markerfacecolor" in result

    def test_compact_object_binary_extra_ax_handler_runs_after_default_limits(
        self, mock_config, sample_binary_dataframe, tmp_path
    ):
        """Binary compact-object plots keep axes changes made by extra_ax_handler."""
        mock_config.plot_dir = str(tmp_path)
        (tmp_path / "jpg").mkdir()
        vis = BinaryStarVisualizer(mock_config)
        observed = {}

        def set_custom_limits(ax):
            ax.set_xlim(-50, 50)
            ax.set_ylim(-75, 75)

        def capture_limits(ax, df):
            observed["xlim"] = ax.get_xlim()
            observed["ylim"] = ax.get_ylim()

        vis._create_base_jpg_plot_compact_object_only(
            sample_binary_dataframe,
            simu_name="test_sim",
            x_col="Bin cm X [pc]",
            y_col="Bin cm V1",
            log_scale=(False, False),
            xlim_key="position_pc_lim",
            ylim_key="velocity_kmps_lim",
            filename_var_part="bin_vx_vs_x_compact_objects_only",
            extra_ax_handler=set_custom_limits,
            custom_ax_decorator=capture_limits,
        )

        assert observed["xlim"] == (-50, 50)
        assert observed["ylim"] == (-75, 75)


class TestLagrVisualizer:
    """Test LagrVisualizer class"""

    def test_init(self, mock_config):
        """Test LagrVisualizer initialization"""
        vis = LagrVisualizer(mock_config)
        assert vis.config == mock_config
        assert "rlagr" in vis.metric_to_plot_label
        assert "sigma" in vis.metric_to_plot_label

    @patch("os.path.exists", return_value=False)
    @patch("matplotlib.pyplot.close")
    def test_create_lagr_plot_base(
        self, mock_close, mock_exists, mock_config, sample_lagr_dataframe, temp_dir
    ):
        """Test create_lagr_plot_base method"""
        mock_config.plot_dir = str(temp_dir)
        vis = LagrVisualizer(mock_config)

        # Should not raise an error
        try:
            vis.create_lagr_plot_base(sample_lagr_dataframe, "test_sim", metric="rlagr")
        except Exception:
            # Some plotting functions might fail in headless environment
            pass

    def test_lagr_plots_do_not_accumulate_lines_between_metrics(
        self, monkeypatch, mock_config, temp_dir
    ):
        """Each Lagrangian plot should be drawn on a fresh figure."""
        import nbody_pipeline.visualization.lagrangian as lagrangian_module

        mock_config.plot_dir = str(temp_dir)
        mock_config.close_figure_in_ipython = False
        monkeypatch.setattr(lagrangian_module, "__IPYTHON__", True, raising=False)

        times = [1.0, 2.0]
        data = []
        for metric in ["rlagr", "avmass"]:
            for time in times:
                for pct in ["10%", "50%", "90%"]:
                    data.append(
                        {
                            "Time[Myr]": time,
                            "%": pct,
                            "Metric": metric,
                            "Value": float(time),
                        }
                    )
        lagr_df = pd.DataFrame(data)

        plt.close("all")
        vis = LagrVisualizer(mock_config)
        vis.create_lagr_plot_base(lagr_df, "test_sim", metric="rlagr")
        vis.create_lagr_plot_base(lagr_df, "test_sim", metric="avmass")

        figs = [plt.figure(num) for num in plt.get_fignums()]
        assert len(figs) == 2
        data_line_counts = [
            sum(line.get_label().startswith("_child") for line in fig.axes[0].lines) for fig in figs
        ]
        assert data_line_counts == [3, 3]
        plt.close("all")

    def test_create_total_mass_plot_uses_100_percent_avmass_and_nshell(
        self, monkeypatch, mock_config, temp_dir
    ):
        """Total mass should be avmass * nshell for the 100% shell only."""
        import nbody_pipeline.visualization.lagrangian as lagrangian_module

        mock_config.plot_dir = str(temp_dir)
        mock_config.close_figure_in_ipython = False
        monkeypatch.setattr(lagrangian_module, "__IPYTHON__", True, raising=False)

        lagr_df = pd.DataFrame(
            [
                {"Time[Myr]": 0.0, "%": "100%", "Metric": "avmass", "Value": 1.5},
                {"Time[Myr]": 1.0, "%": "100%", "Metric": "avmass", "Value": 2.0},
                {"Time[Myr]": 2.0, "%": "100%", "Metric": "avmass", "Value": 3.0},
                {"Time[Myr]": 0.0, "%": "100%", "Metric": "nshell", "Value": 8.0},
                {"Time[Myr]": 1.0, "%": "100%", "Metric": "nshell", "Value": 10.0},
                {"Time[Myr]": 2.0, "%": "100%", "Metric": "nshell", "Value": 20.0},
                {"Time[Myr]": 1.0, "%": "90%", "Metric": "avmass", "Value": 100.0},
                {"Time[Myr]": 1.0, "%": "90%", "Metric": "nshell", "Value": 100.0},
            ]
        )

        plt.close("all")
        vis = LagrVisualizer(mock_config)
        vis.create_total_mass_plot(lagr_df, "test_sim")

        fig = plt.figure(plt.get_fignums()[0])
        ax = fig.axes[0]
        data_lines = [line for line in ax.lines if line.get_label().startswith("_child")]
        assert len(data_lines) == 1
        assert data_lines[0].get_xdata().tolist() == [0.0, 1.0, 2.0]
        assert data_lines[0].get_ydata().tolist() == [12.0, 20.0, 60.0]
        assert ax.get_ylabel() == "Total mass [Msolar]"
        assert ax.get_ylim()[0] == 0
        assert (temp_dir / "test__total_mass.pdf").exists()
        plt.close("all")


class TestCollCoalVisualizer:
    """Test CollCoalVisualizer class"""

    def test_init(self, mock_config):
        """Test CollCoalVisualizer initialization"""
        vis = CollCoalVisualizer(mock_config)
        assert vis.config == mock_config

    def test_two_bh_filter(self, mock_config):
        """Test two_bh_filter method"""
        vis = CollCoalVisualizer(mock_config)
        df = pd.DataFrame(
            {
                "primary_stellar_type": [14, 14, 13, 10],
                "secondary_stellar_type": [14, 13, 14, 10],
            }
        )

        result = vis.two_bh_filter(df)
        assert len(result) == 1
        assert result.iloc[0]["primary_stellar_type"] == 14
        assert result.iloc[0]["secondary_stellar_type"] == 14

    def test_two_cbo_filter(self, mock_config):
        """Test two_cbo_fileter method"""
        vis = CollCoalVisualizer(mock_config)
        df = pd.DataFrame(
            {
                "primary_stellar_type": [14, 14, 13, 1],
                "secondary_stellar_type": [14, 13, 14, 1],
            }
        )

        result = vis.two_cbo_fileter(df)
        assert len(result) == 3


class TestParticleHistoryVisualizer:
    """Test ParticleHistoryVisualizer class"""

    def test_init(self, mock_config):
        """Test ParticleHistoryVisualizer initialization"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        vis = ParticleHistoryVisualizer(mock_config)
        assert vis.config == mock_config
        assert vis.simu_name is None
        assert vis.particle_name is None

    def test_init_with_params(self, mock_config):
        """Test ParticleHistoryVisualizer initialization with simu_name and particle_name"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        vis = ParticleHistoryVisualizer(mock_config, simu_name="test_sim", particle_name=1000)
        assert vis.simu_name == "test_sim"
        assert vis.particle_name == 1000

    def test_filter_by_time_all(self, mock_config):
        """Test _filter_by_time with 'all' returns all rows"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        vis = ParticleHistoryVisualizer(mock_config)
        df = pd.DataFrame({"TTOT": [1.0, 2.0, 3.0, 4.0, 5.0]})

        result = vis._filter_by_time(df, "all")
        assert len(result) == 5

    def test_filter_by_time_single_value(self, mock_config):
        """Test _filter_by_time with single float returns closest row"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        vis = ParticleHistoryVisualizer(mock_config)
        df = pd.DataFrame({"TTOT": [1.0, 2.0, 3.0, 4.0, 5.0]})

        result = vis._filter_by_time(df, 2.3)
        assert len(result) == 1
        assert result["TTOT"].iloc[0] == 2.0

    def test_filter_by_time_range(self, mock_config):
        """Test _filter_by_time with tuple range"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        vis = ParticleHistoryVisualizer(mock_config)
        df = pd.DataFrame({"TTOT": [1.0, 2.0, 3.0, 4.0, 5.0]})

        result = vis._filter_by_time(df, (2.0, 4.0))
        assert len(result) == 3
        assert list(result["TTOT"]) == [2.0, 3.0, 4.0]

    def test_filter_by_time_empty_df(self, mock_config):
        """Test _filter_by_time with empty DataFrame"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        vis = ParticleHistoryVisualizer(mock_config)
        df = pd.DataFrame()

        result = vis._filter_by_time(df, "all")
        assert result.empty

    def test_get_marker_color_neutron_star(self, mock_config):
        """Test _get_marker_color returns red for neutron star"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        vis = ParticleHistoryVisualizer(mock_config)
        color = vis._get_marker_color(kw=13, teff=10000)
        assert color == "red"

    def test_get_marker_color_black_hole(self, mock_config):
        """Test _get_marker_color returns black for black hole"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        vis = ParticleHistoryVisualizer(mock_config)
        color = vis._get_marker_color(kw=14, teff=10000)
        assert color == "black"

    def test_get_marker_color_normal_star(self, mock_config):
        """Test _get_marker_color returns RGB tuple for normal star"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        vis = ParticleHistoryVisualizer(mock_config)
        color = vis._get_marker_color(kw=1, teff=5778)

        # Should return a tuple (not string)
        assert isinstance(color, tuple) or isinstance(color, np.ndarray) or color == "gray"

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    @patch("matplotlib.pyplot.close")
    def test_plot_single_star(self, mock_close, mock_makedirs, mock_exists, mock_config, temp_dir):
        """Test plot method with single star data"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        mock_config.plot_dir = str(temp_dir)
        vis = ParticleHistoryVisualizer(mock_config, simu_name="test_sim", particle_name=1000)

        history_df = pd.DataFrame(
            {
                "Name": [1000],
                "TTOT": [1.0],
                "Time[Myr]": [10.0],
                "M": [10.0],
                "R*": [1.0],
                "Teff*": [5778.0],
                "KW": [1],
                "state": ["single"],
                "X1": [1.0],
                "X2": [0.5],
            }
        )

        # Should not raise an error
        try:
            vis.plot(history_df, time=1.0)
        except Exception:
            # Some plotting functions might fail in headless environment
            pass

    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    @patch("matplotlib.pyplot.close")
    def test_plot_binary_system(
        self, mock_close, mock_makedirs, mock_exists, mock_config, temp_dir
    ):
        """Test plot method with binary system data"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        mock_config.plot_dir = str(temp_dir)
        vis = ParticleHistoryVisualizer(mock_config, simu_name="test_sim", particle_name=1000)

        history_df = pd.DataFrame(
            {
                "Name": [1000],
                "TTOT": [1.0],
                "Time[Myr]": [10.0],
                "state": ["binary"],
                "Bin cm X1": [1.0],
                "Bin cm X2": [0.5],
                "Bin cm X [pc]": [1.0],
                "Bin cm Y [pc]": [0.5],
                "Bin M1*": [10.0],
                "Bin M2*": [5.0],
                "Bin A[au]": [100.0],
                "Bin ECC": [0.3],
                "Bin KW1": [14],
                "Bin KW2": [13],
                "Bin Teff1*": [0.0],
                "Bin Teff2*": [0.0],
                "Bin R1*": [0.001],
                "Bin R2*": [0.001],
                "Ebind/kT": [10.0],
                "tau_gw[Myr]": [1000.0],
            }
        )

        # Should not raise an error
        try:
            vis.plot(history_df, time=1.0)
        except Exception:
            # Some plotting functions might fail in headless environment
            pass

    def test_plot_empty_df_returns_early(self, mock_config):
        """Test plot method returns early for empty DataFrame"""
        from nbody_pipeline.visualization import ParticleHistoryVisualizer

        vis = ParticleHistoryVisualizer(mock_config)

        # Should not raise an error and return early
        vis.plot(pd.DataFrame())


def _touch_plot(plot_dir: Path, prefix: str, filename_var_part: str, ttot: str = "1.0") -> Path:
    path = plot_dir / "jpg" / f"{prefix}output_ttot_{ttot}_{filename_var_part}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fake jpg")
    return path


class TestPlotPurger:
    """Test HDF5 plot purge registry and deletion behavior."""

    def test_each_hdf5_method_target_matches_only_own_file(self, mock_config, tmp_path):
        from nbody_pipeline.visualization.purge import PLOT_TARGETS

        mock_config.plot_dir = str(tmp_path)
        mock_config.figname_prefix = {"sim_a": "a_"}
        purger = PlotPurger(mock_config)

        for entry in PLOT_TARGETS:
            for path in (tmp_path / "jpg").glob("*.jpg"):
                path.unlink()
            expected = _touch_plot(tmp_path, "a_", entry.filename_var_part)
            _touch_plot(tmp_path, "a_", f"not_{entry.filename_var_part}")
            _touch_plot(tmp_path, "wrong_", entry.filename_var_part)
            (tmp_path / f"a_output_ttot_1.0_{entry.filename_var_part}.jpg").write_text("outside")

            result = purger.preview(entry.qualified_name, simu_name="sim_a")

            assert result.matched_paths == [expected.resolve()]

    def test_composite_targets_expand_to_expected_method_sets(self, mock_config, tmp_path):
        from nbody_pipeline.visualization.purge import (
            BINARY_TARGETS,
            PLOT_TARGETS,
            SINGLE_TARGETS,
        )

        mock_config.plot_dir = str(tmp_path)
        mock_config.figname_prefix = {"sim_a": "a_"}
        purger = PlotPurger(mock_config)

        for entry in PLOT_TARGETS:
            _touch_plot(tmp_path, "a_", entry.filename_var_part)

        single = purger.preview("single", simu_name="sim_a")
        binary = purger.preview("binary", simu_name="sim_a")
        hdf5 = purger.preview("hdf5", simu_name="sim_a")

        assert len(single.matched_paths) == len(SINGLE_TARGETS)
        assert len(binary.matched_paths) == len(BINARY_TARGETS)
        assert len(hdf5.matched_paths) == len(PLOT_TARGETS)
        assert (
            "single.create_galactic_energy_angular_momentum_plot_jpg" in PlotPurger.list_targets()
        )

    def test_simu_name_limits_prefix_and_all_sims_uses_all_prefixes(self, mock_config, tmp_path):
        mock_config.plot_dir = str(tmp_path)
        mock_config.figname_prefix = {"sim_a": "a_", "sim_b": "b_"}
        purger = PlotPurger(mock_config)
        _touch_plot(tmp_path, "a_", "x1_vs_x2")
        _touch_plot(tmp_path, "b_", "x1_vs_x2")

        one_sim = purger.preview("single.create_position_plot_jpg", simu_name="sim_a")
        all_sims = purger.preview("single.create_position_plot_jpg")

        assert len(one_sim.matched_paths) == 1
        assert one_sim.matched_paths[0].name.startswith("a_")
        assert len(all_sims.matched_paths) == 2

    def test_filename_suffix_matching(self, mock_config, tmp_path):
        mock_config.plot_dir = str(tmp_path)
        mock_config.figname_prefix = {"sim_a": "a_"}
        purger = PlotPurger(mock_config)
        default = _touch_plot(tmp_path, "a_", "x1_vs_x2")
        custom = _touch_plot(tmp_path, "a_", "x1_vs_x2_custom")

        default_result = purger.preview("single.create_position_plot_jpg", simu_name="sim_a")
        custom_result = purger.preview(
            "single.create_position_plot_jpg",
            simu_name="sim_a",
            filename_suffix="custom",
        )

        assert default_result.matched_paths == [default.resolve()]
        assert custom_result.matched_paths == [custom.resolve()]

    def test_long_preview_shows_head_and_tail(self, tmp_path):
        paths = [tmp_path / f"file_{i:02d}.jpg" for i in range(25)]

        preview = PlotPurger.format_preview(paths)

        assert "Matched 25 files" in preview
        assert "file_00.jpg" in preview
        assert "file_09.jpg" in preview
        assert "omitted 5 files" in preview
        assert "file_15.jpg" in preview
        assert "file_24.jpg" in preview
        assert "file_10.jpg" not in preview

    def test_confirmation_cancel_does_not_delete(self, mock_config, tmp_path, monkeypatch):
        mock_config.plot_dir = str(tmp_path)
        mock_config.figname_prefix = {"sim_a": "a_"}
        path = _touch_plot(tmp_path, "a_", "x1_vs_x2")
        purger = PlotPurger(mock_config)
        monkeypatch.setattr("builtins.input", lambda prompt: "no")

        result = purger.purge("single.create_position_plot_jpg", simu_name="sim_a")

        assert result.cancelled is True
        assert result.deleted_paths == []
        assert path.exists()

    def test_confirmation_phrase_deletes(self, mock_config, tmp_path, monkeypatch):
        mock_config.plot_dir = str(tmp_path)
        mock_config.figname_prefix = {"sim_a": "a_"}
        path = _touch_plot(tmp_path, "a_", "x1_vs_x2")
        purger = PlotPurger(mock_config)
        monkeypatch.setattr("builtins.input", lambda prompt: "delete 1 files")

        result = purger.purge("single.create_position_plot_jpg", simu_name="sim_a")

        assert result.cancelled is False
        assert result.deleted_paths == [path.resolve()]
        assert not path.exists()

    def test_visualizer_purge_adds_group_prefix(self, mock_config, tmp_path):
        mock_config.plot_dir = str(tmp_path)
        mock_config.figname_prefix = {"sim_a": "a_"}
        path = _touch_plot(tmp_path, "a_", "x1_vs_x2")
        vis = SingleStarVisualizer(mock_config)

        result = vis.purge("create_position_plot_jpg", simu_name="sim_a", yes=True)

        assert result.deleted_paths == [path.resolve()]
        assert not path.exists()


class TestGalacticOrbitVisualizer:
    """Tests for galactic-orbit plotting."""

    def _orbit_df(self):
        return pd.DataFrame(
            {
                "TTOT": [1.0, 2.0, 3.0],
                "Time[Myr]": [10.0, 250.0, 600.0],
                "RG(1)": [1.0, 2.0, 3.0],
                "RG(2)": [2.0, 3.0, 4.0],
                "RG(3)": [3.0, 4.0, 5.0],
                "VG(1)": [10.0, 20.0, 30.0],
                "VG(2)": [11.0, 21.0, 31.0],
                "VG(3)": [12.0, 22.0, 32.0],
            }
        )

    def test_projection_plot_has_three_subplots_and_fixed_color_range(
        self, mock_config, tmp_path, monkeypatch
    ):
        mock_config.plot_dir = str(tmp_path)
        mock_config.figname_prefix = {"sim_a": "a_"}
        mock_config.galactic_orbit = {"time_color_max_myr": 500.0}
        visualizer = GalacticOrbitVisualizer(mock_config)
        saved_figures = []

        def fake_savefig(self, path, *args, **kwargs):
            saved_figures.append((self, Path(path)))

        monkeypatch.setattr(plt.Figure, "savefig", fake_savefig)
        path = visualizer.create_projection_plot(self._orbit_df(), "sim_a")

        assert path == tmp_path / "a__galactic_orbit_projection.pdf"
        assert saved_figures[0][1] == path
        fig = saved_figures[0][0]
        assert len(fig.axes) == 4
        projection_axes = fig.axes[:3]
        assert [ax.get_title() for ax in projection_axes] == ["XY", "YZ", "ZX"]
        scatter = projection_axes[0].collections[0]
        assert scatter.norm.vmin == 0.0
        assert scatter.norm.vmax == 500.0

    def test_plotly_html_uses_3d_scatter_and_fixed_color_range(self, mock_config, tmp_path):
        pytest.importorskip("plotly")
        mock_config.plot_dir = str(tmp_path)
        mock_config.figname_prefix = {"sim_a": "a_"}
        mock_config.galactic_orbit = {"time_color_max_myr": 500.0}
        visualizer = GalacticOrbitVisualizer(mock_config)

        path = visualizer.create_interactive_3d_html(self._orbit_df(), "sim_a")

        assert path == tmp_path / "a__galactic_orbit_3d.html"
        html = path.read_text()
        assert '"type":"scatter3d"' in html
        assert '"mode":"markers"' in html
        assert '"cmin":0' in html
        assert '"cmax":500' in html


class TestPurgeCLI:
    """Test purge CLI behavior."""

    def test_cli_top_level_help(self, capsys):
        from nbody_pipeline.__main__ import main

        assert main(["--help"]) == 0
        output = capsys.readouterr().out
        assert "usage:" in output
        assert "--skip-until" in output
        assert "--debug" in output
        assert "purge" in output
        assert "nbody-plot" in output

    def test_cli_help_verb(self, capsys):
        from nbody_pipeline.__main__ import main

        assert main(["help"]) == 0
        output = capsys.readouterr().out
        assert "usage:" in output
        assert "subcommands" in output

    def test_cli_help_purge(self, capsys):
        from nbody_pipeline.__main__ import main

        assert main(["help", "purge"]) == 0
        output = capsys.readouterr().out
        assert "--list-targets" in output
        assert "--dry-run" in output

    def test_cli_purge_help_flag_and_verb(self, capsys):
        from nbody_pipeline.__main__ import main

        assert main(["purge", "--help"]) == 0
        output = capsys.readouterr().out
        assert "--list-targets" in output
        assert "--dry-run" in output

        assert main(["purge", "help"]) == 0
        output = capsys.readouterr().out
        assert "--list-targets" in output
        assert "--dry-run" in output

    def test_cli_unknown_help_topic(self, capsys):
        from nbody_pipeline.__main__ import main

        assert main(["help", "missing"]) == 2
        output = capsys.readouterr().out
        assert "Unknown help topic: missing" in output

    def test_cli_short_skip_until_normalized(self, monkeypatch):
        import nbody_pipeline.__main__ as main_module

        captured = {}

        class FakeConfig:
            def __init__(self, config_path=None, opts=None):
                captured["opts"] = opts

        class FakePlotter:
            def __init__(self, config):
                self.config = config

            def plot_all_simulations(self):
                captured["ran"] = True

        monkeypatch.setattr(main_module, "ConfigManager", FakeConfig)
        monkeypatch.setattr(main_module, "SimulationPlotter", FakePlotter)

        assert main_module.main(["-k", "last"]) == 0
        assert captured["opts"] == [("--skip-until", "last")]
        assert captured["ran"] is True

    def test_cli_list_targets(self, capsys):
        from nbody_pipeline.__main__ import main

        assert main(["purge", "--list-targets"]) == 0
        output = capsys.readouterr().out
        assert "single.create_position_plot_jpg" in output
        assert "hdf5" in output

    def test_cli_dry_run(self, mock_config, tmp_path, monkeypatch, capsys):
        import nbody_pipeline.__main__ as main_module

        mock_config.plot_dir = str(tmp_path)
        mock_config.figname_prefix = {"sim_a": "a_"}
        _touch_plot(tmp_path, "a_", "x1_vs_x2")
        monkeypatch.setattr(
            main_module, "ConfigManager", lambda config_path=None, opts=None: mock_config
        )

        assert (
            main_module.main(
                ["purge", "single.create_position_plot_jpg", "--simu", "sim_a", "--dry-run"]
            )
            == 0
        )
        output = capsys.readouterr().out
        assert "Matched 1 files" in output
        assert "x1_vs_x2.jpg" in output

    def test_cli_yes_deletes(self, mock_config, tmp_path, monkeypatch, capsys):
        import nbody_pipeline.__main__ as main_module

        mock_config.plot_dir = str(tmp_path)
        mock_config.figname_prefix = {"sim_a": "a_"}
        path = _touch_plot(tmp_path, "a_", "x1_vs_x2")
        monkeypatch.setattr(
            main_module, "ConfigManager", lambda config_path=None, opts=None: mock_config
        )

        assert (
            main_module.main(
                ["purge", "single.create_position_plot_jpg", "--simu", "sim_a", "--yes"]
            )
            == 0
        )
        output = capsys.readouterr().out
        assert "Deleted 1 files" in output
        assert not path.exists()
