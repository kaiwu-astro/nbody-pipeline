"""Base visualization classes and helper functions"""

import logging
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dragon3_pipelines.utils import BlackbodyColorConverter

logger = logging.getLogger(__name__)


def set_mpl_fonts() -> None:
    """Configure matplotlib fonts and styling for plots."""
    # SMALL_SIZE = 8
    # MEDIUM_SIZE = 10
    BIGGER_SIZE = 15
    plt.rc("mathtext", fontset="dejavuserif")
    plt.rc("font", family="serif")
    plt.rc("font", size=BIGGER_SIZE)
    plt.rc("axes", titlesize=BIGGER_SIZE)
    plt.rc("axes", labelsize=BIGGER_SIZE)
    plt.rc("xtick", labelsize=BIGGER_SIZE)
    plt.rc("ytick", labelsize=BIGGER_SIZE)
    plt.rc("legend", fontsize=BIGGER_SIZE)
    plt.rc("figure", titlesize=BIGGER_SIZE)
    plt.rc("figure", figsize=(6, 6))
    plt.rc("savefig", transparent=True, dpi=330, bbox="tight")
    plt.rc("errorbar", capsize=3)
    plt.rc("legend", framealpha=0.1)


def add_grid(axs: Union[plt.Axes, List[plt.Axes]], which: str = "both", axis: str = "both") -> None:
    """
    Add grid to axes.

    Args:
        axs: Single axis or list of axes
        which: {'both', 'major', 'minor'}
        axis: {'both', 'x', 'y'}
    """
    try:
        axs[0]
        axes = axs
    except Exception:
        axes = [
            axs,
        ]

    for ax in axes:
        if which == "both" or which == "minor":
            if ax.get_yscale() == "linear":
                ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
            if ax.get_xscale() == "linear":
                ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(visible=True, which=which, axis=axis, color="black", alpha=0.1)


class BaseVisualizer:
    """Base visualization class for handling plotting functionality"""

    def __init__(self, config_manager: Any) -> None:
        """
        Initialize the visualizer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.teff_to_rgb_converter = BlackbodyColorConverter(
            cache_path=getattr(config_manager, "teff_rgb_cache", None)
        )
        self.setup_figure_style()

    def setup_figure_style(self) -> None:
        """Set up figure styling"""
        set_mpl_fonts()
        plt.rc("savefig", dpi=233)

    def luminosity_to_plot_alpha(self, L_arr: np.ndarray) -> np.ndarray:
        """
        Convert luminosity array to alpha values for plotting.

        Args:
            L_arr: Array of luminosity values

        Returns:
            Normalized alpha values array
        """
        result = np.log10(L_arr)
        special_mask = result == -10
        min_val = np.min(result)
        max_val = np.max(result)
        result = (result - min_val) / (max_val - min_val)
        result[special_mask] = 1
        return result


class BaseHDF5Visualizer(BaseVisualizer):
    """Base class for HDF5-based visualization"""

    def decorate_jointfig(
        self,
        ax: plt.Axes,
        data_at_t: pd.DataFrame,
        x: str,
        y: str,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        simu_name: str,
        ttot: float,
        tmyr: float,
        t_over_tcr0: float,
        t_over_trh0: float,
        highlight_outlier: bool = True,
    ) -> None:
        """
        Decorate joint plot with labels and styling.

        Args:
            ax: Matplotlib axes
            data_at_t: DataFrame with current timestep data
            x: X-axis column name
            y: Y-axis column name
            xlim: X-axis limits
            ylim: Y-axis limits
            simu_name: Simulation name
            ttot: Total time
            tmyr: Time in Myr
            t_over_tcr0: Time over TCR0
            t_over_trh0: Time over TRH0
            highlight_outlier: Whether to highlight outliers in red
        """
        x_min, x_max = data_at_t[x].min(), data_at_t[x].max()
        y_min, y_max = data_at_t[y].min(), data_at_t[y].max()
        ax.set(xlim=xlim, ylim=ylim)
        xlabel = self.config.colname_to_label.get(x, x)
        ylabel = self.config.colname_to_label.get(y, y)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        with mpl.rc_context(**self.config.fixed_width_font_context):
            ax.set_xlabel(
                f"{xlabel}\n{x_min:9.1f} - {x_max:9.1f}",
                family=self.config.fixed_width_font_context["rc"]["font.family"],
            )
            ax.set_ylabel(
                f"{ylabel}\n{y_min:9.1f} - {y_max:9.1f}",
                family=self.config.fixed_width_font_context["rc"]["font.family"],
            )
            if highlight_outlier:
                if x_min < xlim[0] or x_max > xlim[1]:
                    ax.xaxis.label.set_color("red")
                    ax.tick_params(axis="x", which="both", colors="red")
                    ax.spines["bottom"].set_color("red")
                if y_min < ylim[0] or y_max > ylim[1]:
                    ax.yaxis.label.set_color("red")
                    ax.tick_params(axis="y", which="both", colors="red")
                    ax.spines["left"].set_color("red")
            ax.figure.suptitle(
                f"{simu_name} | Time = {ttot:9.3f} NB = {tmyr:9.2f} Myr\n{t_over_tcr0:7.0f} TCR0 = {t_over_trh0:3.1f} TRH0",
            )

    def _create_jointplot_density(
        self,
        df_at_t: pd.DataFrame,
        simu_name: str,
        x_col: str,
        y_col: str,
        log_scale: Tuple[bool, bool],
        filename_var_part: str,
        xlim_key: Optional[str] = None,
        ylim_key: Optional[str] = None,
        extra_data_handler: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        extra_ax_handler: Optional[Callable[[plt.Axes], None]] = None,
        custom_ax_joint_decorator: Optional[Callable[[plt.Axes, pd.DataFrame], None]] = None,
        save_pdf=False,
    ) -> None:
        """
        Helper function to create density plots based on sns.jointplot.

        Args:
            df_at_t: DataFrame containing plot data
            simu_name: Simulation name
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            log_scale: Tuple (bool, bool) indicating whether x and y axes use log scale
            filename_var_part: Variable part for constructing filename
            xlim_key: Key for x-axis limits in self.config.limits, defaults to x_col
            ylim_key: Key for y-axis limits in self.config.limits, defaults to y_col
            extra_data_handler: Optional callback to process DataFrame before plotting
            extra_ax_handler: Optional callback to customize joint axes after main plot
            custom_ax_joint_decorator: Optional callback for specific ax_joint operations
                                       after decorate_jointfig and before saving
        """
        if xlim_key is None:
            xlim_key = x_col
        if ylim_key is None:
            ylim_key = y_col
        ttot = df_at_t["TTOT"].iloc[0]
        tmyr = df_at_t["Time[Myr]"].iloc[0]
        t_over_tcr0 = df_at_t["TTOT/TCR0"].iloc[0]
        t_over_trh0 = df_at_t["TTOT/TRH0"].iloc[0]

        processed_df = df_at_t
        if extra_data_handler is not None:
            processed_df = extra_data_handler(df_at_t)

        base_filename = (
            f"{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_{filename_var_part}"
        )
        save_pdf_path = f"{self.config.plot_dir}/{base_filename}.pdf"
        save_jpg_path = f"{self.config.plot_dir}/jpg/{base_filename}.jpg"

        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return

        g = sns.jointplot(
            data=processed_df, x=x_col, y=y_col, kind="hist", bins=100, log_scale=log_scale
        )

        if extra_ax_handler is not None:
            extra_ax_handler(g.ax_joint)

        self.decorate_jointfig(
            g.ax_joint,
            processed_df,
            x_col,
            y_col,
            self.config.limits[xlim_key],
            self.config.limits[ylim_key],
            simu_name,
            ttot,
            tmyr,
            t_over_tcr0,
            t_over_trh0,
        )

        if custom_ax_joint_decorator is not None:
            custom_ax_joint_decorator(g.ax_joint, processed_df)

        add_grid(g.ax_joint)
        g.savefig(save_jpg_path)
        if save_pdf:
            g.savefig(save_pdf_path)
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(g.figure)
        except NameError:
            plt.close(g.figure)

    def _symlogY_and_fill_handler(self, ax: plt.Axes, linthresh: float = 10) -> None:
        """
        Set y-axis to symlog scale and fill linear region.

        Args:
            ax: Matplotlib axes
            linthresh: Linear threshold for symlog scale
        """
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.axhspan(-linthresh, linthresh, color="lightgray", alpha=0.3)


class HDF5Visualizer:
    """Wrapper class for HDF5 visualization combining single and binary star visualizers"""

    def __init__(self, config_manager: Any) -> None:
        """
        Initialize HDF5 visualizer.

        Args:
            config_manager: Configuration manager instance
        """
        from dragon3_pipelines.visualization.single_star import SingleStarVisualizer
        from dragon3_pipelines.visualization.binary_star import BinaryStarVisualizer

        self.single = SingleStarVisualizer(config_manager)
        self.binary = BinaryStarVisualizer(config_manager)


class BaseContinousFileVisualizer(BaseVisualizer):
    """Base class for continuous file visualization"""

    pass
