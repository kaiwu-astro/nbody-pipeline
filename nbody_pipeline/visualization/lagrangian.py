"""Lagrangian visualization tools"""

import logging
from typing import Any, Callable, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nbody_pipeline.io import LagrFileProcessor
from nbody_pipeline.visualization.base import BaseContinousFileVisualizer, add_grid

logger = logging.getLogger(__name__)


class LagrVisualizer(BaseContinousFileVisualizer):
    """
    Visualizer for Lagrangian radii data.
    All lagr plots are redrawn each time, skip_existing_plot parameter is ignored.
    """

    def __init__(self, config_manager: Any) -> None:
        """
        Initialize Lagrangian visualizer.

        Args:
            config_manager: Configuration manager instance
        """
        super().__init__(config_manager)
        self.metric_to_plot_label: Dict[str, str] = {
            "rlagr": "Lagrangian radii [pc]",
            "rlagr_s": "Lagrangian radii of single stars [pc]",
            "rlagr_b": "Lagrangian radii of binary stars [pc]",
            "avmass": "Average mass [Msolar]",
            "nshell": "Number of stars",
            "total_mass": "Total mass [Msolar]",
            "vx": "Mass weighted X velocity [km/s]",
            "vy": "Mass weighted Y velocity [km/s]",
            "vz": "Mass weighted Z velocity [km/s]",
            "v": "Mass weighted velocity [km/s]",
            "vr": "Mass weighted radial velocity [km/s]",
            "vt": "Mass weighted tangential velocity [km/s]",
            "sigma2": "Mass weighted\nvelocity dispersion squared [${km}^2~s^{-2}$]",
            "sigma": "Mass weighted velocity dispersion [km/s]",
            "sigma_r2": "Mass weighted\nradial velocity dispersion squared [${km}^2~s^{-2}$]",
            "sigma_r": "Mass weighted radial velocity dispersion [km/s]",
            "sigma_t2": "Mass weighted\ntangential velocity dispersion squared [${km}^2~s^{-2}$]",
            "sigma_t": "Mass weighted tangential velocity dispersion [km/s]",
            "vrot": "Mass weighted\nrotational velocity [km/s]",
        }

    @staticmethod
    def _normalize_lagr_percent(percent: Any) -> str:
        """Normalize configured Lagrangian percent values to plotted labels."""
        if isinstance(percent, str):
            return percent
        return f"{percent}%"

    def create_lagr_plot_base(
        self,
        l7df_sns: pd.DataFrame,
        simu_name: str,
        metric: str = "rlagr",
        filename_suffix: Optional[str] = None,
        extra_ax_handler: Optional[Callable[[plt.Axes], None]] = None,
    ) -> None:
        """
        Create base Lagrangian plot.

        Args:
            l7df_sns: DataFrame with Lagrangian data
            simu_name: Simulation name
            metric: Metric to plot
            filename_suffix: Optional suffix for filename
            extra_ax_handler: Optional callback to customize axes
        """
        if filename_suffix is None:
            save_pdf_path = (
                f"{self.config.plot_dir}/{self.config.figname_prefix[simu_name]}_{metric}.pdf"
            )
        else:
            save_pdf_path = f"{self.config.plot_dir}/{self.config.figname_prefix[simu_name]}_{metric}_{filename_suffix}.pdf"
        selected_lagr_percent = [
            self._normalize_lagr_percent(percent) for percent in self.config.selected_lagr_percent
        ]
        l7df_sns_selected_metric = l7df_sns[
            (l7df_sns["Metric"] == metric) & (l7df_sns["%"].isin(selected_lagr_percent))
        ]
        l7df_sns_selected_metric = l7df_sns_selected_metric[
            l7df_sns_selected_metric["Time[Myr]"] > 0
        ]
        fig, ax = plt.subplots()
        sns.lineplot(
            data=l7df_sns_selected_metric,
            x="Time[Myr]",
            y="Value",
            hue="%",
            hue_order=selected_lagr_percent,
            ax=ax,
        )
        ax.set(yscale="log", ylabel=self.metric_to_plot_label[metric], title=simu_name)
        if extra_ax_handler is not None:
            extra_ax_handler(ax)

        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        add_grid(ax)
        fig.savefig(save_pdf_path)
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(fig)
        except NameError:
            plt.close(fig)

    def _extra_ax_handler_rlagr(self, ax: plt.Axes) -> None:
        """Set y-limits for rlagr plots"""
        ax.set(ylim=(4e-2, 20))

    def _extra_ax_handler_avmass(self, ax: plt.Axes) -> None:
        """Set y-limits for average mass plots"""
        ax.set(ylim=(0.3, 30))

    def _extra_ax_handler_sigma(self, ax: plt.Axes) -> None:
        """Set y-limits for velocity dispersion plots"""
        ax.set(ylim=(1e1, 1e15))

    def _extra_ax_handler_logx(self, ax: plt.Axes) -> None:
        """Set x-axis to log scale"""
        ax.set_xscale("log")

    def _extra_ax_handler_rlagr_logx(self, ax: plt.Axes) -> None:
        """Combined handler for rlagr with log x-axis"""
        self._extra_ax_handler_rlagr(ax)
        self._extra_ax_handler_logx(ax)

    def _extra_ax_handler_avmass_logx(self, ax: plt.Axes) -> None:
        """Combined handler for average mass with log x-axis"""
        self._extra_ax_handler_avmass(ax)
        self._extra_ax_handler_logx(ax)

    def _extra_ax_handler_sigma_logx(self, ax: plt.Axes) -> None:
        """Combined handler for velocity dispersion with log x-axis"""
        self._extra_ax_handler_sigma(ax)
        self._extra_ax_handler_logx(ax)

    def create_total_mass_plot(self, l7df_sns: pd.DataFrame, simu_name: str) -> None:
        """Create total mass plot from 100% average mass and shell count."""
        save_pdf_path = (
            f"{self.config.plot_dir}/{self.config.figname_prefix[simu_name]}_total_mass.pdf"
        )

        total_mass_df = LagrFileProcessor.build_total_mass_df(l7df_sns)

        fig, ax = plt.subplots()
        sns.lineplot(data=total_mass_df, x="Time[Myr]", y="total_mass", ax=ax)
        ax.set(
            ylabel=self.metric_to_plot_label["total_mass"],
            title=simu_name,
        )
        ax.set_ylim(bottom=0)
        add_grid(ax)
        fig.savefig(save_pdf_path)
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(fig)
        except NameError:
            plt.close(fig)

    def create_lagr_radii_plot(self, l7df_sns: pd.DataFrame, simu_name: str) -> None:
        """Create Lagrangian radii plots (both linear and log-log)"""
        self.create_lagr_plot_base(
            l7df_sns, simu_name, metric="rlagr", extra_ax_handler=self._extra_ax_handler_rlagr
        )
        self.create_lagr_plot_base(
            l7df_sns,
            simu_name,
            metric="rlagr",
            filename_suffix="loglog",
            extra_ax_handler=self._extra_ax_handler_rlagr_logx,
        )

    def create_lagr_avmass_plot(self, l7df_sns: pd.DataFrame, simu_name: str) -> None:
        """Create average mass plots (both linear and log-log)"""
        self.create_lagr_plot_base(
            l7df_sns, simu_name, metric="avmass", extra_ax_handler=self._extra_ax_handler_avmass
        )
        self.create_lagr_plot_base(
            l7df_sns,
            simu_name,
            metric="avmass",
            filename_suffix="loglog",
            extra_ax_handler=self._extra_ax_handler_avmass_logx,
        )

    def create_lagr_velocity_dispersion_plot(self, l7df_sns: pd.DataFrame, simu_name: str) -> None:
        """Create velocity dispersion plots (both linear and log-log)"""
        self.create_lagr_plot_base(
            l7df_sns, simu_name, metric="sigma", extra_ax_handler=self._extra_ax_handler_sigma
        )
        self.create_lagr_plot_base(
            l7df_sns,
            simu_name,
            metric="sigma",
            filename_suffix="loglog",
            extra_ax_handler=self._extra_ax_handler_sigma_logx,
        )
