"""Binary star visualization tools"""

import logging
import os
from typing import Callable, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nbody_pipeline.utils import log_time
from nbody_pipeline.visualization.base import BaseHDF5Visualizer, add_grid
from nbody_pipeline.visualization.purge import PlotPurger, PurgeResult

logger = logging.getLogger(__name__)


class BinaryStarVisualizer(BaseHDF5Visualizer):
    """Visualizer for binary star data"""

    def purge(
        self,
        target: str,
        simu_name: str | None = None,
        plot_dir: str | os.PathLike[str] | None = None,
        filename_suffix: str | None = None,
        yes: bool = False,
    ) -> PurgeResult:
        """Purge binary-star HDF5 JPG outputs for a target."""
        qualified_target = target if target.startswith("binary.") else f"binary.{target}"
        return PlotPurger(self.config).purge(
            qualified_target,
            simu_name=simu_name,
            plot_dir=plot_dir,
            filename_suffix=filename_suffix,
            yes=yes,
        )

    def _create_base_jpg_plot_compact_object_only(
        self,
        binary_df_at_t: pd.DataFrame,
        simu_name: str,
        x_col: str,
        y_col: str,
        log_scale: Tuple[bool, bool],
        filename_var_part: str,
        xlim_key: Optional[str] = None,
        ylim_key: Optional[str] = None,
        extra_ax_handler: Optional[Callable[[plt.Axes], None]] = None,
        custom_ax_decorator: Optional[Callable[[plt.Axes, pd.DataFrame], None]] = None,
    ) -> None:
        """
        Helper function to create basic JPG scatter or line plots for compact objects only.

        Args:
            binary_df_at_t: Binary star DataFrame at timestep
            simu_name: Simulation name
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            log_scale: Tuple (bool, bool) indicating whether x and y axes use log scale
            filename_var_part: Variable part for constructing filename
            xlim_key: Key for x-axis limits in self.config.limits, defaults to x_col
            ylim_key: Key for y-axis limits in self.config.limits, defaults to y_col
            extra_ax_handler: Optional callback to customize axes after main plot
            custom_ax_decorator: Optional callback for specific ax operations
        """
        if xlim_key is None:
            xlim_key = x_col
        if ylim_key is None:
            ylim_key = y_col
        ttot = binary_df_at_t["TTOT"].iloc[0]
        tmyr = binary_df_at_t["Time[Myr]"].iloc[0]
        t_over_tcr0 = binary_df_at_t["TTOT/TCR0"].iloc[0]
        t_over_trh0 = binary_df_at_t["TTOT/TRH0"].iloc[0]

        save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_{filename_var_part}.jpg"
        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return

        binary_df_at_t_cbo = self.get_compact_object_only(binary_df_at_t)
        vc = binary_df_at_t_cbo["Stellar Type"].value_counts()
        vc_df = vc.reset_index()

        fig, ax = plt.subplots()
        for _st in binary_df_at_t_cbo["Stellar Type"].unique():
            sns.lineplot(
                data=binary_df_at_t_cbo[binary_df_at_t_cbo["Stellar Type"] == _st],
                x=x_col,
                y=y_col,
                lw=0,
                **self.get_binary_cookie_dict(_st.split("-")[0], _st.split("-")[1]),
                legend=False,
                ax=ax,
            )
        if log_scale[0]:
            ax.set(xscale="log")
        if log_scale[1]:
            ax.set(yscale="log")

        legend_elements = []
        if len(binary_df_at_t_cbo) > 0:
            _table = ax.table(
                cellText=vc_df.values,
                colLabels=vc_df.columns,
                colWidths=[0.18, 0.1],
                colLoc="right",
                loc="lower left",
            )

            stellar_kws = np.unique(
                np.concatenate(
                    (binary_df_at_t_cbo["Bin KW1"].unique(), binary_df_at_t_cbo["Bin KW2"].unique())
                )
            )
            for _kw in stellar_kws:
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        lw=0,
                        markerfacecolor=self.config.palette_st[_kw + 1],
                        markeredgecolor="black",
                        markersize=10,
                        alpha=0.6,
                        label=self.config.kw_to_stellar_type[_kw],
                    )
                )
        else:
            pass
        _emptylabel = " " * 5
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                lw=0,
                markerfacecolor="none",
                markeredgecolor="none",
                markersize=10,
                alpha=0.6,
                label=_emptylabel,
            )
        )
        with mpl.rc_context(**self.config.fixed_width_font_context):
            ax.legend(
                handles=legend_elements,
                loc="upper left",
                frameon=True,
                title="Stellar Types",
                bbox_to_anchor=(1, 1),
            )

        self.decorate_jointfig(
            ax,
            binary_df_at_t_cbo,
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

        if extra_ax_handler is not None:
            extra_ax_handler(ax)

        if custom_ax_decorator is not None:
            custom_ax_decorator(ax, binary_df_at_t_cbo)

        add_grid(ax)
        fig.savefig(save_jpg_path)
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(ax.figure)
        except NameError:
            plt.close(fig)

    def get_compact_object_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to include only compact objects"""
        from nbody_pipeline.io.hdf5_reader import HDF5FileProcessor

        hdf5_file_processor = HDF5FileProcessor(self.config)
        compact_object_mask = hdf5_file_processor.get_compact_object_mask(df)
        return df[compact_object_mask]

    @log_time(logger)
    def get_binary_cookie_dict(self, st1: str, st2: str) -> dict:
        """Get marker styling dictionary for binary stars"""
        return self.get_binary_cookie_dict_num(
            self.config.stellar_type_to_kw[st1], self.config.stellar_type_to_kw[st2]
        )

    @log_time(logger)
    def get_binary_cookie_dict_num(self, kw1: int, kw2: int) -> dict:
        """Get marker styling dictionary for binary stars by KW numbers"""
        cookie_tastes = self.config.palette_st
        sts = self.config.kw_to_stellar_type[kw1] + "-" + self.config.kw_to_stellar_type[kw2]
        fs = "top" if sts in self.config.wow_binary_st_list else "left"
        ms = 15 if sts in self.config.wow_binary_st_list else 10
        return dict(
            marker=self.config.marker_fill_list[kw1 + 1],
            markerfacecolor=cookie_tastes[kw1 + 1],
            markerfacecoloralt=cookie_tastes[kw2 + 1],
            markeredgecolor="black",
            fillstyle=fs,
            ms=ms,
            alpha=0.6,
        )

    @log_time(logger)
    def create_mass_ratio_m1_plot_density(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create mass ratio plot"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col="primary_mass[solar]",
            y_col="mass_ratio",
            log_scale=(True, True),
            xlim_key="M",
            filename_var_part="mass_ratio_vs_primary_mass_loglog",
        )

    @log_time(logger)
    def create_mass_ratio_m1_plot_jpg_compact_object_only(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create mass ratio plot for compact objects only"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col="primary_mass[solar]",
            y_col="mass_ratio",
            log_scale=(True, False),
            xlim_key="M",
            filename_var_part="mass_ratio_vs_primary_mass_loglog_compact_objects_only",
        )

    def _decorator_semi_m1(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Decorator to add solar radius line"""
        ax.axhline(0.00465, color="darkred", linestyle="--", label="Solar radius")
        ax.text(
            ax.get_xlim()[-1],
            0.00465,
            "Solar radius",
            color="darkred",
            fontsize=10,
            horizontalalignment="right",
        )

    @log_time(logger)
    def create_semi_m1_plot_density(self, binary_df_at_t: pd.DataFrame, simu_name: str) -> None:
        """Create semi-major axis vs primary mass plot"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col="primary_mass[solar]",
            y_col="Bin A[au]",
            log_scale=(True, True),
            xlim_key="M",
            filename_var_part="a_vs_primary_mass_loglog",
            custom_ax_joint_decorator=self._decorator_semi_m1,
        )

    @log_time(logger)
    def create_semi_m1_plot_jpg_compact_object_only(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create semi-major axis vs primary mass plot for compact objects only"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col="primary_mass[solar]",
            y_col="Bin A[au]",
            log_scale=(True, True),
            xlim_key="M",
            filename_var_part="a_vs_primary_mass_loglog_compact_objects_only",
            custom_ax_decorator=self._decorator_semi_m1,
        )

    def _decorator_ebin_semi(self, ax: plt.Axes, df: pd.DataFrame) -> None:
        """Decorator to add hard/soft binary line and statistics"""
        ax.axhline(y=1, linestyle="--", color="darkred", linewidth=1.5)
        hard_num = np.sum(df["is_hard_binary"])
        soft_num = len(df) - hard_num
        hard_frac, soft_frac = (
            (hard_num / len(df), soft_num / len(df)) if len(df) > 0 else (0.0, 0.0)
        )

        xmax = ax.get_xlim()[1]
        with mpl.rc_context(**self.config.fixed_width_font_context):
            ax.text(
                xmax,
                1.0,
                f"{hard_num}\n{hard_frac:.1%} hard",
                color="darkred",
                ha="right",
                va="bottom",
            )
            ax.text(
                xmax,
                0.9,
                f"{soft_frac:.1%} soft\n{soft_num}",
                color="darkred",
                ha="right",
                va="top",
            )

    @log_time(logger)
    def create_ebind_semi_plot_density(self, binary_df_at_t: pd.DataFrame, simu_name: str) -> None:
        """Create binding energy vs semi-major axis plot"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col="Bin A[au]",
            y_col="Ebind/kT",
            log_scale=(True, True),
            filename_var_part="ebind_vs_a_loglog",
            custom_ax_joint_decorator=self._decorator_ebin_semi,
        )

    @log_time(logger)
    def create_ebind_semi_plot_jpg_compact_object_only(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create binding energy vs semi-major axis plot for compact objects only"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col="Bin A[au]",
            y_col="Ebind/kT",
            log_scale=(True, True),
            filename_var_part="ebind_vs_a_loglog_compact_objects_only",
            custom_ax_decorator=self._decorator_ebin_semi,
        )

    @log_time(logger)
    def create_ecc_semi_plot_density(self, binary_df_at_t: pd.DataFrame, simu_name: str) -> None:
        """Create eccentricity vs semi-major axis plot"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col="Bin A[au]",
            y_col="Bin ECC",
            log_scale=(True, False),
            filename_var_part="ecc_vs_a",
        )

    @log_time(logger)
    def create_ecc_semi_plot_jpg_compact_object_only(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create eccentricity vs semi-major axis plot for compact objects only"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col="Bin A[au]",
            y_col="Bin ECC",
            log_scale=(True, False),
            filename_var_part="ecc_vs_a_compact_objects_only",
        )

    @log_time(logger)
    def create_ecc_semi_plot_jpg_compact_object_only_loglog(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create eccentricity vs semi-major axis plot (log-log) for compact objects only"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col="Bin A[au]",
            y_col="Bin ECC",
            log_scale=(True, True),
            filename_var_part="ecc_vs_a_loglog_compact_objects_only",
        )

    @log_time(logger)
    def create_taugw_semi_plot_jpg_compact_object_only(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create GW merger timescale vs semi-major axis plot for compact objects only"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col="Bin A[au]",
            y_col="tau_gw[Myr]",
            log_scale=(True, True),
            filename_var_part="taugw_vs_a_compact_objects_only",
        )

    @log_time(logger)
    def create_mtot_distance_plot_density(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create total mass vs distance density plot"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col="Distance_to_cluster_center[pc]",
            y_col="total_mass[solar]",
            log_scale=(True, True),
            ylim_key="M",
            filename_var_part="mtot_vs_distance_loglog",
        )

    @log_time(logger)
    def create_mtot_distance_plot_jpg_compact_object_only(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create total mass vs distance plot for compact objects only"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col="Distance_to_cluster_center[pc]",
            y_col="total_mass[solar]",
            log_scale=(True, True),
            ylim_key="M",
            filename_var_part="mtot_vs_distance_loglog_compact_objects_only",
        )

    @log_time(logger)
    def create_semi_distance_plot_density(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create semi-major axis vs distance density plot"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col="Distance_to_cluster_center[pc]",
            y_col="Bin A[au]",
            log_scale=(True, True),
            filename_var_part="a_vs_distance",
        )

    @log_time(logger)
    def create_semi_distance_plot_jpg_compact_object_only(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create semi-major axis vs distance plot for compact objects only"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col="Distance_to_cluster_center[pc]",
            y_col="Bin A[au]",
            log_scale=(True, True),
            filename_var_part="a_vs_distance_compact_objects_only",
        )

    @log_time(logger)
    def create_bin_vx_x_plot_density(self, binary_df_at_t: pd.DataFrame, simu_name: str) -> None:
        """Create binary vx vs x density plot"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col="Bin cm X [pc]",
            y_col="Bin cm V1",
            log_scale=(False, False),
            filename_var_part="bin_vx_vs_x",
            xlim_key="position_pc_lim",
            ylim_key="velocity_kmps_lim",
        )

    @log_time(logger)
    def create_bin_vx_x_plot_jpg_compact_object_only(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create binary vx vs x plot for compact objects only"""
        self._create_base_jpg_plot_compact_object_only(
            binary_df_at_t,
            simu_name=simu_name,
            x_col="Bin cm X [pc]",
            y_col="Bin cm V1",
            log_scale=(False, False),
            xlim_key="position_pc_lim",
            ylim_key="velocity_kmps_lim",
            filename_var_part="bin_vx_vs_x_compact_objects_only",
        )

    def create_bin_peri_r1r2_plot_density(
        self, binary_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create periapsis vs sum of radii density plot"""
        self._create_jointplot_density(
            df_at_t=binary_df_at_t,
            simu_name=simu_name,
            x_col="sum_of_radius[au]",
            y_col="peri[au]",
            log_scale=(True, True),
            filename_var_part="peri_vs_r1r2_loglog",
        )
