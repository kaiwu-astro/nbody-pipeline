"""Single star visualization tools"""

import logging
import os
from typing import Callable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dragon3_pipelines.utils import log_time
from dragon3_pipelines.analysis.galactic_energy_angular_momentum import E_GAL_COL, L_Z_GAL_COL
from dragon3_pipelines.visualization.base import BaseHDF5Visualizer, add_grid
from dragon3_pipelines.visualization.purge import PlotPurger, PurgeResult

logger = logging.getLogger(__name__)


class SingleStarVisualizer(BaseHDF5Visualizer):
    """Visualizer for single star data"""

    ORBITAL_X_R_COL = "x_R [pc]"
    ORBITAL_X_T_COL = "x_T [pc]"
    ORBITAL_X_L_COL = "x_L [pc]"
    GALACTIC_E_LZ_FILENAME_VAR_PART = "galactic_E_vs_Lz"

    def _save_position_figure(self, fig: plt.Figure, ax: plt.Axes, save_jpg_path: str) -> None:
        """Save position plots with fixed canvas size and square data axes."""
        ax.set_aspect("equal", adjustable="box")
        fig.subplots_adjust(left=0.18, right=0.96, bottom=0.18, top=0.82)
        with mpl.rc_context({"savefig.bbox": None}):
            fig.savefig(save_jpg_path, transparent=False)

    def galactic_energy_angular_momentum_plot_jpg_path(
        self, df_at_t: pd.DataFrame, simu_name: str
    ) -> str:
        """Return the JPG path for the snapshot galactic E-vs-Lz plot."""
        ttot = df_at_t["TTOT"].iloc[0]
        return (
            f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}"
            f"output_ttot_{ttot}_{self.GALACTIC_E_LZ_FILENAME_VAR_PART}.jpg"
        )

    def purge(
        self,
        target: str,
        simu_name: str | None = None,
        plot_dir: str | os.PathLike[str] | None = None,
        filename_suffix: str | None = None,
        yes: bool = False,
    ) -> PurgeResult:
        """Purge single-star HDF5 JPG outputs for a target."""
        qualified_target = target if target.startswith("single.") else f"single.{target}"
        return PlotPurger(self.config).purge(
            qualified_target,
            simu_name=simu_name,
            plot_dir=plot_dir,
            filename_suffix=filename_suffix,
            yes=yes,
        )

    @log_time(logger)
    def create_mass_distance_plot_density(
        self, single_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create mass-distance relationship plot"""
        self._create_jointplot_density(
            df_at_t=single_df_at_t,
            simu_name=simu_name,
            x_col="Distance_to_cluster_center[pc]",
            y_col="M",
            log_scale=(True, True),
            filename_var_part="mass_vs_distance_loglog",
        )

    @log_time(logger)
    def create_vx_x_plot_density(self, single_df_at_t: pd.DataFrame, simu_name: str) -> None:
        """Create velocity-position plot"""
        self._create_jointplot_density(
            df_at_t=single_df_at_t,
            simu_name=simu_name,
            x_col="X [pc]",
            y_col="V1",
            log_scale=(False, False),
            filename_var_part="allstar_vx_vs_x",
            xlim_key="position_pc_lim",
            ylim_key="velocity_kmps_lim",
        )

    @log_time(logger)
    def create_CMD_plot_density(self, single_df_at_t: pd.DataFrame, simu_name: str) -> None:
        """Create color-magnitude diagram"""

        def _custom_decorator(ax: plt.Axes, df: pd.DataFrame) -> None:
            ax.invert_xaxis()

        self._create_jointplot_density(
            df_at_t=single_df_at_t,
            simu_name=simu_name,
            x_col="Teff*",
            y_col="L*",
            log_scale=(True, True),
            filename_var_part="L_vs_Teff_loglog",
            custom_ax_joint_decorator=_custom_decorator,
        )

    @log_time(logger)
    def create_galactic_energy_angular_momentum_plot_jpg(
        self, df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create a snapshot galactic total-energy versus ``L_z`` plot."""
        save_jpg_path = self.galactic_energy_angular_momentum_plot_jpg_path(df_at_t, simu_name)
        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return

        finite_mask = np.isfinite(df_at_t[L_Z_GAL_COL]) & np.isfinite(df_at_t[E_GAL_COL])
        plot_df = df_at_t.loc[finite_mask]
        if plot_df.empty:
            logger.warning("Skipping galactic E-vs-Lz plot because no finite points are available")
            return

        plot_config = getattr(self.config, "galactic_energy_angular_momentum", {}) or {}
        percentile_limits = plot_config.get("percentile_limits", [0.1, 99.9])
        low_pct, high_pct = float(percentile_limits[0]), float(percentile_limits[1])
        xlim = tuple(np.percentile(plot_df[L_Z_GAL_COL].to_numpy(dtype=float), [low_pct, high_pct]))
        ylim = tuple(np.percentile(plot_df[E_GAL_COL].to_numpy(dtype=float), [low_pct, high_pct]))

        ttot = df_at_t["TTOT"].iloc[0]
        tmyr = df_at_t["Time[Myr]"].iloc[0]
        t_over_tcr0 = df_at_t["TTOT/TCR0"].iloc[0]
        t_over_trh0 = df_at_t["TTOT/TRH0"].iloc[0]

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=plot_df,
                x=L_Z_GAL_COL,
                y=E_GAL_COL,
                marker=".",
                lw=0,
                s=2,
                color="white",
                alpha=0.25,
                ax=ax,
            )
            self.decorate_jointfig(
                ax,
                plot_df,
                L_Z_GAL_COL,
                E_GAL_COL,
                xlim,
                ylim,
                simu_name,
                ttot,
                tmyr,
                t_over_tcr0,
                t_over_trh0,
                highlight_outlier=False,
            )
            add_grid(ax)
            fig.savefig(save_jpg_path, transparent=False)
            try:
                __IPYTHON__
                if self.config.close_figure_in_ipython:
                    plt.close(fig)
            except NameError:
                plt.close(fig)

    @log_time(logger)
    def create_position_plot_jpg(
        self,
        single_df_at_t: pd.DataFrame,
        simu_name: str,
        filename_suffix: Optional[str] = None,
        extra_data_handler: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        extra_ax_handler: Optional[Callable[[plt.Axes], None]] = None,
        custom_ax_decorator: Optional[Callable[[plt.Axes], None]] = None,
        uniform_color_and_size: bool = False,
    ) -> None:
        """Create position scatter plot"""
        ttot = single_df_at_t["TTOT"].iloc[0]
        tmyr = single_df_at_t["Time[Myr]"].iloc[0]
        t_over_tcr0 = single_df_at_t["TTOT/TCR0"].iloc[0]
        t_over_trh0 = single_df_at_t["TTOT/TRH0"].iloc[0]
        if extra_data_handler is not None:
            single_df_at_t = extra_data_handler(single_df_at_t)
        if filename_suffix is None:
            save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_x1_vs_x2.jpg"
        else:
            save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_x1_vs_x2_{filename_suffix}.jpg"

        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return
        color_rgb_darker = (
            self.teff_to_rgb_converter.get_rgb(single_df_at_t["Teff*"].values)
            * self.luminosity_to_plot_alpha(single_df_at_t["L*"].values)[:, np.newaxis]
        )
        if not uniform_color_and_size:
            size = np.sqrt(single_df_at_t["R*"])
            color = color_rgb_darker
        else:
            size = 10
            color = "white"
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots()
            ax = sns.scatterplot(
                data=single_df_at_t,
                x="X [pc]",
                y="Y [pc]",
                marker=".",
                lw=0,
                s=size,
                color=color,
                ax=ax,
            )
            self.decorate_jointfig(
                ax,
                single_df_at_t,
                "X [pc]",
                "Y [pc]",
                self.config.limits["position_pc_lim"],
                self.config.limits["position_pc_lim"],
                simu_name,
                ttot,
                tmyr,
                t_over_tcr0,
                t_over_trh0,
            )
            if extra_ax_handler is not None:
                extra_ax_handler(ax)
            if custom_ax_decorator is not None:
                custom_ax_decorator(ax)
            self._save_position_figure(fig, ax, save_jpg_path)
            try:
                __IPYTHON__
                if self.config.close_figure_in_ipython:
                    plt.close(fig)
            except NameError:
                plt.close(fig)

    @log_time(logger)
    def create_position_plot_wide_pc_jpg(
        self, single_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create wide position plot"""

        def _set_wide_pos_lim_pc(ax: plt.Axes) -> None:
            ax.set_xlim(*self.config.limits["position_pc_lim_MAX"])
            ax.set_ylim(*self.config.limits["position_pc_lim_MAX"])

        self.create_position_plot_jpg(
            single_df_at_t=single_df_at_t,
            simu_name=simu_name,
            filename_suffix="wide_pc",
            extra_ax_handler=_set_wide_pos_lim_pc,
            uniform_color_and_size=True,
        )

    def _orbital_frame_basis(
        self, scalar_row_at_t: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return orbital-frame unit vectors (e_R, e_T, e_L) from scalar RG/VG columns."""
        rg = np.array([scalar_row_at_t[f"RG({i})"] for i in range(1, 4)], dtype=float)
        vg = np.array([scalar_row_at_t[f"VG({i})"] for i in range(1, 4)], dtype=float)
        angular_momentum = np.cross(rg, vg)

        rg_norm = np.linalg.norm(rg)
        angular_momentum_norm = np.linalg.norm(angular_momentum)
        if (
            not np.isfinite(rg).all()
            or not np.isfinite(vg).all()
            or not np.isfinite(rg_norm)
            or not np.isfinite(angular_momentum_norm)
            or rg_norm == 0
            or angular_momentum_norm == 0
        ):
            raise ValueError("Degenerate or non-finite orbital-frame RG/VG vectors")

        e_r = rg / rg_norm
        e_l = angular_momentum / angular_momentum_norm
        e_t = np.cross(e_l, e_r)
        return e_r, e_t, e_l

    def _single_df_with_orbital_frame_positions(
        self, single_df_at_t: pd.DataFrame, scalar_row_at_t: pd.Series
    ) -> pd.DataFrame | None:
        """Project cluster-centered particle positions onto the snapshot orbital frame."""
        try:
            e_r, e_t, e_l = self._orbital_frame_basis(scalar_row_at_t)
            positions = single_df_at_t[["X [pc]", "Y [pc]", "Z [pc]"]].to_numpy(dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            ttot = single_df_at_t["TTOT"].iloc[0] if "TTOT" in single_df_at_t else "unknown"
            logger.warning("Skipping orbital-frame position plot at TTOT=%s: %s", ttot, exc)
            return None

        projected_df = single_df_at_t.copy()
        projected_df[self.ORBITAL_X_R_COL] = positions @ e_r
        projected_df[self.ORBITAL_X_T_COL] = positions @ e_t
        projected_df[self.ORBITAL_X_L_COL] = positions @ e_l
        return projected_df

    def _create_position_plot_orbital_wide_pc_jpg(
        self,
        single_df_at_t: pd.DataFrame,
        scalar_row_at_t: pd.Series,
        simu_name: str,
        y_col: str,
        filename_var_part: str,
    ) -> None:
        """Create a wide single-star position plot in the snapshot orbital frame."""
        projected_df = self._single_df_with_orbital_frame_positions(single_df_at_t, scalar_row_at_t)
        if projected_df is None:
            return

        ttot = projected_df["TTOT"].iloc[0]
        tmyr = projected_df["Time[Myr]"].iloc[0]
        t_over_tcr0 = projected_df["TTOT/TCR0"].iloc[0]
        t_over_trh0 = projected_df["TTOT/TRH0"].iloc[0]
        save_jpg_path = (
            f"{self.config.plot_dir}/jpg/"
            f"{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_{filename_var_part}.jpg"
        )
        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return

        position_pc_lim_max = self.config.limits["position_pc_lim_MAX"]
        with plt.style.context("dark_background"):
            fig, ax = plt.subplots()
            ax = sns.scatterplot(
                data=projected_df,
                x=self.ORBITAL_X_T_COL,
                y=y_col,
                marker=".",
                lw=0,
                s=10,
                color="white",
                ax=ax,
            )
            self.decorate_jointfig(
                ax,
                projected_df,
                self.ORBITAL_X_T_COL,
                y_col,
                position_pc_lim_max,
                position_pc_lim_max,
                simu_name,
                ttot,
                tmyr,
                t_over_tcr0,
                t_over_trh0,
            )
            self._save_position_figure(fig, ax, save_jpg_path)
            try:
                __IPYTHON__
                if self.config.close_figure_in_ipython:
                    plt.close(fig)
            except NameError:
                plt.close(fig)

    @log_time(logger)
    def create_position_plot_orbital_xT_xR_wide_pc_jpg(
        self, single_df_at_t: pd.DataFrame, scalar_row_at_t: pd.Series, simu_name: str
    ) -> None:
        """Create a wide position plot of x_T versus x_R in the orbital frame."""
        self._create_position_plot_orbital_wide_pc_jpg(
            single_df_at_t=single_df_at_t,
            scalar_row_at_t=scalar_row_at_t,
            simu_name=simu_name,
            y_col=self.ORBITAL_X_R_COL,
            filename_var_part="orbital_xT_vs_xR_wide_pc",
        )

    @log_time(logger)
    def create_position_plot_orbital_xT_xL_wide_pc_jpg(
        self, single_df_at_t: pd.DataFrame, scalar_row_at_t: pd.Series, simu_name: str
    ) -> None:
        """Create a wide position plot of x_T versus x_L in the orbital frame."""
        self._create_position_plot_orbital_wide_pc_jpg(
            single_df_at_t=single_df_at_t,
            scalar_row_at_t=scalar_row_at_t,
            simu_name=simu_name,
            y_col=self.ORBITAL_X_L_COL,
            filename_var_part="orbital_xT_vs_xL_wide_pc",
        )

    @log_time(logger)
    def create_position_plot_hightlight_compact_objects_jpg(
        self,
        single_df_at_t: pd.DataFrame,
        simu_name: str,
        filename_suffix: Optional[str] = None,
        extra_data_handler: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        extra_ax_handler: Optional[Callable[[plt.Axes], None]] = None,
        custom_ax_decorator: Optional[Callable[[plt.Axes], None]] = None,
    ) -> None:
        """Create position plot highlighting compact objects"""
        ttot = single_df_at_t["TTOT"].iloc[0]
        tmyr = single_df_at_t["Time[Myr]"].iloc[0]
        t_over_tcr0 = single_df_at_t["TTOT/TCR0"].iloc[0]
        t_over_trh0 = single_df_at_t["TTOT/TRH0"].iloc[0]

        processed_df = single_df_at_t
        if extra_data_handler is not None:
            processed_df = extra_data_handler(single_df_at_t)

        if filename_suffix is None:
            save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_x1_vs_x2_highlight_compact_objects.jpg"
        else:
            save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_x1_vs_x2_highlight_compact_objects_{filename_suffix}.jpg"

        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return

        fig, ax = plt.subplots()

        df_hewd = processed_df[processed_df["KW"] == 10]
        df_cowd = processed_df[processed_df["KW"] == 11]
        df_onewd = processed_df[processed_df["KW"] == 12]
        df_ns = processed_df[processed_df["KW"] == 13]
        df_bh = processed_df[processed_df["KW"] == 14]
        df_others = processed_df[~processed_df["KW"].isin([10, 11, 12, 13, 14])]

        if not df_others.empty:
            sns.scatterplot(
                data=df_others,
                x="X [pc]",
                y="Y [pc]",
                marker="o",
                lw=0,
                s=np.sqrt(df_others["R*"]),
                color="gray",
                ax=ax,
            )

        compact_object_plot_configs = [
            (df_hewd, self.config.marker_nofill_list[10], "blue", "HeWD", None, 1.5),
            (df_cowd, self.config.marker_nofill_list[11], "blue", "COWD", None, 1.5),
            (df_onewd, self.config.marker_nofill_list[12], "blue", "ONeWD", None, 1.5),
            (df_ns, self.config.marker_fill_list[13], "red", "NS", "green", 0),
            (df_bh, self.config.marker_fill_list[14], "black", "BH", "white", 0),
        ]

        for df_compact, marker, color, label, edgecolors, lw in compact_object_plot_configs:
            if not df_compact.empty:
                plot_kwargs = {
                    "data": df_compact,
                    "x": "X [pc]",
                    "y": "Y [pc]",
                    "marker": marker,
                    "s": 30,
                    "color": color,
                    "label": label,
                    "alpha": 0.7,
                    "edgecolors": edgecolors,
                    "lw": lw,
                    "ax": ax,
                }
                sns.scatterplot(**plot_kwargs)

        self.decorate_jointfig(
            ax,
            processed_df,
            "X [pc]",
            "Y [pc]",
            self.config.limits["position_pc_lim"],
            self.config.limits["position_pc_lim"],
            simu_name,
            ttot,
            tmyr,
            t_over_tcr0,
            t_over_trh0,
        )

        if extra_ax_handler is not None:
            extra_ax_handler(ax)

        if ax.legend_ is not None:
            ax.legend_.remove()

        if custom_ax_decorator is not None:
            custom_ax_decorator(ax)

        add_grid(ax)
        self._save_position_figure(fig, ax, save_jpg_path)
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(fig)
        except NameError:
            plt.close(fig)

    @log_time(logger)
    def create_position_plot_hightlight_compact_objects_wide_pc_jpg(
        self, single_df_at_t: pd.DataFrame, simu_name: str
    ) -> None:
        """Create wide position plot highlighting compact objects"""

        def _set_wide_pos_lim_pc(ax: plt.Axes) -> None:
            ax.set_xlim(*self.config.limits["position_pc_lim_MAX"])
            ax.set_ylim(*self.config.limits["position_pc_lim_MAX"])

        self.create_position_plot_hightlight_compact_objects_jpg(
            single_df_at_t=single_df_at_t,
            simu_name=simu_name,
            filename_suffix="wide_pc",
            extra_ax_handler=_set_wide_pos_lim_pc,
        )

    @log_time(logger)
    def create_color_CMD_jpg(
        self,
        single_df_at_t: pd.DataFrame,
        simu_name: str,
        extra_data_handler: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        extra_ax_handler: Optional[Callable[[plt.Axes], None]] = None,
    ) -> None:
        """Create color CMD plot"""
        ttot = single_df_at_t["TTOT"].iloc[0]
        tmyr = single_df_at_t["Time[Myr]"].iloc[0]
        t_over_tcr0 = single_df_at_t["TTOT/TCR0"].iloc[0]
        t_over_trh0 = single_df_at_t["TTOT/TRH0"].iloc[0]
        if extra_data_handler is not None:
            single_df_at_t = extra_data_handler(single_df_at_t)
        save_jpg_path = f"{self.config.plot_dir}/jpg/{self.config.figname_prefix[simu_name]}output_ttot_{ttot}_CMD.jpg"
        if self.config.skip_existing_plot and os.path.exists(save_jpg_path):
            logger.debug(f"Skip existing plot: {save_jpg_path}")
            return
        all_stellar_types_sorted = sorted(
            self.config.kw_to_stellar_type_verbose.values(), key=lambda x: int(x.split(":")[0])
        )

        ax = sns.scatterplot(
            data=single_df_at_t,
            x="Teff*",
            y="L*",
            hue="Stellar Type",
            hue_order=all_stellar_types_sorted,
            style="Stellar Type",
            palette=self.config.st_verbose_to_color,
            s=20,
            linewidths=0.8,
            markers=self.config.star_type_verbose_to_marker,
            legend="full",
        )
        ax.set(xscale="log", yscale="log")
        if extra_ax_handler is not None:
            extra_ax_handler(ax)
        self.decorate_jointfig(
            ax,
            single_df_at_t,
            "Teff*",
            "L*",
            self.config.limits["Teff*"],
            self.config.limits["L*"],
            simu_name,
            ttot,
            tmyr,
            t_over_tcr0,
            t_over_trh0,
        )

        legend_handles = []
        placeholder_marker_color = (0, 0, 0, 0.0)
        legend_marker_size = np.sqrt(20)
        legend_marker_edge_width = 0.8

        current_stellar_types = set(single_df_at_t["Stellar Type"].unique())
        for st_type in all_stellar_types_sorted:
            marker_shape = self.config.star_type_verbose_to_marker.get(st_type, "+")

            if st_type in current_stellar_types:
                label = st_type
                color = self.config.st_verbose_to_color.get(st_type, "black")
                mec = color
            else:
                label = " " * 8
                mec = placeholder_marker_color

            handle = mpl.lines.Line2D(
                [0],
                [0],
                marker=marker_shape,
                linestyle="None",
                label=label,
                markeredgecolor=mec,
                markersize=legend_marker_size,
                markeredgewidth=legend_marker_edge_width,
            )
            legend_handles.append(handle)

        with mpl.rc_context(**self.config.fixed_width_font_context):
            plt.legend(
                handles=legend_handles,
                title="Stellar Type",
                fontsize=12,
                title_fontsize=12,
                bbox_to_anchor=(1, 1),
                loc="upper left",
            )

        ax.invert_xaxis()
        ax.figure.savefig(save_jpg_path)
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(ax.figure)
        except NameError:
            plt.close(ax.figure)
