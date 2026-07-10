"""Collision and coalescence visualization tools"""

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dragon3_pipelines.io import load_GWTC_catalog
from dragon3_pipelines.visualization.base import BaseContinousFileVisualizer, add_grid

logger = logging.getLogger(__name__)


class CollCoalVisualizer(BaseContinousFileVisualizer):
    """Visualizer for collision and coalescence events"""

    def two_bh_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to include only BH-BH binaries.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame with only BH-BH systems
        """
        return df[(df["primary_stellar_type"] == 14) & (df["secondary_stellar_type"] == 14)]

    def two_cbo_fileter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to include only compact object binaries.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame with only compact object systems
        """
        return df[
            (df["primary_stellar_type"].isin(self.config.compact_object_KW))
            & (df["secondary_stellar_type"].isin(self.config.compact_object_KW))
        ]

    def create_mass_ratio_primary_plot_cbo(self, df: pd.DataFrame, simu_name: str) -> None:
        """
        Create mass ratio vs primary mass plot for compact object binaries.

        Args:
            df: Input DataFrame with merger/collision data
            simu_name: Simulation name
        """
        df_cbo = self.two_cbo_fileter(df)

        ax = sns.scatterplot(
            data=df_cbo,
            x="primary_mass[solar]",
            y="mass_ratio",
            hue="Stellar Type",
            style="Merger_type",
            alpha=0.5,
        )

        marker_size = mpl.rcParams["lines.markersize"]
        for i, row in df_cbo.iterrows():
            ax.text(
                row["primary_mass[solar]"],
                row["mass_ratio"] - 0.002 * marker_size,
                f"{row['Time[Myr]']:.2f}",
                fontsize=8,
                ha="center",
                va="top",
            )

        if len(df_cbo) > 0:
            ax.text(
                1.01,
                0,
                "Marked text is time[Myr]\n  of each event in simulations",
                transform=ax.transAxes,
                fontsize=10,
                ha="left",
                va="bottom",
                color="black",
                fontstyle="italic",
            )

        gwtc_catalog_csv = getattr(self.config, "gwtc_catalog_csv", None)
        if gwtc_catalog_csv:
            gwtc_df = load_GWTC_catalog(gwtc_catalog_csv)
            y_err_lower_abs = np.abs(gwtc_df["mass_ratio_lower"])
            y_err_upper_abs = gwtc_df["mass_ratio_upper"]

            ax.errorbar(
                x=gwtc_df["mass_1_source"],
                y=gwtc_df["mass_ratio"],
                yerr=[y_err_lower_abs, y_err_upper_abs],
                fmt="o",
                markersize=5,
                capsize=3,
                alpha=0.2,
                color="gray",
                label="GWTC Data",
            )
        else:
            logger.warning(
                "paths.gwtc_catalog_csv not configured; skipping GWTC overlay "
                "(see config.example.yaml)."
            )

        ax.fill_betweenx(y=ax.get_ylim(), x1=40, x2=150, color="pink", alpha=0.2)

        ax.set(xscale="log", xlim=(2.5, 500), ylim=(0, 1.05))
        add_grid(ax)

        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

        ax.figure.savefig(
            f"{self.config.plot_dir}/{self.config.figname_prefix[simu_name]}_merger_mass_ratio_vs_primary_mass_2cbo.pdf"
        )
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(ax.figure)
        except NameError:
            plt.close(ax.figure)
