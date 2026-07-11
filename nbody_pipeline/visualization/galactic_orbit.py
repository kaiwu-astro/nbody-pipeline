"""Galactic-orbit visualization tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from nbody_pipeline.visualization.base import BaseContinousFileVisualizer, add_grid

logger = logging.getLogger(__name__)


class GalacticOrbitVisualizer(BaseContinousFileVisualizer):
    """Create static and interactive plots for cluster galactic orbits."""

    def _orbit_config(self) -> dict[str, Any]:
        defaults = {"time_color_max_myr": 500.0}
        user_config = getattr(self.config, "galactic_orbit", {}) or {}
        return {**defaults, **user_config}

    def create_projection_plot(self, df: pd.DataFrame, simu_name: str) -> Path:
        """Create XY, YZ, and ZX projection scatter plots."""
        save_pdf_path = (
            Path(self.config.plot_dir)
            / f"{self.config.figname_prefix[simu_name]}_galactic_orbit_projection.pdf"
        )
        save_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        color_max = float(self._orbit_config()["time_color_max_myr"])
        norm = Normalize(vmin=0.0, vmax=color_max)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        projections = [
            ("RG(1)", "RG(2)", "XY"),
            ("RG(2)", "RG(3)", "YZ"),
            ("RG(3)", "RG(1)", "ZX"),
        ]
        scatter = None
        for ax, (x_col, y_col, title) in zip(axes, projections):
            scatter = ax.scatter(
                df[x_col],
                df[y_col],
                c=df["Time[Myr]"],
                cmap="rainbow",
                norm=norm,
                s=12,
                alpha=0.85,
                linewidths=0,
            )
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_aspect("equal", adjustable="datalim")
            add_grid(ax)

        if scatter is None:
            scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap="rainbow")
        else:
            scalar_mappable = scatter
        colorbar = fig.colorbar(scalar_mappable, ax=axes, shrink=0.85)
        colorbar.set_label("Time [Myr]")
        fig.suptitle(simu_name)
        fig.savefig(save_pdf_path)
        self._close_figure(fig)
        return save_pdf_path

    def create_interactive_3d_html(self, df: pd.DataFrame, simu_name: str) -> Path | None:
        """Create a self-contained Plotly 3D scatter HTML, if Plotly is installed."""
        save_html_path = (
            Path(self.config.plot_dir)
            / f"{self.config.figname_prefix[simu_name]}_galactic_orbit_3d.html"
        )
        save_html_path.parent.mkdir(parents=True, exist_ok=True)
        color_max = float(self._orbit_config()["time_color_max_myr"])

        try:
            import plotly.graph_objects as go
        except ImportError:
            logger.warning("Plotly is unavailable; creating static 3D galactic-orbit PDF fallback")
            self.create_static_3d_fallback(df, simu_name)
            return None

        customdata = np.stack(
            [
                df["TTOT"].to_numpy(dtype=float),
                df["Time[Myr]"].to_numpy(dtype=float),
                df["VG(1)"].to_numpy(dtype=float),
                df["VG(2)"].to_numpy(dtype=float),
                df["VG(3)"].to_numpy(dtype=float),
            ],
            axis=-1,
        )
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=df["RG(1)"],
                    y=df["RG(2)"],
                    z=df["RG(3)"],
                    mode="markers",
                    marker={
                        "size": 3,
                        "color": df["Time[Myr]"],
                        "colorscale": "Rainbow",
                        "cmin": 0,
                        "cmax": color_max,
                        "colorbar": {"title": "Time [Myr]"},
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "RG=(%{x:.3g}, %{y:.3g}, %{z:.3g})<br>"
                        "TTOT=%{customdata[0]:.6g}<br>"
                        "Time=%{customdata[1]:.3f} Myr<br>"
                        "VG=(%{customdata[2]:.3g}, %{customdata[3]:.3g}, "
                        "%{customdata[4]:.3g})<extra></extra>"
                    ),
                )
            ]
        )
        fig.update_layout(
            title=simu_name,
            scene={
                "xaxis_title": "RG(1)",
                "yaxis_title": "RG(2)",
                "zaxis_title": "RG(3)",
            },
        )
        fig.write_html(save_html_path, include_plotlyjs=True)
        return save_html_path

    def create_static_3d_fallback(self, df: pd.DataFrame, simu_name: str) -> Path:
        """Create a Matplotlib 3D PDF fallback when Plotly cannot be imported."""
        save_pdf_path = (
            Path(self.config.plot_dir)
            / f"{self.config.figname_prefix[simu_name]}_galactic_orbit_3d.pdf"
        )
        save_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        color_max = float(self._orbit_config()["time_color_max_myr"])
        norm = Normalize(vmin=0.0, vmax=color_max)
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            df["RG(1)"],
            df["RG(2)"],
            df["RG(3)"],
            c=df["Time[Myr]"],
            cmap="rainbow",
            norm=norm,
            s=10,
            alpha=0.85,
        )
        ax.set_xlabel("RG(1)")
        ax.set_ylabel("RG(2)")
        ax.set_zlabel("RG(3)")
        ax.set_title(simu_name)
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
        colorbar.set_label("Time [Myr]")
        fig.savefig(save_pdf_path)
        self._close_figure(fig)
        return save_pdf_path

    def _close_figure(self, fig: plt.Figure) -> None:
        try:
            __IPYTHON__
            if self.config.close_figure_in_ipython:
                plt.close(fig)
        except NameError:
            plt.close(fig)
