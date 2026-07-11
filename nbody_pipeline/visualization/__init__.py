"""Visualization tools for creating plots and figures"""

from nbody_pipeline.visualization.base import (
    BaseVisualizer,
    BaseHDF5Visualizer,
    HDF5Visualizer,
    BaseContinousFileVisualizer,
    set_mpl_fonts,
    add_grid,
)
from nbody_pipeline.visualization.single_star import SingleStarVisualizer
from nbody_pipeline.visualization.binary_star import BinaryStarVisualizer
from nbody_pipeline.visualization.lagrangian import LagrVisualizer
from nbody_pipeline.visualization.galactic_orbit import GalacticOrbitVisualizer
from nbody_pipeline.visualization.collision import CollCoalVisualizer
from nbody_pipeline.visualization.particle_history import ParticleHistoryVisualizer
from nbody_pipeline.visualization.purge import PlotPurger, PurgeResult

__all__ = [
    "BaseVisualizer",
    "BaseHDF5Visualizer",
    "HDF5Visualizer",
    "BaseContinousFileVisualizer",
    "SingleStarVisualizer",
    "BinaryStarVisualizer",
    "LagrVisualizer",
    "GalacticOrbitVisualizer",
    "CollCoalVisualizer",
    "ParticleHistoryVisualizer",
    "PlotPurger",
    "PurgeResult",
    "set_mpl_fonts",
    "add_grid",
]
