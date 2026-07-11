"""
Color conversion utilities for blackbody radiation
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union, Optional
from scipy.interpolate import interp1d
from rich.progress import track

try:
    from colour.colorimetry import SpectralDistribution, msds_to_XYZ, planck_law
    from colour.models import XYZ_to_sRGB
    import colour

    COLOUR_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    COLOUR_AVAILABLE = False


DEFAULT_TEFF_RGB_CACHE_PATH = str(Path("~/.cache/nbody_pipeline/teff_to_rgb.pkl").expanduser())


class BlackbodyColorConverter:
    """Convert blackbody radiation temperature to RGB colors"""

    def __init__(
        self,
        cache_path: Optional[str] = None,
    ):
        """
        Initialize the color converter.

        Args:
            cache_path: Path to cache file for RGB interpolators. Defaults to
                ``~/.cache/nbody_pipeline/teff_to_rgb.pkl``; the cache is
                recomputed automatically if missing.
        """
        if not COLOUR_AVAILABLE:
            raise ImportError(
                "The 'colour-science' package is required for BlackbodyColorConverter. "
                "Install it with: pip install colour-science"
            )

        self.cache_path = cache_path or DEFAULT_TEFF_RGB_CACHE_PATH
        self.r_interp: Optional[interp1d] = None
        self.g_interp: Optional[interp1d] = None
        self.b_interp: Optional[interp1d] = None
        self.prepare_rgb_interpolator()

    def get_blackbody_rgb_df(self) -> pd.DataFrame:
        """
        Calculate RGB values for different blackbody temperatures.

        Returns:
            DataFrame with columns: Teff, R, G, B
        """
        # Define temperature range
        temperatures = np.concatenate([np.arange(0, 20000, 10), np.arange(20000, 1000000, 1000)])

        # Wavelength range (visible spectrum: 380-780 nm)
        wavelengths = np.arange(380, 781, 5)

        # Store results
        results = []

        # Calculate RGB for each temperature
        for temp in track(temperatures, description="Computing RGB values..."):
            # Calculate blackbody radiation spectrum
            spd_data = {}
            for wavelength in wavelengths:
                spd_data[wavelength] = planck_law(wavelength * 1e-9, temp)

            # Create spectral distribution object
            spd = SpectralDistribution(spd_data)

            # Normalize spectrum for appropriate brightness
            spd.normalise()

            # Convert spectrum to XYZ tristimulus values
            XYZ = msds_to_XYZ(spd, method="integration")

            # Convert XYZ to sRGB
            RGB_linear = XYZ_to_sRGB(XYZ / 100)

            # Clip linear RGB to [0,1] range
            RGB_linear = np.clip(RGB_linear, 0, 1)

            # Apply sRGB gamma correction
            RGB = colour.cctf_encoding(RGB_linear)

            # Add result to list
            results.append([temp] + RGB.tolist())

        return pd.DataFrame(results, columns=["Teff", "R", "G", "B"])

    def prepare_rgb_interpolator(self) -> None:
        """
        Create interpolation functions from temperature to RGB.

        Loads from cache if available, otherwise computes and saves.
        """
        # Import here to avoid circular dependency
        from nbody_pipeline.utils.serialization import save, read

        if os.path.exists(self.cache_path):
            self.r_interp, self.g_interp, self.b_interp = read(self.cache_path)
        else:
            df = self.get_blackbody_rgb_df()
            df_sorted = df.sort_values("Teff")

            # Create three interpolation functions for R, G, B
            r_interp = interp1d(
                df_sorted["Teff"], df_sorted["R"], kind="cubic", fill_value="extrapolate"
            )
            g_interp = interp1d(
                df_sorted["Teff"], df_sorted["G"], kind="cubic", fill_value="extrapolate"
            )
            b_interp = interp1d(
                df_sorted["Teff"], df_sorted["B"], kind="cubic", fill_value="extrapolate"
            )

            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            save(self.cache_path, [r_interp, g_interp, b_interp])
            self.r_interp, self.g_interp, self.b_interp = r_interp, g_interp, b_interp

    def get_rgb(self, teff: Union[float, np.ndarray]) -> np.ndarray:
        """
        Get RGB color for specified temperature(s).

        Args:
            teff: Temperature value or array

        Returns:
            RGB values, shape (n, 3) or (3,)
        """
        r = np.clip(self.r_interp(teff), 0, 1)
        g = np.clip(self.g_interp(teff), 0, 1)
        b = np.clip(self.b_interp(teff), 0, 1)
        return np.array([r, g, b]).T

    def plot_colorbar(
        self, teff_min: float = 10, teff_max: float = 12000, step: float = 100
    ) -> None:
        """
        Visualize temperature to RGB color mapping.

        Args:
            teff_min: Minimum temperature
            teff_max: Maximum temperature
            step: Temperature step size
        """
        import matplotlib.pyplot as plt

        teff = np.arange(teff_min, teff_max, step)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(teff, np.ones(len(teff)), c=self.get_rgb(teff), s=100, marker="s")
        ax.set_xlabel("Teff (K)")
        ax.set_yticks([])  # Hide y-axis ticks
        ax.set_title("Black body temperature - RGB Color Mapping")
        plt.show()
