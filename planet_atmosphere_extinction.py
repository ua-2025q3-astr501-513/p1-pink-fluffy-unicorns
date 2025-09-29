import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

class ExoplanetAtmosphere:
    def __init__(self, opacity_file="./opacity_data.npz"):
        """
        Initialize an exoplanet atmosphere model with an Earth-like density profile
        and wavelength-dependent opacities.

        Parameters
        ----------
        opacity_file : str
            Path to a .npz file containing:
              - 'wavelengths': array of wavelengths [µm]
              - 'opacities'  : array of opacities [cm^2/g]
        """

        # --- Earth atmosphere density profile (up to 80 km) ---
        den_prof = [
            [0, 1.225], [1000, 1.112], [2000, 1.007], [3000, 0.9093],
            [4000, 0.8194], [5000, 0.7364], [6000, 0.6601], [7000, 0.5900],
            [8000, 0.5258], [9000, 0.4671], [10000, 0.4135], [15000, 0.1948],
            [20000, 0.08891], [25000, 0.04008], [30000, 0.01841], [40000, 0.003996],
            [50000, 0.001027], [60000, 0.0003097], [70000, 0.00008283],
            [80000, 0.00001846]
        ]

        heights, densities = np.array(den_prof).T
        heights_cm = heights * 100.0      # m → cm
        dens_gcm3 = densities * 1e-3      # kg/m³ → g/cm³

        # Interpolator for density profile
        self._density_interp = interp1d(
            heights_cm, dens_gcm3,
            kind="linear", bounds_error=False, fill_value="extrapolate"
        )

        # Load opacity data
        data = np.load(opacity_file)
        wavelengths = data["wavelengths"]
        opacities = data["opacities"]

        # Interpolator for opacity
        self._opacity_interp = interp1d(
            wavelengths, opacities,
            kind="linear", bounds_error=False, fill_value="extrapolate"
        )

    # --- Methods ---

    def density_height(self, h_cm):
        """Atmospheric density [g/cm³] at height h_cm [cm].
           Returns NaN if h outside 0–8e6 cm (~80 km)."""
        if np.any((h_cm < 0) | (h_cm > 8e6)):
            return np.nan
        return self._density_interp(h_cm)

    def density_proj_dist(self, d_cm, r_cm, r_planet):
        """Atmospheric density [g/cm³] at projected distance d_cm [cm]."""
        h = np.sqrt(d_cm**2 + r_cm**2) - r_planet
        return self.density_height(h)

    def get_opacity(self, lamb_um):
        """Return opacity [cm²/g] at wavelength lamb_um [µm]."""
        return self._opacity_interp(lamb_um)

    def extinction_sphere(self, h_cm, lamb_um, r_planet_cm):
        """
        Calculate extinction factor exp(-tau) through a spherical atmosphere.

        Parameters
        ----------
        h_cm : float
            Altitude [cm] above surface.
        lamb_um : float
            Wavelength [µm].
        r_planet_cm : float
            Planet radius [cm].

        Returns
        -------
        extinction : float
            Transmission factor exp(-tau). NaN if above cutoff height.
        """
        # cutoff condition
        if h_cm / r_planet_cm * 6e8 > 8e6:
            return np.nan

        kappa = self.get_opacity(lamb_um)

        max_height = r_planet_cm + 80 / 6000 * r_planet_cm  # ~80 km scaled to Earth radius
        arg = max_height**2 - (r_planet_cm+h_cm)**2
        if arg <= 0:
            return np.nan
        integral_length = np.sqrt(arg)

        res, _ = quad(
            self.density_proj_dist,
            -integral_length, integral_length,
            args=(r_planet_cm+h_cm, r_planet_cm,)
        )

        tau = kappa * res
        return np.exp(-tau)
