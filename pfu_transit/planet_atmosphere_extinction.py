from scipy.integrate import quad
import numpy as np
from multiprocessing.pool import ThreadPool as Pool
from astropy import constants as c
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def exponential_atmosphere(h, R):
    """
    Initialize an exoplanet atmosphere model with a exponential absorption profile.

    Parameters
    ----------
    h - height
    R - max height of the atmosphere
    """
    
    return (np.e/(np.e-1))*(np.exp(-1*h/R) - 1) + 1

class Integrate(object):
    """
    Class for integrating projected density profile for multiprocessing
    """

    def __init__(self, params):
        self.params = params
    
    def __call__(self, i):

        func, integral_length, r_planet_cm, h_cm_masked = self.params

        res, _ = quad(func,
            -integral_length[i], integral_length[i],
            args=(r_planet_cm+h_cm_masked[i], r_planet_cm,))

        return res, i

class ExoplanetAtmosphere:
    def __init__(self, opacity_file, density_file, planet_type = "gaseous", max_height = None):
        """
        Initialize an exoplanet atmosphere model with an Earth-like/Jupiter-like density profile
        and wavelength-dependent opacities.

        Parameters
        ----------
        opacity_file : str
            Path to a .npz file containing:
              - 'wavelengths': array of wavelengths [µm]
              - 'opacities'  : array of opacities [cm^2/g]
        """

        # Load opacity profile and Earth-like density profile
        data = np.load(opacity_file)
        dens = np.load(density_file)

        if planet_type == 'gaseous':
            self.scale_max_height = 3.2e8
            if max_height is None:
                self.max_height = 7000 / (c.R_jup.to("km").value)
            else:
                self.max_height = max_height
        elif planet_type == 'rocky':
            self.scale_max_height = 8e6
            if max_height is None:
                self.max_height = 80 / (c.R_earth.to("km").value)
            else:
                self.max_height = max_height
                
            
        # Create interpolators for opacity and density
        self._opacity_interp = interp1d(
        data["wavelengths"], data["opacities"],
        kind="linear", bounds_error=False, fill_value="extrapolate"
    )

        self._density_interp = interp1d(
        dens["arr_0"], dens["arr_1"], #radius in cm, density in g cm^-3
        kind="linear", bounds_error=False, fill_value="extrapolate"
    )

    # --- Methods ---

    def density_height(self, h_cm, r_planet_cm, planet = 'earth'):
        """Atmospheric density [g/cm³] at height h_cm [cm].
           Returns NaN if h outside 0–8e6 cm (~80 km)."""

        scaled_height = self.scale_max_height*h_cm/(self.max_height*r_planet_cm)

        if np.any((scaled_height < 0)):
            return np.nan
        elif np.any((scaled_height > self.scale_max_height)):
            return 0
        d = self._density_interp(scaled_height)
        return d

    def density_proj_dist(self, d_cm, r_cm, r_planet_cm):
        """Atmospheric density [g/cm³] at projected distance d_cm [cm]."""
        h = np.sqrt(d_cm**2 + r_cm**2) - r_planet_cm
        return self.density_height(h, r_planet_cm)

    def get_opacity(self, lamb_um):
        """Return opacity [cm²/g] at wavelength lamb_um [µm]."""

        return self._density_interp(lamb_um)

    def extinction_sphere(self, h, r_planet, lamb_um, threadcount=20):
        h_cm  = (h*c.R_sun).to("cm").value
        r_planet_cm  = (r_planet*c.R_sun).to("cm").value
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
        
        kappa = self.get_opacity(lamb_um)

        max_radius = r_planet_cm + self.max_height * r_planet_cm 
        
        arg = max_radius**2 - (r_planet_cm+h_cm)**2
        
        mask = (h_cm / r_planet_cm  < self.max_height) | (arg > 0)
        
        integral_length = np.sqrt(arg[mask])

        with Pool(threadcount) as pool:
            res, _ = zip(*tqdm(pool.imap(Integrate((self.density_proj_dist, integral_length, r_planet_cm, h_cm[mask])), range(len(integral_length))), total=len(integral_length)))

        res = np.asarray(res)

        tau = np.zeros(len(h_cm))
        tau[mask] = kappa * res

        profile = 1 - np.exp(-tau)
        return profile
