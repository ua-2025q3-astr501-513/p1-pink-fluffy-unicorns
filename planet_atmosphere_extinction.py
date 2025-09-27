''' Calculate extinction due to (spherical) exoplanet atmosphere '''
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

#Mimicking Earth Atmosphere density profile for h < 80km
# Earth density profile (from engineeringtoolbox.com) (height (m), density (kg/m^3))
den_prof = [
    [0,1.225],[1000,1.112],[2000,1.007],[3000,0.9093],[4000,0.8194],
    [5000,0.7364],[6000,0.6601],[7000,0.5900],[8000,0.5258],[9000,0.4671],
    [10000,0.4135],[15000,0.1948],[20000,0.08891],[25000,0.04008],[30000,0.01841],
    [40000,0.003996],[50000,0.001027],[60000,0.0003097],[70000,0.00008283],[80000,0.00001846]
]

# Convert to arrays (once)
heights, densities = np.array(den_prof).T
heights_cm = heights * 100.0     # m -> cm
dens_gcm3 = densities * 1e-3     # kg/m^3 -> g/cm^3

# Build interpolator (once)
_density_cgs_interp = interp1d(
    heights_cm, dens_gcm3,
    kind="linear", bounds_error=False, fill_value="extrapolate"
)

def density_height(h_cm):
    """Return atmospheric density (g/cm³) at height h_cm (in cm).
       Returns NaN if h is outside 0–8e6 cm (~80 km)."""
    if np.any((h_cm < 0) | (h_cm > 8e6)):
        return np.nan
    return _density_cgs_interp(h_cm)
    
def density_proj_dist(d_cm, r_planet):
    """Return atmospheric density (g/cm³) at projected distance d_cm (in cm)."""
    h = np.sqrt(d_cm**2 + r_planet**2) - r_planet
    return density_height(h)

#opacity from ARCiS table for H20 and O2 with 273K 0.1 bar condition
data = np.load("./opacity_data.npz")
wavelengths = data["wavelengths"]
opacities = data["opacities"]

_opacity_interp = interp1d(
    wavelengths, opacities,
    kind="linear", bounds_error=False, fill_value="extrapolate"
)

def get_opacity(lamb):
    return _opacity_interp(lamb)

def extinction_sphere(h, lamb, r_planet): 
    #h, r_planet in cm, lamb in um 

    if h / r_planet * 6e8 > 8e6: # if the normalized height is higher than 80km -> it gives nan
        return np.nan

    kappa = get_opacity(lamb)

    max_height = r_planet + 80/6000*r_planet #~80km in the Earth (r~6000km) atmosphere
    integral_length = np.sqrt(max_height**2 - r_planet**2)
    res, err = quad(density_proj_dist, -1*integral_length, integral_length, args = (r_planet,))
    tau = kappa * res
    
    return np.exp(-1*tau)
