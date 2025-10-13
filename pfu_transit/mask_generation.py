import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Box2DKernel
from cv2 import distanceTransform, DIST_L2, DIST_MASK_PRECISE
from pfu_transit.planet_atmosphere_extinction import ExoplanetAtmosphere
from astropy import constants as c
from astropy import units as u
from multiprocessing.pool import ThreadPool as Pool
from tqdm import tqdm


class Eddington(object):
    '''
    Class for parallelizing calculation of limb darkening coefficients,
    an integration of the Eddington approximation
    '''
    def __init__(self, params):
        self.params = params
    
    def __call__(self, u):

        midpoints, freq = self.params

        y = integ_func(midpoints, freq, u)
        return np.sum(y*(midpoints[1] - midpoints[0])), u


def generate_exoplanet(framew = 2500,frameh = 2500, image_scale = 3/2500,  planet_radius = 0.5 * c.R_jup/c.R_sun,\
                            planet_oblateness= 0.1, planet_rot_obliquity = -5*np.pi/180, wavelength = 1, b = 0.3, max_height=1, planet_type="rocky", surface=True, no_atmosphere=False):
    """
    Generate an exoplanet mask with an Earth-like atmospheric profile.

    Parameters
    ----------
    framew - Frame width in pixels
    frameh - Frame height in pixels
    image_scale - Image scale in solar radii / px
    planet_radius - Planet radius in solar radii
    planet_oblateness - Planet oblateness
    planet_rot_obliquity - Obliquity of the rotational axis of the planet, assuming ellipticity by rotation
    wavelength - Wavelength observed in microns
    b - Offset of planet due to inclination from the center of the image, in projected solar radii
    max_height - Maximum height of the atmosphere relative to Earth's normalized maximum height
    planet_type - Planet type, rocky or gaseous
    surface - Changes definition of planet radius, whether to include or exclude atmosphere (for bodies without a surface)
    no_atmosphere - Removes atmosphere, overrides other parameters
    """

    if surface and not no_atmosphere:
        mask_radius = planet_radius*(1+max_height)
    else:
        mask_radius = planet_radius
    
    # Create exoplanet mask with parameters without atmosphere
    exoplanet_mask =  elliptical_mask([framew/2 - 0.5, frameh/2 - 0.5 - b/image_scale], framew, frameh,\
                                    mask_radius/image_scale, (1-planet_oblateness)*mask_radius/image_scale, p = planet_rot_obliquity)

    if not no_atmosphere:
        # Create ExoplanetAtmosphere class
        if planet_type == "rocky":
            exoplanet_atmosphere = ExoplanetAtmosphere("assets/opacity_data_earth.npz","assets/density_profile_earth.npz",  planet_type=planet_type, max_height=max_height)
    
        else:
            exoplanet_atmosphere = ExoplanetAtmosphere("assets/opacity_data_jupiter.npz","assets/density_profile_jupiter.npz", planet_type=planet_type, max_height=max_height)
    
    
        # Prepare function that generates profile
        def atmo_func(radii):
            return exoplanet_atmosphere.extinction_sphere(radii*image_scale, planet_radius, wavelength)
    
    
        # Generate atmospheric profile and add onto the mask 
        exoplanet_mask = add_atmosphere(exoplanet_mask, atmo_func, max_height*planet_radius/image_scale)

    return exoplanet_mask

def generate_star(framew = 2500,frameh = 2500, image_scale = 3/2500, stellar_radius = 1, stellar_oblateness = 0.1,\
                 stellar_rot_obliquity = 10*np.pi/180,  stellar_temperature = 5778, wavelength=1):

    """
    Generate a star mask assuming a blackbody and Eddington approximated limb darkening.

    Parameters
    ----------
    framew - Frame width in pixels
    frameh - Frame height in pixels
    image_scale - Image scale in solar radii / px
    stellar_radius - Stellar radius in solar radii
    stellar_oblateness - Stellar oblateness
    stellar_rot_obliquity - Obliquity of the rotational axis of the star, assuming ellipticity by rotation
    stellar_temperature - Temperature of the star in Kelvin
    wavelength - Wavelength observed in microns
    """
    # Create star mask with parameters without limb darkening or flux
    star_mask = elliptical_mask([framew/2 - 0.5, frameh/2 - 0.5], framew, frameh,\
                          stellar_radius/image_scale, (1-stellar_oblateness)*stellar_radius/image_scale, p = stellar_rot_obliquity )

    # Apply limb darkening and stellar properties to the mask
    star_mask = limb_darken(star_mask, stellar_radius, stellar_temperature, wavelength)

    return star_mask


def limb_darken(mask, stellar_radius, T, l):

    """
    Apply stellar properties and Eddington approximated limb darkening.

    Parameters
    ----------
    mask - Star mask
    stellar_radius - Stellar radius in solar radii
    T - Temperature of the star in Kelvin
    l - wavelength in microns
    """

    # Get frequency in dimensionless units (h*c/kT * lambda)
    freq = (1/3.9302)*(c.h * c.c/(c.k_B*T*u.K)).to(u.micron).value/l

    # Calculate expected luminosity based on stellar properties
    luminosity = ((((8 * c.h * (c.c*np.pi*stellar_radius*c.R_sun)**2)/((l*u.micron)**5))/(np.exp(freq) - 1)).to(u.erg/(u.s*u.micron))).value

    # Convert mask into integer units
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Create array representing distance from surface, and normalize to stellar radius
    skeleton = distanceTransform(mask, DIST_L2, DIST_MASK_PRECISE)
    R = np.max(skeleton)
    skeleton = skeleton.astype(np.float64)/R

    # Create arrays representing relative brightness and distance from surface
    radii = np.linspace(0,1, 2*int(R))[1:]
    brightness = np.linspace(1,0, 2*int(R))
    new_contour = np.copy(skeleton)

    # Obtain limb darkening coefficients
    eddington_values = eddington(radii, freq)

    print("Creating mask...")
    for i in tqdm(range(len(brightness)-1)):

        new_contour[(brightness[i] >= skeleton) & (skeleton >= brightness[i+1])] = eddington_values[i]

    return luminosity*new_contour/np.sum(new_contour)

def eddington(r, freq, Nint=100000, wavelength_limit = 100, threadcount=50):

    """
    Calculate Eddington approximated limb darkening.

    Parameters
    ----------
    r - distance from surface
    freq - dimensionless frequency
    Nint - Number of points for numerical integration
    wavelength_limit - Bound of integration in dimensionless units
    threadcount - Number of threads for multiprocessing
    """
    
    xlist = np.linspace(0, wavelength_limit, Nint)
    midpoints = 0.5*(xlist[1:] + xlist[:-1])
    u = np.cos(np.arcsin(r))

    print("Calculating limb darkening coefficients....")
    with Pool(threadcount) as pool:
        values, _ = zip(*tqdm(pool.imap(Eddington((midpoints, freq)), u), total=len(u)))

    return values

def integ_func(t, x, u):
    
    return (1/u)*np.exp(-1*t/u)*(x**3)/(-1+np.exp(x/np.sqrt(np.sqrt(0.75*(t+(2/3))))))

    
def add_atmosphere(mask, atmo_func, max_height):
    
    '''
    Add an exoplanet atmosphere with a certain profile to an existing exoplanet mask 

    Parameters
    ----------
    Mask : Image, populated with 1s and 0s

    atmo_func : Exoplanet atmosphere profile

    max_height: Maximum height of atmosphere (pixels)
    '''
    # Convert mask into integer units
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Create array representing distance from surface, and normalize to stellar radius
    skeleton = distanceTransform(mask, DIST_L2, DIST_MASK_PRECISE)

    # Create arrays representing relative brightness and distance from surface
    #radii = np.linspace(0,1, 2*int(R))[1:]
    height = np.linspace(0, max_height, int(2*max_height))
    #brightness = np.linspace(1,0, 2*int(R))
    new_contour = np.copy(skeleton.astype(np.float64))
    new_contour[skeleton > max_height] = 1
    # Obtain limb darkening coefficients
    atmo_values = atmo_func(height)

    print("Creating mask...")
    for i in tqdm(range(len(height) - 1)):
        
        new_contour[(skeleton <= height[i+1]) &  (skeleton > height[i])] = atmo_values[-i]

    return new_contour


def elliptical_mask(center, width, height, x_radius, y_radius, p=0):
    '''
    Creates elliptical mask of certain radii in an image of certain width and height, roated at angle p
    '''
    Y, X = np.ogrid[:height, :width]
    X = X - center[0]
    Y = Y - center[1]
    distance = np.sqrt(((X*np.cos(p) + Y*np.sin(p))/x_radius)**2 + ((X*np.sin(p) - Y*np.cos(p))/y_radius)**2)
    mask = distance <= 1
    
    return mask