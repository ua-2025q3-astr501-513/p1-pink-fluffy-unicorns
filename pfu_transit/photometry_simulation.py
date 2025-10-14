from astropy import constants as c
from multiprocessing.pool import ThreadPool as Pool
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

class CalculateFlux(object):

    """
    Class for calculating observed flux in transit, and for saving transit images. 
    Intended for multiprocessing
    """

    def __init__(self, params):
        self.params = params
    
    def __call__(self, i):

        new_planet, star, image_directory = self.params

        # Create observable by multiplying masks
        image = (1 - np.concatenate((new_planet[:,i:].T, new_planet[:,:i].T)).T)*star

        # Optionally save image as a jpg
        if image_directory is not None:
            norm = Image.fromarray(255*image/np.max(image)).convert('RGB')
            norm.save(os.path.join(image_directory, "sim_"+str(i)+".jpg"), format="JPEG") 
            
        return np.sum(image), i

def photometric_observation(star, planet, velocity=1, image_scale=3/2500, step=1, image_directory=None, threadcount = 25):
    """
    Get times, fluxes, and optionally save images for simulated exoplanet transit.

    Parameters
    ----------
    star - Star mask
    planet - Planet mask
    velocity - Projected velocity of the planet in solar radii / hour
    image_scale - Image scale in solar radii / px
    step - Simulation step size in px
    image_directory - Directory to save images if wanted
    threadcount - Number of threads to use for multiprocessing
    """

    # Find the rightmost edge of the planet so the planet can be moved to the edge of the image
    planet_indices = np.where(planet >0)
    
    begin_offset = np.max(planet_indices[1])
    planet_width = begin_offset - np.min(planet_indices[1])
    
    array_shape = np.shape(planet)
    new_planet = planet[:,:begin_offset]

    # New image of the planet located to the edge
    new_planet = np.pad(new_planet, ((0, 0), (array_shape[1] - begin_offset, 0)), mode="constant")

    # Create steps for the simulator
    iterator = np.arange(0, array_shape[1] - planet_width, step=step)

    if threadcount > 1:
        with Pool(threadcount) as pool:
            flux, _ = zip(*tqdm(pool.imap(CalculateFlux((new_planet, star, image_directory)), iterator), total=len(iterator)))
    else:
        flux_calculator = CalculateFlux((new_planet, star, image_directory))
        flux = np.zeros(len(iterator))
        for i in range(len(iterator)):
            flux[i], _ = flux_calculator(iterator[i])

    return np.asarray(iterator)*image_scale/velocity, np.asarray(flux)
                                

# Get a simulated observation -- photometric points are just a sum of the returned array:
def photometry(Rs, Rp, P, t, b, F=2.4, FoV=None, n=None, shape='sphere'):
    """
    Create an array for every frame for our transit.

    Parameters
    ----------
    Rs - Stellar radius [m]
    Rp - Planet radius [m]
    P - Planet period [d]
    t - array of observing times [d]
    b - lateral displacement from center [n/Rs]
    F - flux [solar flux]
    FoV - frame size [Rs]
    n - resolution
    shape - 'sphere', 'square', or custom mask
    """
    
    if not FoV:
        FoV = int(Rs*3) # field of view
        
    if not n:
        n = FoV*100 # resolution
            
    A = ((P/365.25)**(2/3))*215  # P = [year]; A = [AU]*[215RS/AU]
    w = (2*A)/(P/2) # velocity [RS/day]
    a = t*w*FoV/Rs # [RS]; where t is [-tobs, tobs], and normalized to pixel scale
    
    # Create disks
    x = np.linspace(-FoV/2 + FoV/(2*n), FoV/2 - FoV/(2*n), n) # make sure our array is centered properly
    X, Y = np.meshgrid(x,x)
    star_disk = X**2 + Y**2 # large disk equation
    
    # trim to radii
    star_disk = star_disk<Rs**2      
    if shape=='sphere':
        planet_disk = (X-a)**2 + (Y-b)**2 # small disk equation
        planet_disk = planet_disk<Rp**2
    elif shape=='square':
        planet_disk = np.abs(X-a) + np.abs(Y-b) # small square equation
        planet_disk = planet_disk<Rp
    else:
        # load boolean mask into an array of ones and zeros
        data = np.load('unicorn_boolean_mask.npz')
        shape_val = np.zeros(np.shape(data['arr_0']))
        shape_zero = (np.where(data['arr_0'] == 0))
        shape_val[shape_zero]=1

        new_planet = np.pad(shape_val,((int(np.ceil(((n-np.shape(shape_val)[0])/2)+b)),int(np.floor(((n-np.shape(shape_val)[0])/2)-b))),
                                      (int(np.ceil(((n-np.shape(shape_val)[1])/2)+a)),int(np.floor(((n-np.shape(shape_val)[1])/2)-a)))), 'edge')

        planet_disk = new_planet
    
    # if the planet leaves the star image, make sure the value is 0 instead of 1
    star_mask = np.where(planet_disk>star_disk)
    planet_disk[star_mask] = 0

    # spread over flux
    dF = 1/(np.sum(np.sum(star_disk,axis=0))) # share of flux per lit pixels

    # convert back from boolean to int
    obs = np.multiply(star_disk, 1)*dF*F-np.multiply(planet_disk, 1)*dF*F
    
    return obs

def rec_tt(P, Rs):
    """
    Calcualte recommended observing times based on planet period and stellar radius.

    Parameters
    ----------
    P - Planet period [d]
    Rs - Stellar radius [m]
    """
    
    # Very very simplified method: just finding a fraction of the circumference and compare to planet period
    circ = 2*np.pi*((P/365.25)**(2/3))*215 # P = [year]; a = [AU]*[215RS/AU]
    rtt = P*(Rs/circ) # [days]
    return rtt
