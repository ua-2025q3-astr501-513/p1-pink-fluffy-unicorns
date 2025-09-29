import numpy as np
import matplotlib.pyplot as plt 
from astropy import constants as c

# Get a simulated observation -- photometric points are just a sum of the returned array:
def photometry(Rs, Rp, P, t, b, F=2.4, FoV=None, n=None, shape='sphere'):
    # shape: must be either sphere, or an image of ones and zeros
    if shape != 'sphere':
        # load data as ones and zeros
        data = load('unicorn_boolean_mask.npz')
        shape_val = np.zeros(np.shape(data['arr_0']))
        shape_zero = (np.where(data['arr_0'] == 0))
        shape_val[shape_zero]=1

        new_planet = np.pad(shape_val,((int(np.ceil(((n-np.shape(shape_val)[0])/2)+b)),int(np.floor(((n-np.shape(shape_val)[0])/2)-b))),
                                      (int(np.ceil(((n-np.shape(shape_val)[1])/2)+a)),int(np.floor(((n-np.shape(shape_val)[1])/2)-a)))), 'edge')

    Rs *= 1 # solar radii
    Rp *= c.R_jup/c.R_sun
    
    if not FoV:
        FoV = int(Rs*3) # field of view
    if not n:
        n = FoV*100 # resolution
            
    A = ((P/365.25)**(2/3))*(214.9**3)  # P = [year]; A = [AU]*[215RS/AU]
    w = (2*A)/(P/2) # velocity [RS/day]
    a = t*w*FoV/Rs # [RS]; where t is [-tobs, tobs], and normalized to pixel scale
    
    # Create disks
    x = np.linspace(-FoV/2 + FoV/(2*n), FoV/2 - FoV/(2*n), n) # make sure our array is centered properly
    X, Y = np.meshgrid(x,x)
    star_disk = X**2 + Y**2 # large disk equation
    planet_disk = (X-a)**2 + (Y-b)**2 # small disk equation
    
    # trim to radii
    star_disk = star_disk<Rs**2
    if shape!='sphere':
        planet_disk = new_planet
    else:
        planet_disk = planet_disk<Rp**2
    
    # if the planet leaves the star image, make sure the value is 0 instead of 1
    star_mask = np.where(planet_disk>star_disk)
    planet_disk[star_mask] = 0

    # spread over flux
    dF = 1/(np.sum(np.sum(star_disk,axis=0))) # share of flux per lit pixels

    # convert back from boolean to int
    obs = np.multiply(star_disk, 1)*dF*F-np.multiply(planet_disk, 1)*dF*F
    
    return obs

# Calcualte recommended "observing session length" based on planet period
def rec_tt(P, Rs):
    # Very very simplified method: just finding a fraction of the circumference 
    circ = 2*np.pi*((P/365.25)**(2/3))*(215**3) # P = [year]; a = [AU]*[215RS/AU]
    rtt = P*(Rs/circ) # [days]
    return rtt

# Example:
Rs = 1 # stellar radius [solar radii]
Rp = 1 # planet radius [jupiter radii]
P = 10 # period [days]
b = 0 # y-axis displacement [n/Rs]
F = 5 # stellar flux
nobs = 25 # number of observation 
tobs = 2*rec_tt(P,Rs) # observing session length (tobs/nobs = "exposure times") [days]

tt = np.linspace(-1*tobs/2, tobs/2, nobs)
photom = []
        
fig, axs = plt.subplots(nobs,1,figsize=(1,nobs))
for ii in range(nobs):
    data = photometry(Rs=Rs, Rp=Rp, P=P, t=tt[ii], b=b, F=F, shape='sphere')
    axs[ii].imshow(data)
    photom.append(np.sum(data))
    
plt.show()
plt.scatter(range(nobs),photom, marker = 'o')
plt.ylim(min(photom)-0.01*min(photom),max(photom)+0.01*max(photom))
plt.show()
