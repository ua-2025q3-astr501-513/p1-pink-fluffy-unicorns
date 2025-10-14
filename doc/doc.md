**assets/**
- hosts example data
- Mass density profile of Earth and Jupiter atmosphere
- Opacity of Earth and Jupiter atmosphere


**pfu_transit/**
- hosts all of our functions


**pfu_transit/mask_generation.py**

Create star and planet masks.
- *class*: Eddington(object)
    - Class for parallelizing calculation of limb darkening coefficients, an integration of the Eddington approximation
- *function*: **generate_exoplanet**(framew - Frame width in pixels, frameh - Frame height in pixels, image_scale - Image scale in solar radii / px, planet_radius - Planet radius in solar radii, planet_oblateness - Planet oblateness, planet_rot_obliquity - Obliquity of the rotational axis of the planet, assuming ellipticity by rotation, wavelength - Wavelength observed in microns, b - Impact parameter: offset of planet due to inclination from the center of the image, in projected solar radii, max_height - Maximum height of the atmosphere relative to Earth's normalized maximum height, planet_type)
- Planet type, rocky or gaseous, surface - Changes definition of planet radius, whether to include or exclude atmosphere (for bodies without a surface), no_atmosphere - Removes atmosphere, overrides other parameters)
    - Generate an exoplanet mask with an Earth-like or Jupiter-like atmospheric profile.
- *function*: **generate_star**(framew - Frame width in pixels, frameh - Frame height in pixels, image_scale - Image scale in solar radii / px, stellar_radius - Stellar radius in solar radii, stellar_oblateness - Stellar oblateness, stellar_rot_obliquity - Obliquity of the rotational axis of the star, assuming ellipticity by rotation, stellar_temperature - Temperature of the star in Kelvin, wavelength - Wavelength observed in microns)
    - Generate a star mask assuming a blackbody and Eddington approximated limb darkening.
- *function*: **limb_darken**(mask - Star mask, stellar_radius - Stellar radius in solar radii, T - Temperature of the star in Kelvin, l - wavelength in microns)
    - Apply stellar properties and Eddington approximated limb darkening.
- *function*: **eddington**(r - distance from surface, freq - dimensionless frequency, Nint - Number of points for numerical integration, wavelength_limit - Bound of integration in dimensionless units, threadcount - Number of threads for multiprocessing)
    - Calculate Eddington approximated limb darkening.
- *function*: **add_atmosphere**(Mask - Image, populated with 1s and 0s, atmo_func - Exoplanet atmosphere profile, max_height - Maximum height of atmosphere (pixels))
    - Add an exoplanet atmosphere with a certain profile to an existing exoplanet mask.
- *function*: **elliptical_mask**(center, width, height, x_radius, y_radius, p - angle of rotation)
    - Creates elliptical mask of certain radii in an image of certain width and height, rotated at angle p.


**pfu_transit/planet_atmosphere_extinction.py**

Model exoplanet atmosphere extinction.
- *function*: **exponential_atmosphere**(h - height, R - max height of the atmosphere)
    - Initialize an exoplanet atmosphere with an exponential absorption profile
- *class*: **Integrate**(self, params)
    - Class for integrating projected density profile for multiprocessing
- *class*: **ExoplanetAtmosphere**()
    - Initialize an exoplanet atmosphere model with an Earth-like/Jupiter-like density profile and wavelength-dependent opacities.
    - *function*: **density_height**(self, h_cm - Atmosphere heigh [cm], r_planet_cm - Planet radius [cm], planet - Stored atmosphere (ex: 'earth'))
        - Atmospheric density [g/cm³] at height h_cm [cm].
    - *function*: **density_proj_distance**(self, d_cm - Projected distance [cm], r_cm - Overall radius [cm], r_planet_cm - Planet radius [cm])
        - Atmospheric density [g/cm³] at projected distance d_cm [cm].
    - *function*: **get_opacity**(self, lamb_um - Wavelength [µm])
        - Return opacity [cm²/g] at wavelength lamb_um [µm].
    - *function*: **extinctino_sphere**(self, h - Altitude [cm] above surface, r_planet - Planet radius [cm], lamb_um - Wavelength [µm], threadcount=20)
        - Calculate extinction factor exp(-tau) through a spherical atmosphere.
    

**pfu_transit/photometry_simulation.py**

Simulate transit and return photometry points.
- *class*: **CalculateFlux**()
    - Class for calculating observed flux in transit, and for saving transit images. Intended for multiprocessing.
- *function*: **photometric_observation**(star - Star mask, planet - Planet mask, velocity - Projected velocity of the planet in solar radii / hour, image_scale - Image scale in solar radii / px, step - Simulation step size in px, image_directory - Directory to save images if wanted, threadcount - Number of threads to use for multiprocessing)
    - Get times, fluxes, and optionally save images for simulated exoplanet transit.
- *function*: **photometry**(Rs - Stellar radius [m], Rp - Planet radius [m], P - Planet period [d], t - array of observing times [d], b - lateral displacement from center [n/Rs], F - flux [solar flux], FoV - frame size [Rs], n - resolution, shape - 'sphere', 'square', or custom mask)
    - Create an array for every frame for our transit.
- *function*: **rec_tt**(P - Planet period [d], Rs - Stellar radius [m])
    - Calculate recommended observing times based on planet period and stellar radius.



**pfu_transit/smoothening.py**

Smooth out real photometric datapoints by binning.
- *function*: **weighted_stats**(fluxes, errors)
    - Takes in a set of normalized fluxes and their errors and calculates the weighted mean and standard deviation for the set.
- *function*: **smoothen**(dates - Array of time values in days (e.g., Julian dates)(must be sorted in ascending order), fluxes - Array of normalized flux measurements corresponding to *dates*, errors - Array of 1-sigma uncertainties associated with each flux measurement, n - Number of equally spaced time bins to divide the total observation duration into (default is 25))
    - Smoothens a time-series light curve by binning data into equally spaced time intervals and computing the weighted mean flux and its associated error for each bin.


**pfu_transit/cubic_interp.py**

Create our own cubic spline interpolator. 
- *function*: **_natural_cubic_spline_build**(x - 1D array of strictly increasing values (the independent variable), y - 1D array of dependent variable values corresponding to *x* (must be the same length as *x*))
    - Computes cubic spline coefficients for 1D data using natural boundary conditions.
- *function*: **cubic_spline_interpolator**(x - 1D array of strictly increasing values (the independent variable), y - 1D array of dependent variable values corresponding to *x* (must be the same length as *x*))
    - Constructs a 1D natural cubic spline interpolator for smooth curve fitting.
- *function*: **read_raw_exotic**(path - path to .txt file)
    - Read DATE, DIFF, ERR columns from EXOTIC-style .txt (comma-delimited).
