import numpy as np

def weighted_stats(fluxes, errors):
    """
    Takes in a set of normalized fluxes and their errors and calculates the weighted mean 
    and standard deviation for the set.
    """
    inverse_var   = 1/(errors)**2
    weighted_mean = np.sum(fluxes * inverse_var)/np.sum(inverse_var)
    weighted_std  = np.sqrt(1/np.sum(inverse_var))

    return weighted_mean, weighted_std

def smoothen(dates, fluxes, errors, n=15):
    """
    Smoothens a time-series light curve by binning data into equally spaced time intervals
    and computing the weighted mean flux and its associated error for each bin.

    Parameters
    ----------
    dates : array-like
        Array of time values in days (e.g., Julian dates). Must be sorted in ascending order.
    fluxes : array-like
        Array of normalized flux measurements corresponding to `dates`.
    errors : array-like
        Array of 1-sigma uncertainties associated with each flux measurement.
    n : int, optional
        Number of equally spaced time bins to divide the total observation duration into.
        Default is 25.

    Returns
    -------
    bin_centers : ndarray
        Array of time bin centers (in minutes) used for smoothing, length `n-1`.
    smooth_fluxes : ndarray
        Weighted mean flux in each time bin.
    smooth_errs : ndarray
        Uncertainty on the weighted mean flux in each bin, derived from inverse variance weighting.

    Notes
    -----
    - Dates are internally shifted so that the first observation corresponds to time = 0 minutes.
    - The total observing span is rounded to the nearest 5 minutes before binning.
    - Weighted statistics are used:
        * Mean:   sum(flux_i / err_i^2) / sum(1 / err_i^2)
        * Error:  sqrt(1 / sum(1 / err_i^2))
    - Points with smaller error bars contribute more strongly to the weighted mean.
    """

    n+= 1
    
    # calculate the total time of data collection in units of minutes (rounded to closest 5)
    total_time   = (dates[-1] - dates[0]) * 24 * 60
    rounded_time = np.round(total_time/5) * 5
    # ensure that date array starts at 0 and convert units from days to minutes
    dates_mins   = (dates - dates[0]) * 24 * 60
    # create equally spaced time array, using n to determine how many elements are needed
    timeslots    = np.linspace(0, rounded_time, n)
    bin_centers  = 0.5 * (timeslots[:-1] + timeslots[1:])
    

    
    smooth_fluxes = []
    smooth_errs   = []
    
    # bin all data points within a time slot
    for start, end in zip(timeslots[:-1], timeslots[1:]):
        indices = (dates_mins > start) & (dates_mins <= end)
        if np.sum(indices) == 0:
            smooth_fluxes.append(np.nan)
            smooth_errs.append(np.nan)
        else:
            mean, std = weighted_stats(fluxes[indices], errors[indices])
            smooth_fluxes.append(mean)
            smooth_errs.append(std)
        
    return bin_centers, np.array(smooth_fluxes), np.array(smooth_errs)
    
