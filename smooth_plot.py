import matplotlib.pyplot as plt
import numpy as np
from smoothening import smoothen

# datasets = ["Qatar-6 b.txt", "TrES-2 b.txt", "TrES-5 b.txt", "WASP-2 b.txt", "WASP-52 b.txt"]

dataset = "Qatar-6 b.txt" # can input any of the 5 datasets listed above

data  = np.genfromtxt(dataset, delimiter=",", names=True, skip_header=23)

dates = data['DATE']
dates_mins = (dates - dates[0]) * 24 * 60
flux  = data['DIFF']
errs  = data['ERR']

bins, smooth_fluxes, smooth_errs = smoothen(dates, flux, errs, n = 25) # can change n = number of datapoints in the smoothened dataset

plt.figure(figsize = [10, 6])

plt.scatter(dates_mins, flux, zorder = 0, c = 'lightgray', label = 'Raw data', alpha = 0.75)
plt.errorbar(dates_mins, flux, yerr = errs, ls = '', alpha = 0.5, ecolor = 'lightgray', zorder = -1)
plt.scatter(bins, smooth_fluxes, c = 'k', zorder = 2, label = 'Smoothened data')
plt.errorbar(bins, smooth_fluxes, yerr = smooth_errs, ls = '', alpha = 0.5, ecolor = 'k', zorder = 1)
plt.axhline(1, ls = '--', c = 'k', zorder= -2, alpha = 0.3)

plt.xlabel(f"Times since JD{data['DATE'][0]} [mins]")
plt.ylabel("Normalized flux")
plt.title(f"{dataset[:-6]}")

plt.legend()
plt.show()
    
