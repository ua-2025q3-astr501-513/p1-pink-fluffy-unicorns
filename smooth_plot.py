import matplotlib.pyplot as plt
import numpy as np
from smoothening import smoothen

# datasets = ["Qatar-6 b.txt", "TrES-2 b.txt", "TrES-5 b.txt", "WASP-2 b.txt", "WASP-52 b.txt"]

dataset = "Qatar-6 b.txt" # can input any of the 5 datasets listed above

data  = np.genfromtxt("assets/exoplanet_data/"+dataset, delimiter=",", names=True, skip_header=23)

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
plt.title(f"{dataset[:-4]}")

plt.legend()
plt.show()

unicorn = np.load("unicorn_boolean_mask.npz")
mask = ~unicorn["arr_0"]
mask[mask > 0] = 1


exoplanet = add_atmosphere(mask, func)
plt.imshow(exoplanet)
plt.colorbar()
plt.show()

limb_darkened_star = limbDarken(mask, 1)

plt.imshow(limb_darkened_star)
plt.show()

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
    
<<<<<<< HEAD
=======
plt.show()
plt.scatter(range(nobs),photom, marker = 'o')
plt.ylim(min(photom)-0.01*min(photom),max(photom)+0.01*max(photom))
plt.show()

>>>>>>> f4cb92cf2726582ce543167b1c501012bfd501d9
