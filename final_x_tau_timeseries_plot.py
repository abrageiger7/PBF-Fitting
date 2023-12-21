import numpy as np
import pickle
from fit_functions import *
import matplotlib.ticker as tick
from profile_class import Profile_Fitting
from astropy.time import Time
from scipy.stats import pearsonr
import os

plt.style.use('dark_background')

title1 = f'powerlaw_data|MODELED|BETA=3.667|MEDIUM=thick|TIME_AVG_EVERY8.0.npz'

plaw_data1 = np.load(title1)['plaw_data']

title2 = f'powerlaw_data|MODELED|BETA=3.667|MEDIUM=thin|TIME_AVG_EVERY8.0.npz'

plaw_data2 = np.load(title2)['plaw_data']

title3 = f'powerlaw_data|MODELED|EXP|MEDIUM=exp|TIME_AVG_EVERY8.0.npz'

plaw_data3 = np.load(title3)['plaw_data']

#load in the data
with open('j1903_data.pkl', 'rb') as fp:
    data_dict = pickle.load(fp)

mjd_strings = list(data_dict.keys())
mjds = np.zeros(np.size(mjd_strings))
for i in range(np.size(mjd_strings)):
    mjds[i] = data_dict[mjd_strings[i]]['mjd']

plt.figure(1)

fig, axs = plt.subplots(nrows=2, ncols=1, figsize = (9,5), sharex = True)
plt.rc('font', family = 'serif')
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')

label1 = r'Extended Medium $\beta$ = 3.667'
label2 = r'Thin Screen $\beta$ = 3.667'
label3 = 'Decaying Exponential'

markers, caps, bars = axs.flat[0].errorbar(x = mjds, y = plaw_data1[:,0], yerr = [plaw_data1[:,1], plaw_data1[:,2]], fmt = 's', ms = 5, color = 'grey', capsize = 2, label = label1)

print(r'Thick $X_{\tau}$ Average = ', np.average(plaw_data1[:,0]), np.std(plaw_data1[:,0]))

[bar.set_alpha(0.7) for bar in bars]
[cap.set_alpha(0.7) for cap in caps]

markers, caps, bars = axs.flat[0].errorbar(x = mjds, y = plaw_data2[:,0], yerr = [plaw_data2[:,1], plaw_data2[:,2]], fmt = 'o', ms = 5, color = 'g', capsize = 2, label = label2)

print(r'Thin $X_{\tau}$ Average = ', np.average(plaw_data2[:,0]), np.std(plaw_data2[:,0]))

[bar.set_alpha(0.7) for bar in bars]
[cap.set_alpha(0.7) for cap in caps]

markers, caps, bars = axs.flat[0].errorbar(x = mjds, y = plaw_data3[:,0], yerr = [plaw_data3[:,1], plaw_data3[:,2]], fmt = 'D', ms = 5, color = 'steelblue', capsize = 2, label = label3)

print(r'Exponential $X_{\tau}$ Average = ', np.average(plaw_data3[:,0]), np.std(plaw_data3[:,0]))

[bar.set_alpha(0.7) for bar in bars]
[cap.set_alpha(0.7) for cap in caps]

axs.flat[0].set_ylabel(r'$X_{\tau}$', fontsize = 14)

axs.flat[0].legend(loc = 'lower right', fontsize = 7.5)
axis2 = axs.flat[0].twiny()
XLIM = axs.flat[0].get_xlim()
XLIM = list(map(lambda x: Time(x,format='mjd',scale='utc').decimalyear,XLIM))
axis2.set_xlim(XLIM)
axis2.set_xlabel('Years', fontsize = 14)
axis2.tick_params(axis='x', labelsize = 'medium')

axs.flat[1].set_ylabel(r'$\tau_0$ ($\mu$s)', fontsize = 14)
axs.flat[1].set_xlabel('MJD', fontsize = 14)

markers, caps, bars = axs.flat[1].errorbar(x = mjds, y = plaw_data1[:,3], yerr = [plaw_data1[:,4], plaw_data1[:,5]], fmt = 's', ms = 5, color = 'grey', capsize = 2, label = label1)

print(r'Thick $\tau_0$ Average = ', np.average(plaw_data1[:,3]), np.std(plaw_data1[:,3]))

[bar.set_alpha(0.7) for bar in bars]
[cap.set_alpha(0.7) for cap in caps]

markers, caps, bars = axs.flat[1].errorbar(x = mjds, y = plaw_data2[:,3], yerr = [plaw_data2[:,4], plaw_data2[:,5]], fmt = 'o', ms = 5, color = 'g', capsize = 2, label = label2)

print(r'Thin $\tau_0$ Average = ', np.average(plaw_data2[:,3]), np.std(plaw_data2[:,3]))

[bar.set_alpha(0.7) for bar in bars]
[cap.set_alpha(0.7) for cap in caps]

markers, caps, bars = axs.flat[1].errorbar(x = mjds, y = plaw_data3[:,3], yerr = [plaw_data3[:,4], plaw_data3[:,5]], fmt = 'D', ms = 5, color = 'steelblue', capsize = 2, label = label3)

print(r'Exponential $\tau_0$ Average = ', np.average(plaw_data3[:,3]), np.std(plaw_data3[:,3]))

[bar.set_alpha(0.7) for bar in bars]
[cap.set_alpha(0.7) for cap in caps]

fig.tight_layout()

plt.savefig('time_scales_thin_thick_exp.pdf', bbox_inches='tight')
plt.savefig('time_scales_thin_thick_exp.png', dpi = 300, bbox_inches='tight')
plt.close('all')
