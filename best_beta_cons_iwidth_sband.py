"""
Created July 2023
@author: Abra Geiger abrageiger7

Calculations of Best Fit Beta Over all MJD and Frequency

Only 100 pbf to choose from when ran
- set fitting_params.py in this way

Set gwidth to 91 and 107 microseconds index 12 and 14 when 50 gwidths
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
import matplotlib.ticker as tick
from astropy.time import Time
import pickle


from profile_class_sband_intrinsic import Profile_Intrinss as pcs
from fit_functions import *

#import data
with open('j1903_data.pkl', 'rb') as fp:
    data_dict = pickle.load(fp)

mjd_strings = list(data_dict.keys())
mjds = np.zeros(np.size(mjd_strings))
for i in range(np.size(mjd_strings)):
    mjds[i] = data_dict[mjd_strings[i]]['mjd']

data_collect = {}

iwidth = 107.0

iwidth_ind = find_nearest((gauss_fwhm, iwidth))[1][0][0]

for i in range(56):

    mjd = data_dict[mjd_strings[i]]['mjd']
    data = data_dict[mjd_strings[i]]['data']
    freqs = data_dict[mjd_strings[i]]['freqs']
    dur = data_dict[mjd_strings[i]]['dur']

    p = pcs(mjd, data, freqs, dur)

    freq_list = np.zeros(p.num_sub)

    tau_listb = np.zeros(p.num_sub)

    beta_listb = np.zeros(p.num_sub)
    beta_low_listb = np.zeros(p.num_sub)
    beta_up_listb = np.zeros(p.num_sub)
    iwidth_listb = np.zeros(p.num_sub)

    for ii in range(p.num_sub):

        print(f'Frequency {ii}')

        datab = p.fit(ii, 'beta', iwidth_ind = iwidth_ind)
        tau_listb[ii] = datab['tau_fin']
        beta_listb[ii] = datab['beta']
        beta_low_listb[ii] = datab['beta_low']
        beta_up_listb[ii] = datab['beta_up']
        iwidth_listb[ii] = datab['intrins_width_set']

        freq_list[ii] = p.freq_suba

    data_collect[f'{int(np.round(mjd))}'] = {}
    data_collect[f'{int(np.round(mjd))}']['beta_fit'] = beta_listb
    data_collect[f'{int(np.round(mjd))}']['beta_low'] = beta_low_listb
    data_collect[f'{int(np.round(mjd))}']['beta_up'] = beta_up_listb
    data_collect[f'{int(np.round(mjd))}']['tau'] = tau_listb
    data_collect[f'{int(np.round(mjd))}']['mjd'] = mjd
    data_collect[f'{int(np.round(mjd))}']['frequencies'] = freq_list
    data_collect[f'{int(np.round(mjd))}']['intrinsic_width_set'] = iwidth_listb

mjd_list = data_collect.keys()

freqs = np.array([1180,1230,1281,1330,1380,1431,1480,1531,1581,1639,1693,1742])

beta_based_on_freq_setiwidth = {}

for i in range(np.size(freqs)):

    beta_collect = []

    for ii in mjd_list:

        here = np.where(((data_collect[ii]['frequencies'] > (freqs[i]-25)) & (data_collect[ii]['frequencies'] < (freqs[i]+25))))

        if np.size(here) > 0:

            beta_collect.append(data_collect[ii]['beta_fit'][here][0])

    beta_based_on_freq_setiwidth[f'{freqs[i]}'] = beta_collect

beta_avgs_setiwidth = np.zeros(np.size(freqs))
beta_stds_setiwidth = np.zeros(np.size(freqs))
num_prof_per_freq_setiwidth = np.zeros(np.size(freqs))


ind = 0

for i in beta_based_on_freq_setiwidth.keys():

    print(f'Frequency = {i}')

    print(f'    Beta average = {np.average(beta_based_on_freq_setiwidth[i])}')

    num_prof_per_freq_setiwidth[ind] = len(beta_based_on_freq_setiwidth[i])

    beta_avgs_setiwidth[ind] = np.average(beta_based_on_freq_setiwidth[i])

    print(f'    Beta std = {np.std(beta_based_on_freq_setiwidth[i])}')

    beta_stds_setiwidth[ind] = np.std(beta_based_on_freq_setiwidth[i])

    ind += 1

plt.rc('font', family = 'serif')

markers, caps, bars = plt.errorbar(x = freqs, y = beta_avgs_setiwidth, yerr = beta_stds_setiwidth/np.sqrt(num_prof_per_freq_setiwidth), fmt = '.', capsize = 2, color = 'k')

[bar.set_alpha(0.4) for bar in bars]
[cap.set_alpha(0.4) for cap in caps]


plt.figure(1)
plt.xlabel('Frequency [MHz]')
plt.ylabel('Beta')
plt.title(rf'Best Fit Beta, Intrinsic FWHM of {int(np.round(gauss_fwhm[iwidth_ind]))} $\mu$s')
plt.savefig(f'best_beta_allmjd_versus_freq_ifwhm{int(np.round(gauss_fwhm[iwidth_ind]))}micros.pdf')
plt.show()
plt.close('all')

with open(f'best_beta_intrins_sband_allmjd_allfreq_iwidth_{int(np.round(gauss_fwhm[iwidth_ind]))}micros.pkl', 'wb') as fp:
    pickle.dump(data_collect, fp)
