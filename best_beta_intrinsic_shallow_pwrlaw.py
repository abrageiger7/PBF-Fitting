"""
Created July 2023
@author: Abra Geiger abrageiger7

Calculations of Best Fit Beta Over all MJD and Frequency

Only 100 pbf to choose from when ran
- set fitting_params.py in this way

Set intrinsic width to powerlaw starting at 101 microseconds
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

powerlaw_inter = 101.0 # this is about the fwhm of the sband average
freq_inter = 1742.0 # this is the starting freq for the powerlaw
powerlaws = np.array([0.1,0.3,0.5]) # these are the tested powerlaws

#import data
with open('j1903_data.pkl', 'rb') as fp:
    data_dict = pickle.load(fp)

mjd_strings = list(data_dict.keys())
mjds = np.zeros(np.size(mjd_strings))
for i in range(np.size(mjd_strings)):
    mjds[i] = data_dict[mjd_strings[i]]['mjd']

data_collect = {}

for pwrlaw_ind in range(np.size(powerlaws)):

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

            freq_list[ii] = p.freq_suba

            iwidth = powerlaw_inter * np.power((p.freq_suba/freq_inter),powerlaws[pwrlaw_ind])

            iwidth_ind = find_nearest(gauss_fwhm, iwidth))[1][0][0]

            datab = p.fit(ii, 'beta', iwidth_ind = iwidth_ind)
            tau_listb[ii] = datab['tau_fin']
            beta_listb[ii] = datab['beta']
            beta_low_listb[ii] = datab['beta_low']
            beta_up_listb[ii] = datab['beta_up']
            iwidth_listb[ii] = datab['intrins_width_set']

        data_collect[f'{int(np.round(mjd))}'] = {}
        data_collect[f'{int(np.round(mjd))}']['beta_fit'] = beta_listb
        data_collect[f'{int(np.round(mjd))}']['beta_low'] = beta_low_listb
        data_collect[f'{int(np.round(mjd))}']['beta_up'] = beta_up_listb
        data_collect[f'{int(np.round(mjd))}']['tau'] = tau_listb
        data_collect[f'{int(np.round(mjd))}']['mjd'] = mjd
        data_collect[f'{int(np.round(mjd))}']['frequencies'] = freq_list
        data_collect[f'{int(np.round(mjd))}']['intrinsic_width_set'] = iwidth_listb

    with open(f'best_beta_intrins_sband_allmjd_allfreq_iwidth_{int(np.round(gauss_fwhm[iwidth_ind]))}micros_pwrlaw{powerlaws[pwrlaw_ind]}.pkl', 'wb') as fp:
        pickle.dump(data_collect, fp)
