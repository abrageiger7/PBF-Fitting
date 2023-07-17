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


powerlaw_inter = 110.0 # this is about the fwhm of the sband average
freq_inter = 1742.0 # this is the starting freq for the powerlaw
powerlaws = np.array([0.0]) # these are the powerlaws to test

#import data
with open('j1903_data.pkl', 'rb') as fp:
    data_dict = pickle.load(fp)

mjd_strings = list(data_dict.keys())
mjds = np.zeros(np.size(mjd_strings))
for i in range(np.size(mjd_strings)):
    mjds[i] = data_dict[mjd_strings[i]]['mjd']

for pwrlaw_ind in range(np.size(powerlaws)):

    data_collect = {}

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

            p.init_freq_subint(ii)

            freq_list[ii] = p.freq_suba

            iwidth = powerlaw_inter * np.power((p.freq_suba/freq_inter),powerlaws[pwrlaw_ind])

            iwidth_ind = find_nearest(gauss_fwhm, iwidth)[1][0][0]

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

    with open(f'best_beta_intrins_sband_allmjd_allfreq_iwidth_{int(np.round(powerlaw_inter))}micros_pwrlaw{powerlaws[pwrlaw_ind]}.pkl', 'wb') as fp:
        pickle.dump(data_collect, fp)


#===============================================================================
# Now plotting the results - best fit beta mean and standard error for each of
# 12 frequencies
# ==============================================================================

    mjd_list = data_collect.keys()

    freqs = np.array([1180,1230,1281,1330,1380,1431,1480,1531,1581,1639,1693,1742])

    beta_based_on_freq_pwrlawiwidth = {}

    for i in range(np.size(freqs)):

        beta_collect = []

        for ii in mjd_list:

            here = np.where(((data_collect[ii]['frequencies'] > (freqs[i]-25)) & (data_collect[ii]['frequencies'] < (freqs[i]+25))))

            if np.size(here) > 0:

                beta_collect.append(data_collect[ii]['beta_fit'][here][0])

        beta_based_on_freq_pwrlawiwidth[f'{freqs[i]}'] = beta_collect

    beta_avgs_pwrlawiwidth = np.zeros(np.size(freqs))
    beta_stds_pwrlawiwidth = np.zeros(np.size(freqs))
    num_prof_per_freq_pwrlawiwidth = np.zeros(np.size(freqs))

    beta_based_on_freq_pwrlawiwidth = dict(sorted(beta_based_on_freq_pwrlawiwidth.items()))

    ind = 0

    for i in beta_based_on_freq_pwrlawiwidth.keys():

        print(f'Frequency = {i}')

        print(f'    Beta average = {np.average(beta_based_on_freq_pwrlawiwidth[i])}')

        num_prof_per_freq_pwrlawiwidth[ind] = len(beta_based_on_freq_pwrlawiwidth[i])

        beta_avgs_pwrlawiwidth[ind] = np.average(beta_based_on_freq_pwrlawiwidth[i])

        print(f'    Beta std = {np.std(beta_based_on_freq_pwrlawiwidth[i])}')

        beta_stds_pwrlawiwidth[ind] = np.std(beta_based_on_freq_pwrlawiwidth[i])

        ind += 1

    plt.figure(1)
    plt.rc('font', family = 'serif')

    fig, ax = plt.subplots()

    markers, caps, bars = ax.errorbar(x = freqs, y = beta_avgs_pwrlawiwidth, yerr = beta_stds_pwrlawiwidth/np.sqrt(num_prof_per_freq_pwrlawiwidth), fmt = '.', capsize = 2, color = 'k')

    [bar.set_alpha(0.4) for bar in bars]
    [cap.set_alpha(0.4) for cap in caps]

    #ax.set_yticks(np.linspace(3.3,3.6,6))
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Beta')
    ax.set_title(f'Best Fit Beta, FWHM({freq_inter}) = {int(powerlaw_inter)}, X_FWHM = {powerlaws[pwrlaw_ind]}')
    plt.savefig(f'best_beta_allmjd_versus_freq_ifwhm_{int(np.round(powerlaw_inter))}micros_pwrlaw{powerlaws[pwrlaw_ind]}.pdf')
    #plt.show()
    plt.close('all')

    del(beta_based_on_freq_pwrlawiwidth)
    del(beta_avgs_pwrlawiwidth)
    del(beta_stds_pwrlawiwidth)
    del(num_prof_per_freq_pwrlawiwidth)
    del(data_collect)
