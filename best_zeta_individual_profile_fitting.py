import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
import matplotlib.ticker as tick
from astropy.time import Time
import pickle
import sys

from profile_class import Profile_Fitting
from fit_functions import *

'''Adapt for one profile'''

pbf_type = 'zeta'
intrinsic_shape = 'modeled'

if __name__ == '__main__':

    ua_pbfs = {}
    tau_values = {}

    ua_pbfs['beta'] = np.load(f'beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']
    ua_pbfs['zeta'] = np.load(f'zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']
    ua_pbfs['exp'] = np.load(f'exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']

    tau_values['beta'] = np.load(f'beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']
    tau_values['zeta'] = np.load(f'zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']
    tau_values['exp'] = np.load(f'exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']

    betas = np.load(f'beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['betas']
    zetas = np.load(f'zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['zetas']

    fitting_profiles = ua_pbfs
    intrinsic_fwhms = -1

    #import data
    with open('j1903_data.pkl', 'rb') as fp:
        data_dict = pickle.load(fp)

    mjd_strings = list(data_dict.keys())
    mjds = np.zeros(np.size(mjd_strings))
    for i in range(np.size(mjd_strings)):
        mjds[i] = data_dict[mjd_strings[i]]['mjd']

    data_collect = {}

    mjd = ...
    data = data_dict[mjd_strings[i]]['data']
    freqs = data_dict[mjd_strings[i]]['freqs']
    dur = data_dict[mjd_strings[i]]['dur']

    prof = Profile_Fitting(mjd, data, freqs, dur, intrinsic_shape, betas, zetas, fitting_profiles, tau_values, intrinsic_fwhms)

    freq_list = np.zeros(prof.num_sub)

    tau_listb = np.zeros(prof.num_sub)

    beta_listb = np.zeros(prof.num_sub)
    beta_low_listb = np.zeros(prof.num_sub)
    beta_up_listb = np.zeros(prof.num_sub)

    for ii in range(prof.num_sub):

        datab = prof.fit(ii, pbf_type)

        print(f'Frequency = {prof.freq_round}')

        freq_list[ii] = prof.freq_suba
        tau_listb[ii] = datab['tau_fin']
        beta_listb[ii] = datab[pbf_type]
        beta_low_listb[ii] = datab[f'{pbf_type}_low']
        beta_up_listb[ii] = datab[f'{pbf_type}_up']

    data_collect[f'{int(np.round(mjd))}'] = {}
    data_collect[f'{int(np.round(mjd))}'][f'{pbf_type}_fit'] = beta_listb
    data_collect[f'{int(np.round(mjd))}'][f'{pbf_type}_low'] = beta_low_listb
    data_collect[f'{int(np.round(mjd))}'][f'{pbf_type}_up'] = beta_up_listb
    data_collect[f'{int(np.round(mjd))}']['tau'] = tau_listb
    data_collect[f'{int(np.round(mjd))}']['mjd'] = mjd
    data_collect[f'{int(np.round(mjd))}']['frequencies'] = freq_list

    with open(f'best_fit_{pbf_type}|{intrinsic_shape.upper()}.pkl', 'wb') as fp:
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

                beta_collect.append(data_collect[ii][f'{pbf_type}_fit'][here][0])

        beta_based_on_freq_pwrlawiwidth[f'{freqs[i]}'] = beta_collect

    beta_avgs_pwrlawiwidth = np.zeros(np.size(freqs))
    beta_stds_pwrlawiwidth = np.zeros(np.size(freqs))
    num_prof_per_freq_pwrlawiwidth = np.zeros(np.size(freqs))

    beta_based_on_freq_pwrlawiwidth = dict(sorted(beta_based_on_freq_pwrlawiwidth.items()))

    ind = 0

    for i in beta_based_on_freq_pwrlawiwidth.keys():

        print(f'Frequency = {i}')

        print(f'    {pbf_type} average = {np.average(beta_based_on_freq_pwrlawiwidth[i])}')

        num_prof_per_freq_pwrlawiwidth[ind] = len(beta_based_on_freq_pwrlawiwidth[i])

        beta_avgs_pwrlawiwidth[ind] = np.average(beta_based_on_freq_pwrlawiwidth[i])

        print(f'    {pbf_type} std = {np.std(beta_based_on_freq_pwrlawiwidth[i])}')

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
    ax.set_ylabel(f'{pbf_type[0].upper()+pbf_type[1:]}')
    ax.set_title(f'Best Fit {pbf_type[0].upper()+pbf_type[1:]}')
    plt.savefig(f'best_fit_{pbf_type}|{intrinsic_shape.upper()}.pdf')
    #plt.show()
    plt.close('all')

    del(beta_based_on_freq_pwrlawiwidth)
    del(beta_avgs_pwrlawiwidth)
    del(beta_stds_pwrlawiwidth)
    del(num_prof_per_freq_pwrlawiwidth)
    del(data_collect)
