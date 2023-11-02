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

pbf_type = sys.argv[1]
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
    data_collect = {}

    #import data
    with open('j1903_data.pkl', 'rb') as fp:
        data_dict = pickle.load(fp)

    mjd_strings = list(data_dict.keys())
    mjds = np.zeros(np.size(mjd_strings))
    for i in range(np.size(mjd_strings)):
        mjds[i] = data_dict[mjd_strings[i]]['mjd']

    for i in range(len(mjd_strings)):

        mjd = data_dict[mjd_strings[i]]['mjd']
        data = data_dict[mjd_strings[i]]['data']
        freqs = data_dict[mjd_strings[i]]['freqs']
        dur = data_dict[mjd_strings[i]]['dur']

        prof = Profile_Fitting(mjd, data, freqs, dur, intrinsic_shape, betas, zetas, fitting_profiles, tau_values, intrinsic_fwhms)

        freq_list = np.zeros(prof.num_sub)

        tau_listb = np.zeros(prof.num_sub)

        beta_listb = np.zeros(prof.num_sub)
        beta_low_listb = np.zeros(prof.num_sub)
        beta_up_listb = np.zeros(prof.num_sub)

        chi_sq_list = np.zeros(prof.num_sub)

        for ii in range(prof.num_sub):

            datab = prof.fit(ii, pbf_type)

            print(f'Frequency = {prof.freq_round}')

            freq_list[ii] = prof.freq_suba
            tau_listb[ii] = datab['tau_fin']
            beta_listb[ii] = datab[pbf_type]
            beta_low_listb[ii] = datab[f'{pbf_type}_low']
            beta_up_listb[ii] = datab[f'{pbf_type}_up']
            chi_sq_list[ii] = datab['low_chi']

        data_collect[f'{mjd}'] = {}
        data_collect[f'{mjd}'][f'{pbf_type}_fit'] = beta_listb
        data_collect[f'{mjd}'][f'{pbf_type}_low'] = beta_low_listb
        data_collect[f'{mjd}'][f'{pbf_type}_up'] = beta_up_listb
        data_collect[f'{mjd}']['tau'] = tau_listb
        data_collect[f'{mjd}']['mjd'] = mjd
        data_collect[f'{mjd}']['frequencies'] = freq_list
        data_collect[f'{mjd}']['low_chi'] = chi_sq_list

    #repeat for sband
    with open('j1903_sband_data.pkl', 'rb') as fp:
        data_dict = pickle.load(fp)

    mjd_strings = list(data_dict.keys())
    mjds = np.zeros(np.size(mjd_strings))
    for i in range(np.size(mjd_strings)):
        mjds[i] = data_dict[mjd_strings[i]]['mjd']

    for i in range(len(mjd_strings)):

        mjd = data_dict[mjd_strings[i]]['mjd']
        data = data_dict[mjd_strings[i]]['data']
        freqs = data_dict[mjd_strings[i]]['freqs']
        dur = data_dict[mjd_strings[i]]['dur']

        prof = Profile_Fitting(mjd, data, freqs, dur, intrinsic_shape, betas, zetas, fitting_profiles, tau_values, intrinsic_fwhms)

        freq_list = np.zeros(prof.num_sub)

        tau_listb = np.zeros(prof.num_sub)

        beta_listb = np.zeros(prof.num_sub)
        beta_low_listb = np.zeros(prof.num_sub)
        beta_up_listb = np.zeros(prof.num_sub)

        chi_sq_list = np.zeros(prof.num_sub)

        for ii in range(prof.num_sub):

            datab = prof.fit(ii, pbf_type)

            print(f'Frequency = {prof.freq_round}')

            freq_list[ii] = prof.freq_suba
            tau_listb[ii] = datab['tau_fin']
            beta_listb[ii] = datab[pbf_type]
            beta_low_listb[ii] = datab[f'{pbf_type}_low']
            beta_up_listb[ii] = datab[f'{pbf_type}_up']
            chi_sq_list[ii] = datab['low_chi']

        data_collect[f'{mjd}'] = {}
        data_collect[f'{mjd}'][f'{pbf_type}_fit'] = beta_listb
        data_collect[f'{mjd}'][f'{pbf_type}_low'] = beta_low_listb
        data_collect[f'{mjd}'][f'{pbf_type}_up'] = beta_up_listb
        data_collect[f'{mjd}']['tau'] = tau_listb
        data_collect[f'{mjd}']['mjd'] = mjd
        data_collect[f'{mjd}']['frequencies'] = freq_list
        data_collect[f'{mjd}']['low_chi'] = chi_sq_list

    with open(f'best_fit_{pbf_type}|{intrinsic_shape.upper()}|L&SBAND.pkl', 'wb') as fp:
        pickle.dump(data_collect, fp)

    with open(f'best_fit_{pbf_type}|{intrinsic_shape.upper()}|L&SBAND.pkl', 'rb') as fp:
        data_collect = pickle.load(fp)

    #===============================================================================
    # Now plotting the results - best fit beta mean and standard error for each of
    # 12 frequencies
    # ==============================================================================

    mjd_list = data_collect.keys()

    freqs = np.array([1180,1230,1280,1330,1380,1430,1480,1530,1580,1635,1690,\
    1740,1800,1850,1900,1950,2000,2050,2100,2150,2200,2250,2300,2350,2400])

    freqs_care = np.copy(freqs)

    beta_based_on_freq_pwrlawiwidth = {}
    chi_sq_based_on_freq_pwrlawiwidth = {}

    del_ind = 0
    for i in range(np.size(freqs)):

        beta_collect = []
        chi_sq_collect = 0.0
        number = 0

        for ii in mjd_list:

            here = np.where(((data_collect[ii]['frequencies'] > (freqs[i]-25)) & (data_collect[ii]['frequencies'] < (freqs[i]+25))))

            if np.size(here) > 0:

                beta_collect.append(data_collect[ii][f'{pbf_type}_fit'][here][0])
                chi_sq_collect += data_collect[ii][f'low_chi'][here][0]
                number += 1

        if number < 12:

            freqs_care = np.delete(freqs_care, i-del_ind)
            del_ind+=1

        else:

            beta_based_on_freq_pwrlawiwidth[f'{freqs[i]}'] = beta_collect
            chi_sq_based_on_freq_pwrlawiwidth[f'{freqs[i]}'] = chi_sq_collect/number

    beta_avgs_pwrlawiwidth = np.zeros(np.size(freqs_care))
    beta_stds_pwrlawiwidth = np.zeros(np.size(freqs_care))
    num_prof_per_freq_pwrlawiwidth = np.zeros(np.size(freqs_care))
    chi_sq_pwrlawiwidth = np.zeros(np.size(freqs_care))

    beta_based_on_freq_pwrlawiwidth = dict(sorted(beta_based_on_freq_pwrlawiwidth.items()))

    ind = 0

    for i in beta_based_on_freq_pwrlawiwidth.keys():

        print(f'Frequency = {i}')

        print(f'    {pbf_type} average = {np.average(beta_based_on_freq_pwrlawiwidth[i])}')

        num_prof_per_freq_pwrlawiwidth[ind] = len(beta_based_on_freq_pwrlawiwidth[i])

        beta_avgs_pwrlawiwidth[ind] = np.average(beta_based_on_freq_pwrlawiwidth[i])

        print(f'    {pbf_type} std = {np.std(beta_based_on_freq_pwrlawiwidth[i])}')

        beta_stds_pwrlawiwidth[ind] = np.std(beta_based_on_freq_pwrlawiwidth[i])

        chi_sq_pwrlawiwidth[ind] = chi_sq_based_on_freq_pwrlawiwidth[i]

        print(f'    Chi-Squared = {chi_sq_based_on_freq_pwrlawiwidth[i]}')

        print(f'    Number of Profiles = {num_prof_per_freq_pwrlawiwidth[ind]}')


        ind += 1

    plt.figure(1)
    plt.rc('font', family = 'serif')

    fig, axs = plt.subplots(3, sharex = True)

    markers, caps, bars = axs.flat[0].errorbar(x = freqs_care, y = beta_avgs_pwrlawiwidth, yerr = beta_stds_pwrlawiwidth/np.sqrt(num_prof_per_freq_pwrlawiwidth), fmt = '.', capsize = 2, color = 'k')

    [bar.set_alpha(0.4) for bar in bars]
    [cap.set_alpha(0.4) for cap in caps]

    axs.flat[1].plot(freqs_care, chi_sq_pwrlawiwidth, color = 'k')
    axs.flat[1].plot(freqs_care, chi_sq_pwrlawiwidth, '.', color = 'k')
    axs.flat[1].set_ylabel(r'$\chi^2$')
    axs.flat[2].plot(freqs_care, num_prof_per_freq_pwrlawiwidth, color = 'k')
    axs.flat[2].plot(freqs_care, num_prof_per_freq_pwrlawiwidth, '.', color = 'k')
    axs.flat[2].set_ylabel('Number of Profiles')

    axs.flat[2].set_xlabel('Frequency [MHz]')
    axs.flat[0].set_ylabel(f'{pbf_type[0].upper()+pbf_type[1:]}')
    axs.flat[0].set_title(f'Best Fit {pbf_type[0].upper()+pbf_type[1:]}')
    plt.savefig(f'best_fit_{pbf_type}|{intrinsic_shape.upper()}|L&SBAND.pdf')
    plt.close('all')
