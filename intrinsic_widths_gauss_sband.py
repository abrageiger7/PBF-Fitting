'''Fitting for best fit intrinsic gaussian and intrinsic sband shape widths at
a number of different frequencies and mjds'''

import sys
import numpy as np
import pickle
from fit_functions import *
import matplotlib.ticker as tick
from profile_class import Profile_Fitting
import os

beta = float(sys.argv[1])
zeta = float(sys.argv[2])

if zeta != 0.0 and beta != 3.667:
    raise Exception('Invalid PBF Selected: Zeta greater that zero is only available for Beta = 3.667.')

if __name__ == '__main__':

    #load in the J1903 data
    with open('j1903_data.pkl', 'rb') as fp:
        data_dict = pickle.load(fp)

    mjd_strings = list(data_dict.keys())
    mjds = np.zeros(np.size(mjd_strings))
    for i in range(np.size(mjd_strings)):
        mjds[i] = data_dict[mjd_strings[i]]['mjd']

    for intrinsic_shape in ['gaussian', 'sband_avg']:

        #load and memory map the profile fitting grid
        fitting_profile_data = np.load(f'convolved_profiles_intrinsic={intrinsic_shape}|PHASEBINS={phase_bins}.npz')
        convolved_profiles = {}
        tau_values = {}

        convolved_profiles_beta = np.memmap('beta_convolved_profiles|SCRIPT=intrinsic_widths_gauss_sband', dtype='float64', mode='w+', shape=np.shape(fitting_profile_data['beta_profiles']))
        convolved_profiles_beta[:] = fitting_profile_data['beta_profiles'][:]
        convolved_profiles_beta.flush()
        convolved_profiles['beta'] = convolved_profiles_beta
        tau_values['beta'] = fitting_profile_data['beta_taus_mus']
        betas = fitting_profile_data['betas']

        convolved_profiles_zeta = np.memmap('zeta_convolved_profiles|SCRIPT=intrinsic_widths_gauss_sband', dtype='float64', mode='w+', shape=np.shape(fitting_profile_data['zeta_profiles']))
        convolved_profiles_zeta[:] = fitting_profile_data['zeta_profiles'][:]
        convolved_profiles_zeta.flush()
        convolved_profiles['zeta'] = convolved_profiles_zeta
        tau_values['zeta'] = fitting_profile_data['zeta_taus_mus']
        zetas = fitting_profile_data['zetas']

        convolved_profiles_exp = np.memmap('exp_convolved_profiles|SCRIPT=intrinsic_widths_gauss_sband', dtype='float64', mode='w+', shape=np.shape(fitting_profile_data['exp_profiles']))
        convolved_profiles_exp[:] = fitting_profile_data['exp_profiles'][:]
        convolved_profiles_exp.flush()
        convolved_profiles['exp'] = convolved_profiles_exp
        tau_values['exp'] = fitting_profile_data['exp_taus_mus']

        fitting_profiles = convolved_profiles

        intrinsic_fwhms = fitting_profile_data['intrinsic_fwhm_mus']

        del(fitting_profile_data)

        if beta != 3.667 or zeta == 0:

            type_test = 'beta'
            bzeta_ind = int(find_nearest(betas,beta)[1][0][0])

        else:

            type_test = 'zeta'
            bzeta_ind = int(find_nearest(zetas,zeta)[1][0][0])

        for pbf_type in ['exp', type_test]:

            frequencies_collect = []
            mjds_collect = []
            best_fit_width_collect = []

            for i in range(0,56,8):

                '''First calculate and collect tau values with errors'''
                print(f'MJD {i}')
                mjd = data_dict[mjd_strings[i]]['mjd']
                data = data_dict[mjd_strings[i]]['data']
                freqs = data_dict[mjd_strings[i]]['freqs']
                dur = data_dict[mjd_strings[i]]['dur']

                prof = Profile_Fitting(mjd, data, freqs, dur, intrinsic_shape, betas, zetas, fitting_profiles, tau_values, intrinsic_fwhms)

                for ii in range(0,prof.num_sub,2):

                    if pbf_type == 'beta' or pbf_type == 'zeta':
                        datab = prof.fit(ii, pbf_type, bzeta_ind = bzeta_ind)
                    elif pbf_type == 'exp':
                        datab = prof.fit(ii, pbf_type)

                    frequencies_collect.append(prof.freq_suba) #frequency
                    mjds_collect.append(prof.mjd) #mjd
                    best_fit_width_collect.append(datab['intrins_width']) # best fit width (microseconds)

            if pbf_type == 'beta':
                title = f'intrinsic_best_fit_widths|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}'
            elif pbf_type == 'zeta':
                title = f'intrinsic_best_fit_widths|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}'
            elif pbf_type == 'exp':
                title = f'intrinsic_best_fit_widths|{intrinsic_shape.upper()}|{pbf_type.upper()}'

            np.savez(title, freqs = frequencies_collect, mjds = mjds_collect, widths = best_fit_width_collect)


            # PLOTTING

            plt.figure(1)
            plt.plot(frequencies_collect, best_fit_width_collect, '.', color = 'k', alpha = 0.5)
            plt.xlabel('Frequency [MHz]')
            if intrinsic_shape == 'gaussian':
                plt.ylabel(r'Gaussian Width [$\mu$s]')
                plt.title('Best Fit Intrinsic Width')
            elif intrinsic_shape == 'sband_avg':
                plt.ylabel(r'Intrinsic Width [$\mu$s]')
                plt.title('Best Fit Intrinsic Width')

            if pbf_type == 'beta':
                title = f'intrinsic_best_fit_widths_plotted_vs_freq|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}.pdf'
            elif pbf_type == 'zeta':
                title = f'intrinsic_best_fit_widths_plotted_vs_freq|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}.pdf'
            elif pbf_type == 'exp':
                title = f'intrinsic_best_fit_widths_plotted_vs_freq|{intrinsic_shape.upper()}|{pbf_type.upper()}.pdf'

            plt.savefig(title)
            plt.close('all')


            plt.figure(1)
            plt.plot(mjds_collect, best_fit_width_collect, '.', color = 'k', alpha = 0.5)
            plt.xlabel('MJD')
            if intrinsic_shape == 'gaussian':
                plt.ylabel(r'Gaussian Width [$\mu$s]')
                plt.title('Best Fit Intrinsic Width')
            elif intrinsic_shape == 'sband_avg':
                plt.ylabel(r'Intrinsic Width [$\mu$s]')
                plt.title('Best Fit Intrinsic Width')

            if pbf_type == 'beta':
                title = f'intrinsic_best_fit_widths_plotted_vs_mjd|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}.pdf'
            elif pbf_type == 'zeta':
                title = f'intrinsic_best_fit_widths_plotted_vs_mjd|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}.pdf'
            elif pbf_type == 'exp':
                title = f'intrinsic_best_fit_widths_plotted_vs_mjd|{intrinsic_shape.upper()}|{pbf_type.upper()}.pdf'

            plt.savefig(title)
            plt.close('all')

        os.remove('beta_convolved_profiles|SCRIPT=intrinsic_widths_gauss_sband')
        os.remove('zeta_convolved_profiles|SCRIPT=intrinsic_widths_gauss_sband')
        os.remove('exp_convolved_profiles|SCRIPT=intrinsic_widths_gauss_sband')
