from mcmc_profile_fitting_class import *
from fit_functions import *
from intrinsic_component_powerlaw_fit import *
from pathlib import Path

def epoch_average_intrinsic_fitting(beta, zeta, screen, rerun, nruns=0.0, tau_ges=0.0, amp1_tests='default', amp3_tests='default', cent3_tests='default', width3_tests='default'):

    if rerun == 'mcmc' or rerun == 'both':
        numruns = nruns # number of mcmc runs in order to calculate sband
        # shape parameters
        tau_guess = tau_ges

    plt.rc('font', family = 'serif')

    ##IMPORT DATA##

    sband = np.load(Path('/Users/abrageiger/Documents/research/projects/pbf_fitting/j1903_high_freq_temp_unsmoothed.npy'))
    # reference nanograv notebook s_band_j1903_data.ipynb for frequency calcualtion
    sband_avg_center_freq = 2132.0

    # load in the j1903 subband averages - these are averaged over all epochs
    with open('/Users/abrageiger/Documents/research/projects/pbf_fitting/j1903_average_profiles_lband.pkl', 'rb') as fp:
        lband_avgs = pickle.load(fp)

    freq_keys = []
    lband_freqs = []
    for i in lband_avgs.keys():
        freq_keys.append(i)
        lband_freqs.append(float(i))

    # frequencies corresponding to the j1903 subband averages
    lband_freqs = np.array(lband_freqs)

    lband_data_array = np.zeros((len(freq_keys), np.size(lband_avgs[freq_keys[0]])))
    ind = 0
    for i in freq_keys:
        lband_data_array[ind] = lband_avgs[i]
        ind += 1

    mcmc_fitting_object = MCMC_Profile_Fit(beta, zeta, screen, sband, \
    sband_avg_center_freq, 'mjd_average')

    if rerun == 'mcmc' or rerun == 'both':

        params = mcmc_fitting_object.profile_component_fit(numruns, tau_guess)
        # three_gaussian_fitting_parameters = np.load('mcmc_params|'+mcmc_fitting_object.plot_tag+'.npz')['parameters']
        # mcmc_fitting_object.plot_a_fit_sp(three_gaussian_fitting_parameters[:8], three_gaussian_fitting_parameters[8])

    powerlaw_fitting_object = Intrinsic_Component_Powerlaw_Fit(beta, zeta, \
    'mjd_average', 'mcmc_params|'+mcmc_fitting_object.plot_tag+'.npz', \
    mcmc_fitting_object.frequency, screen, lband_data_array, lband_freqs, \
    sband)

    if rerun == 'powerlaws' or rerun == 'both':

        if amp1_tests == 'default' and amp3_tests == 'default' and cent3_tests == 'default' and width3_tests == 'default':

            # 3.667 0.0 thick
            if (beta == '3.667' and (zeta == '0.0' or zeta == '0') and screen == 'thick'):
                print('Thick with Beta = 3.667 and Zeta = 0.0')
                amp1_tests=np.array([1.0])
                amp3_tests=np.array([-0.2])
                cent3_tests=np.array([0.1])
                width3_tests=np.array([-1.1])

            # 3.975 0.0 thick
            elif (beta == '3.975' and (zeta == '0.0' or zeta == '0') and screen == 'thick'):
                print('Thick with Beta = 3.975 and Zeta = 0.0')
                amp1_tests=np.linspace(0.8,1.5,8)
                amp3_tests=np.linspace(0.0,0.7,8)
                cent3_tests=np.linspace(-0.8,-0.1,8)
                # had to extend search range to 8.0
                width3_tests=np.linspace(1.0,3.0,8)

            # 3.667 5.0 thick
            elif (beta == '3.667' and (zeta == '5.0' or zeta == '5') and screen == 'thick'):
                print('Thick with Beta = 3.667 and Zeta = 5.0')
                amp1_tests=np.linspace(1.2,1.9,8)
                amp3_tests=np.linspace(-0.6,0.1,8)
                cent3_tests=np.linspace(-0.8,-0.1,8)
                #had to extend search range to 4.5
                width3_tests=np.linspace(2.4,3.6,8)

            # 3.1 0.0 thick
            elif (beta == '3.1' and (zeta == '0.0' or zeta == '0') and screen == 'thick'):
                print('Thick with Beta = 3.1 and Zeta = 0.0')
                amp1_tests=np.array([1.6])
                amp3_tests=np.array([-2.0])
                cent3_tests=np.array([0.5])
                width3_tests=np.array([-1.4])

            # 3.5 0.0 thick
            elif (beta == '3.5' and (zeta == '0.0' or zeta == '0') and screen == 'thick'):
                print('Thick with Beta = 3.5 and Zeta = 0.0')
                # ella
                amp1_tests=np.array([1.0])
                amp3_tests=np.array([-1.1])
                cent3_tests=np.array([0.3])
                width3_tests=np.array([-1.3])

            # 3.1 0.01 thin
            elif (beta == '3.1' and (zeta == '0.01') and screen == 'thin'):
                print('Thin with Beta = 3.1 and Zeta = 0.01')
                amp1_tests=np.linspace(-2.5,2.5,8)
                amp3_tests=np.linspace(-2.5,2.5,8)
                cent3_tests=np.linspace(-2.5,2.5,8)
                width3_tests=np.linspace(-2.5,2.5,8)

            # 3.5 0.01 thin
            elif (beta == '3.5' and (zeta == '0.01') and screen == 'thin'):
                print('Thin with Beta = 3.5 and Zeta = 0.01')
                amp1_tests=np.linspace(-2.5,2.5,8)
                amp3_tests=np.linspace(-2.5,2.5,8)
                cent3_tests=np.linspace(-2.5,2.5,8)
                width3_tests=np.linspace(-2.5,2.5,8)

            # 3.667 0.01 thin
            elif (beta == '3.667' and (zeta == '0.01') and screen == 'thin'):
                print('Thin with Beta = 3.667 and Zeta = 0.01')
                amp1_tests=np.linspace(-2.5,2.5,8)
                amp3_tests=np.linspace(-2.5,2.5,8)
                cent3_tests=np.linspace(-2.5,2.5,8)
                width3_tests=np.linspace(-2.5,2.5,8)

            # 3.975 0.01 thin
            elif (beta == '3.975' and (zeta == '0.01') and screen == 'thin'):
                print('Thin with Beta = 3.975 and Zeta = 0.01')
                amp1_tests=np.linspace(-2.5,2.5,8)
                amp3_tests=np.linspace(-2.5,2.5,8)
                cent3_tests=np.linspace(-2.5,2.5,8)
                width3_tests=np.linspace(-2.5,2.5,8)

            # 3.667 5.0 thin
            elif (beta == '3.667' and (zeta == '5.0') and screen == 'thin'):
                print('Thin with Beta = 3.667 and Zeta = 5.0')
                amp1_tests=np.linspace(-2.5,2.5,8)
                amp3_tests=np.linspace(-2.5,2.5,8)
                cent3_tests=np.linspace(-2.5,2.5,8)
                width3_tests=np.linspace(-2.5,2.5,8)

        print(amp3_tests)
        print(cent3_tests)
        print(width3_tests)
        print(amp1_tests)

        powerlaw_fitting_object.fit_comp3(amp3_tests, cent3_tests, width3_tests)
        # powerlaw_fitting_object.fit_comp3(10)
        powerlaw_fitting_object.fit_amp1(amp1_tests)
        powerlaw_fitting_object.plot_modeled()
        powerlaw_fitting_object.plot_modeled_fitted()
