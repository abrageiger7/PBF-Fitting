import pickle
import numpy as np
import matplotlib.pyplot as plt
from pypulse.singlepulse import SinglePulse
from scipy.integrate import trapz
from scipy import optimize
from scipy.interpolate import CubicSpline
import math
from pypulse.singlepulse import SinglePulse
from fit_functions import triple_gauss, convolve, stretch_or_squeeze, \
calculate_tau
import sys
import emcee
import corner
from mcmc_profile_fitting_class import MCMC_Profile_Fit

'''Script to fit the 3 gaussian components and tau of a single profile'''

beta = sys.argv[1]
zeta = sys.argv[2]
numruns = int(sys.argv[3])
rerun = str(sys.argv[4])

plt.rc('font', family = 'serif')

if __name__ == '__main__':

    #import the j1903 subaverages corresponding to these frequencies
    freqs = np.array([1160,1260,1360,1460,1560,1660,1760,2200])
    with open('j1903_average_profiles_lband.pkl', 'rb') as fp:
        lband_avgs = pickle.load(fp)
    sband = np.load('j1903_high_freq_temp_unsmoothed.npy')
    sband = sband / trapz(sband)
    lband_avgs['2200'] = sband
    freq_keys = []
    for i in lband_avgs.keys():
        freq_keys.append(i)
    phase = np.linspace(0,1,np.size(sband))
    t_phasebins = np.arange(np.size(sband))

    #calculate the rms values for each frequency
    opr_size = np.size(lband_avgs[freq_keys[0]])//5
    rms_values = np.zeros(np.size(freqs))
    for ii in range(np.size(freqs)):
        rms_collect = 0
        for i in range(opr_size):
            rms_collect += lband_avgs[freq_keys[ii]][i]**2
        rms = math.sqrt(rms_collect/opr_size)
        rms_values[ii] = rms

    #import the pbf and rescale it to have 2048 phasebins like the data
    pbf = np.load(f'zeta_{zeta}_beta_{beta}_pbf.npy')
    cordes_phase_bins = np.size(pbf)
    phase_bins = np.size(sband)
    subs_time_avg = np.zeros(phase_bins)

    for ii in range(np.size(subs_time_avg)):
            subs_time_avg[ii] = np.average(pbf[((cordes_phase_bins//phase_bins)*ii):((cordes_phase_bins//phase_bins)*(ii+1))])
    subs_time_avg = subs_time_avg / trapz(subs_time_avg)

    tau_subs_time_avg = calculate_tau(subs_time_avg)[0]

    freq_ind = 7
    profile = lband_avgs[freq_keys[freq_ind]]
    frequency = freqs[freq_ind]
    yerr = np.zeros(np.size(lband_avgs[freq_keys[freq_ind]]))
    yerr.fill(rms_values[freq_ind])
    tau_guess = 11.0 #microseconds

    if rerun == 'rerun':

        params = profile_component_fit(frequency, profile, yerr, fixed_order_for_components_starting_values, numruns, 'mjd_average', include_plots = True)
        np.savez(f'mcmc_params|FREQ={frequency}|BETA={beta}|ZETA={zeta}|MJD_AVERAGE', parameters = params[0], params_low_err = params[1], params_high_err = params[2])

    else:

        parameters = np.load(f'mcmc_params|FREQ={frequency}|BETA={beta}|ZETA={zeta}|MJD_AVERAGE.npz')
        comp1 = parameters['parameters'][:3]
        comp2 = [a2, parameters['parameters'][3], parameters['parameters'][4]]
        comp3 = parameters['parameters'][5:8]
        tau = parameters['parameters'][8]

        profile_fitted = convolve(triple_gauss(comp1, comp2, comp3, np.arange(np.size(profile)))[0], stretch_or_squeeze(subs_time_avg, tau/tau_subs_time_avg))

        sp = SinglePulse(profile)
        fitting = sp.fitPulse(profile_fitted)

        sps = SinglePulse(profile_fitted*fitting[2])
        fitted_template = sps.shiftit(fitting[1])

        plt.figure(1)
        plt.plot(phase, profile/np.max(profile), color = 'darkgrey', lw = 2.0)
        plt.plot(phase, fitted_template/np.max(profile), color = 'k')
        plt.ylabel('Normalised Flux')
        plt.xlabel('Pulse Phase')
        plt.savefig(f'mcmc_fitted|FREQ={np.round(frequency)}|BETA={beta}|ZETA={zeta}|MJD_AVERAGE.pdf')
        plt.show()
        plt.close('all')
