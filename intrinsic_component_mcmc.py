import pickle
import numpy as np
import matplotlib.pyplot as plt
from pypulse.singlepulse import SinglePulse
from scipy.integrate import trapz
from scipy import optimize
from scipy.interpolate import CubicSpline
import math
from pypulse.singlepulse import SinglePulse
from fit_functions import triple_gauss, convolve, stretch_or_squeeze, calculate_tau
import sys
import emcee
import corner

'''Script to fit the 3 gaussian components and tau of a single profile'''

beta = sys.argv[1]
zeta = sys.argv[2]
numruns = int(sys.argv[3])
rerun = str(sys.argv[4])

plt.rc('font', family = 'serif')

#the MCMC test parameters in question in this order for the duration of the code
#labels = [r'$X_{A_1}$', r'$X_{A_3}$', r'$X_{\phi_3}$', r'$X_{W_3}$', r'$\tau$',]
labels = [r'$A_1$', r'$\phi_1$', r'$W_1$', r'$\phi_2$', r'$W_2$', r'$A_3$', r'$\phi_3$', r'$W_3$', r'$\tau$']

#defines the likelihood function for the MCMC samples
#takes the parameters to test for this sampling, the data, and the y-error and returns the ln(likelihood)
#the ln(likelihood) is simply the ln() of a gaussian distribution
#the likelihood is calculated based upon the fit residuals, the number of data points, and the yerr
#yerr is calculated by the rms of the off pulse profile

a2 = 1.0

def ln_likelihood(theta, x, y, yerr):
    '''Returns ln(likelihood) for the parameters, theta, which in this case are
    Spectal indices for amplitudes of intrinsic components 1 and 3, as well as
    width of 3, phase of 3, and tau.
    '''
    a1, phi1, w1, phi2, w2, a3, phi3, w3, tau = theta
    comp1 = [a1, phi1, w1]
    comp2 = [a2, phi2, w2]
    comp3 = [a3, phi3, w3]

    profile = convolve(triple_gauss(np.abs(comp1), np.abs(comp2), np.abs(comp3), x)[0], stretch_or_squeeze(subs_time_avg, np.abs(tau/tau_subs_time_avg)))
    sp = SinglePulse(y)
    fitting = sp.fitPulse(profile)
    sps = SinglePulse(profile*fitting[2])
    model = sps.shiftit(fitting[1])

    resids = y - model

    N = len(y)
    lnL = -(N/2)*np.log(2*np.pi) - np.log(yerr).sum() - 0.5*np.power(resids/yerr,2).sum()
    return lnL

def ln_prior(theta):
    '''defines the acceptable probabalistic ranges in which the parameters can be expected to fall 100% of the time
    returns either 0.0 of -np.inf depending if the sample parameters are reasonable or not'''
    a1, phi1, w1, phi2, w2, a3, phi3, w3, tau = theta
    #amplitudes relative to a2 which is 1, phases from 0-1, widths relative to
    #percentage of phase
    if 0.01 < a1 < 0.1 and 0.2 < a3 < 0.6 and 0.31 < phi1 < 0.51 and 0.4 < phi2 < 0.6 and 0.45 < phi3 < 0.65 and 0.005 < w1 < 0.1 and 0.005 < w2 < 0.1 and 0.005 < w3 < 0.1 and 1.0 < tau < 30.0:
        return 0.0
    return -np.inf


def ln_probability(theta, x, y, yerr):
    '''defines probability based upon the established accepted parameter ranges
    and the ln_likelihood function. returns the probability for the sample parameters'''
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(theta, x, y, yerr)


def profile_component_fit(frequency, profile, yerr, starting_values, numruns, include_plots = True):

    starting_values = np.abs(starting_values)

    #initializes the MCMC
    pos = np.zeros((72,9))

    #setting starting amp range
    pos[:,::5] = starting_values[::5] + 1e-4 * np.random.randn(72, 2)

    #setting starting phi range
    pos[:,1:4:2] = starting_values[1:4:2] + 1e-4 * np.random.randn(72, 2)
    pos[:,6] = starting_values[6] + 1e-4 * np.random.randn(72)

    #setting starting width range
    pos[:,2] = starting_values[2] + 1e-5 * np.random.randn(72)
    pos[:,4] = starting_values[4] + 1e-5 * np.random.randn(72)
    pos[:,7] = starting_values[7] + 1e-5 * np.random.randn(72)

    pos[:,8] = starting_values[8] + 1e-4 * np.random.randn(72)

    #pos = starting_values + 1e-4 * np.random.randn(72, 9)
    nwalkers, ndim = pos.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_probability, args=(np.arange(np.size(profile)), profile, yerr))
    #runs the MCMC
    #the parameters from some channels of some files are very poorly estimated by least sqs, so this many samples is
    #necessary
    #for example, file 0 channel 31 is very poorly fit with least sqs and caused problems in runs with fewer samples
    #SET INITIAL GUESS PARAMETERS!
    sampler.run_mcmc(pos, numruns, progress=True);
    if include_plots == True:
        plt.figure(1)
        figure, axes = plt.subplots(np.size(starting_values), figsize = (10,7), sharex=True)
        samples_init = sampler.get_chain()
        for i in range(ndim):
            data = samples_init[:,:,i]
            ax = axes[i]
            ax.set_xlim(0, len(samples_init))
            ax.set_xlabel("Iterations")
            ax.set_ylabel(labels[i])
            ax.plot(data, 'k', alpha=0.2)
        plt.savefig(f'test3_mcmc_sampling|FREQ={np.round(frequency)}.pdf')
        plt.show()
        plt.close('all')
    #discards approximately 15% of the data and thins by approximately half the autocorrelation time
    auto_corr = sampler.get_autocorr_time()
    print(auto_corr)
    samples = sampler.get_chain(discard=int(numruns/2), thin=5, flat=True)
    [a1, phi1, w1, phi2, w2, a3, phi3, w3, tau] = np.percentile(samples, 50, axis = 0)
    parameters = [a1, phi1, w1, phi2, w2, a3, phi3, w3, tau]

    comp1 = [a1, phi1, w1]
    comp2 = [a2, phi2, w2]
    comp3 = [a3, phi3, w3]

    # w1 = w1 * (0.0021499) * 1e6 #microseconds
    # w2 = w2 * (0.0021499) * 1e6 #microseconds
    # w3 = w3 * (0.0021499) * 1e6 #microseconds

    [a1_low, phi1_low, w1_low, phi2_low, w2_low, a3_low, phi3_low, w3_low, tau_low] = (np.percentile(samples, 50, axis = 0) - np.percentile(samples, 16, axis = 0))
    [a1_high, phi1_high, w1_high, phi2_high, w2_high, a3_high, phi3_high, w3_high, tau_high] = (np.percentile(samples, 84, axis = 0) - np.percentile(samples, 50, axis = 0))
    parameters_low = [a1_low, phi1_low, w1_low, phi2_low, w2_low, a3_low, phi3_low, w3_low, tau_low]
    parameters_high = [a1_high, phi1_high, w1_high, phi2_high, w2_high, a3_high, phi3_high, w3_high, tau_high]

    # w1_low = w1_low * (0.0021499) * 1e6 #microseconds
    # w1_high = w1_high * (0.0021499) * 1e6 #microseconds
    # w2_low = w2_low * (0.0021499) * 1e6 #microseconds
    # w2_high = w2_high * (0.0021499) * 1e6 #microseconds
    # w3_low = w3_low * (0.0021499) * 1e6 #microseconds
    # w3_high = w3_high * (0.0021499) * 1e6 #microseconds

    profile_fitted = convolve(triple_gauss(comp1, comp2, comp3, np.arange(np.size(profile)))[0], stretch_or_squeeze(subs_time_avg, tau/tau_subs_time_avg))

    sp = SinglePulse(profile)
    fitting = sp.fitPulse(profile_fitted)

    sps = SinglePulse(profile_fitted*fitting[2])
    fitted_template = sps.shiftit(fitting[1])

    if (include_plots == True):
        plt.figure(1)
        plt.plot(phase, profile, color = 'lightsteelblue', lw = 2.8)
        plt.plot(phase, fitted_template, color = 'midnightblue')
        plt.xlabel('Pulse Phase')
        plt.title(f'J1903 MCMC {frequency} [MHz]')
        plt.savefig(f'test3_mcmc_fitted|FREQ={np.round(frequency)}.pdf')
        plt.show()
        plt.close('all')

        plt.figure(1)
        #beware: returns corner plot a decent amount of run-time after moving on to other channels
        fig = corner.corner(samples,bins=50,color='C0',smooth=0.5,plot_datapoints=False,plot_density=True,plot_contours=True,fill_contour=False,show_titles=True, labels = labels)
        plt.savefig(f'test3_mcmc_corner|FREQ={np.round(frequency)}.pdf')
        fig.show()
        plt.close('all')

    return(parameters, parameters_low, parameters_high)

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

    #starting values
    starting_values = np.zeros(9)
    sp = SinglePulse(profile)
    componented = sp.component_fitting(nmax = 3, full = True, alpha = 0.01)
    sband_comp_params = componented[1]
    #normalize gaussian amplitudes relative to second component amplitude
    sband_comp_params[::3] = sband_comp_params[::3] / sband_comp_params[3]
    for i in range(np.size(sband_comp_params)):
        if i != 3:
            ind = i
            if i > 3:
                ind = i-1
            starting_values[ind] = sband_comp_params[i]
    starting_values[8] = tau_guess

    fixed_order_for_components_starting_values = np.zeros(9)
    fixed_order_for_components_starting_values[:3] = starting_values[5:8]
    fixed_order_for_components_starting_values[3:5] = starting_values[3:5]
    fixed_order_for_components_starting_values[5:8] = starting_values[:3]
    fixed_order_for_components_starting_values[8] = starting_values[8]

    if rerun == 'rerun':

        params = profile_component_fit(frequency, profile, yerr, fixed_order_for_components_starting_values, numruns, include_plots = True)
        np.savez(f'test3_mcmc_params|FREQ={frequency}', parameters = params[0], params_low_err = params[1], params_high_err = params[2])

    else:

        parameters = np.load(f'test3_mcmc_params|FREQ={frequency}.npz')
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
        plt.plot(phase, profile, color = 'lightsteelblue', lw = 2.8)
        plt.plot(phase, fitted_template, color = 'midnightblue')
        plt.xlabel('Pulse Phase')
        plt.title(f'J1903 MCMC {frequency} [MHz]')
        plt.savefig(f'test3_mcmc_fitted|FREQ={np.round(frequency)}.pdf')
        plt.show()
        plt.close('all')
