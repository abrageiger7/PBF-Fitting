from pypulse.singlepulse import SinglePulse
from pypulse.singlepulse import SinglePulse
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline
from scipy import special
import itertools
import pickle
import matplotlib.ticker as tick
import sys
from scipy import optimize
from fit_functions import triple_gauss, calculate_tau, stretch_or_squeeze, convolve, init_data_phase_bins, j1903_period, time_average

plt.rc('font', family = 'serif')

beta = sys.argv[1] # '11_3', '3.99999', '3.1', '3.8', or '3.5'
zeta = sys.argv[2] # '0.05' or '0' (only '0.05' with '11_3' beta)

def best_intrinsic_params(frequency):

    '''Calculates the best fit intrinsic component parameters for J1903 at the
    given frequency. Also calculates the tau value at this frequency corresponding
    to the measured sband tau with this given pbf and a kolmogorov -4.4 powerlaw.

    int/float frequency: the j1903 frequency care about.'''

    # best fit powerlaws for intrinsic shape (as of 9/22/23 from average
    # over mjd)
    a2 = 1.0
    a1_pwrlaw = 0.7
    a3_pwrlaw = -0.4
    phi3_pwrlaw = 0.0
    w3_pwrlaw = -1.0
    kolmogorov_pwrlaw = -4.4

    # sband fitted parameters from average oer mjd (as of 9/22/23)
    starting_freq = 2200.0
    parameters = np.load(f'mcmc_params|FREQ=2200|BETA=11_3|ZETA=0.npz')
    comp1 = parameters['parameters'][:3]
    comp2 = [a2, parameters['parameters'][3], parameters['parameters'][4]]
    comp3 = parameters['parameters'][5:8]

    diff_freq_comp1 = comp1
    diff_freq_comp1[0] = comp1[0]*np.power((frequency/starting_freq),a1_pwrlaw)
    diff_freq_comp2 = comp2
    diff_freq_comp3 = comp3
    diff_freq_comp3[0] = comp3[0]*np.power((frequency/starting_freq),a3_pwrlaw)
    diff_freq_comp3[1] = comp3[1]*np.power((frequency/starting_freq),phi3_pwrlaw)
    diff_freq_comp3[2] = comp3[2]*np.power((frequency/starting_freq),w3_pwrlaw)

    tau_sband = parameters['parameters'][8]
    tau_freq = tau_sband*np.power((frequency/starting_freq),kolmogorov_pwrlaw)

    return(diff_freq_comp1, diff_freq_comp2, diff_freq_comp3, tau_freq)

if __name__ == '__main__':

    pbf = np.load(f'zeta_{zeta}_beta_{beta}_pbf.npy')

    cordes_phase_bins = np.size(pbf)
    phase_bins = init_data_phase_bins

    timer_period = np.linspace(0,2.15,phase_bins)

    if (np.size(pbf) != phase_bins):
        subs_time_avg = time_average(pbf, phase_bins)
    else:
        subs_time_avg = pbf

    for ii in range(np.size(subs_time_avg)):
        subs_time_avg[ii] = np.average(pbf[((cordes_phase_bins//phase_bins)*ii):((cordes_phase_bins//phase_bins)*(ii+1))])
    subs_time_avg = subs_time_avg / trapz(subs_time_avg)

    tau_subs_time_avg = calculate_tau(subs_time_avg)[0]

    frequencies = np.linspace(1100,2400,14)

    plt.figure(1)

    toa_delays = []
    taus = []

    plt.plot([], label = r'$\nu$ [MHs]', color = 'white')

    for i in range(np.size(frequencies)):

        comp1, comp2, comp3, tau = best_intrinsic_params(frequencies[i])

        taus.append(tau)

        intrinsico = triple_gauss(comp1, comp2, comp3, np.arange(phase_bins))[0]
        pbf = stretch_or_squeeze(subs_time_avg, tau/tau_subs_time_avg)

        spi = SinglePulse(intrinsico)

        plt.xlabel('Period [ms]')

        intrinsic = np.copy(intrinsico)

        conv_temp = convolve(intrinsic, pbf)/trapz(convolve(intrinsic, pbf))

        plt.plot(timer_period, conv_temp, label = f'{int(frequencies[i])}')

        #Calculates mode of data profile to shift template to
        x = np.max(conv_temp)
        xind = np.where(conv_temp == x)[0][0]

        intrinsic = intrinsic / np.max(intrinsic) #fitPulse likes template height of one
        z = np.max(intrinsic)
        zind = np.where(intrinsic == z)[0][0]
        ind_diff = zind-xind

        conv_temp = np.roll(conv_temp, ind_diff)
        sp = SinglePulse(conv_temp)

        fitting = sp.fitPulse(intrinsic)

        toa_delays.append((fitting[1]-ind_diff) * j1903_period / phase_bins) #microseconds

    plt.title(r'$\beta$=11/3; $\zeta$=0')
    plt.legend(fontsize = 8, loc = 'upper right')
    plt.savefig('j1903_modeled_convolved_varying_frequency.pdf')
    plt.show()
    plt.close('all')

    plt.figure(2)
    plt.plot(frequencies, toa_delays, '.', color = 'k')
    plt.plot(frequencies, toa_delays, color = 'k')
    plt.title(r'J1903 Modeled Intrinsic; Thick Medium PBF $\beta$=11/3; $\zeta$=0')
    plt.xlabel(r'Frequency [MHz]')
    plt.ylabel('TOA Delay [$\mu$s]')
    plt.savefig('j1903_modeled_time_delay_vs_frequency.pdf')
    plt.show()
    plt.close('all')

    average_pwrlaw = (math.log10(toa_delays[-1])-math.log10(toa_delays[0]))/(math.log10(frequencies[-1])-math.log10(frequencies[0]))
    center_freq_ind = np.size(frequencies)//2
    avg_pwrlaw_line = toa_delays[center_freq_ind]*np.power(frequencies/frequencies[center_freq_ind],average_pwrlaw)

    plt.figure(3)
    fig, ax = plt.subplots()
    ax.plot(frequencies, toa_delays, '.', color = 'k')
    ax.plot(frequencies, toa_delays, color = 'k')
    ax.plot(frequencies, avg_pwrlaw_line, ls = '--', label = f'{np.round(average_pwrlaw,1)}', color = 'grey')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([])
    for ticking in ax.xaxis.get_minor_ticks():
        ticking.label1.set_visible(False)
    ax.xaxis.set_major_formatter(tick.ScalarFormatter())
    ax.set_xticks(frequencies[::3])
    ax.set_title(r'J1903 Modeled Intrinsic; Thick Medium PBF $\beta$=11/3; $\zeta$=0')
    ax.set_xlabel(r'Frequency [MHz]')
    ax.set_ylabel('TOA Delay [$\mu$s]')
    plt.legend()
    plt.savefig('j1903_modeled_time_delay_vs_frequency_logspace.pdf')
    plt.show()
    plt.close('all')

    plt.figure(4)
    plt.plot(taus, toa_delays, '.', color = 'k')
    plt.plot(taus, toa_delays, color = 'k')
    plt.title(r'J1903 Modeled Intrinsic; Thick Medium PBF $\beta$=11/3; $\zeta$=0')
    plt.xlabel(r'$\tau$ [$\mu$s]')
    plt.ylabel('TOA Delay [$\mu$s]')
    plt.savefig('j1903_modeled_time_delay_vs_tau_set-4.4_logspace.pdf')
    plt.show()
    plt.close('all')

    # Now fitting the scattering time delays to a dispersive sweep to calculate
    # the delta DM induced by this Scattering

    # need to update with new scattering dm correction
    calc_delta_DM(1e-2, 0.0, frequencies, toa_delays)
