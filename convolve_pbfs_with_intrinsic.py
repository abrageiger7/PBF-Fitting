import numpy as np
from numpy import zeros, size
from numpy import interp
from scipy.interpolate import CubicSpline
from pypulse.singlepulse import SinglePulse
import math
import glob
import sys
import os

from fit_functions import *


if __name__ == '__main__':

    num_gwidth = 200

    intrinsic_fwhm = np.linspace(0.5,300,num_gwidth) # microseconds

    widths_gaussian = intrinsic_fwhm / ((0.0021499/phase_bins) * 1e6 * (2.0*math.sqrt(2*math.log(2)))) #array of gaussian widths (phase bins)

    #gaussian parameters in phase bins and arbitrary intensity comparitive to data
    parameters = np.zeros((num_gwidth, 3))
    parameters[:,0] = 0.3619 #general amplitude to be scaled
    parameters[:,1] = (1025.0/2048)*phase_bins #general mean
    parameters[:,2] = widths_gaussian #independent variable


    #===============================================================================
    # CONVOLUTION WITH GAUSSIAN INTRINSIC
    # ==============================================================================


    #===============================================================================
    # Beta Varying PBFs
    # ==============================================================================


    beta_pbf_data = np.load('beta_widths_pbf_data.npz')
    beta_tau_values = beta_pbf_data['taus_mus']
    betas = beta_pbf_data['betas']
    beta_pbfs = beta_pbf_data['pbfs_unitheight']

    # first want to scale the time to match the phase bins
    #this way we have 9549 values and they go up to however many phase bins
    times_scaled = np.zeros(cordes_phase_bins)
    for i in range(cordes_phase_bins):
        times_scaled[i] = phase_bins/cordes_phase_bins*i

    #an array of the broadening functions scaled to number of phase bins data values
    beta_pbf_data_freqscale = np.zeros((np.size(betas), np.size(beta_tau_values), phase_bins))

    data_index1 = 0
    for i in beta_pbfs:
        data_index2 = 0
        timetofreq_pbfdata = np.zeros((np.size(beta_tau_values), phase_bins))
        for ii in i:
            interpolate_first_beta = CubicSpline(times_scaled, ii)
            pbfdata_freqs = interpolate_first_beta(np.arange(0,phase_bins,1))
            timetofreq_pbfdata[data_index2] = pbfdata_freqs
            data_index2 = data_index2+1
        beta_pbf_data_freqscale[data_index1] = timetofreq_pbfdata

        data_index1 = data_index1+1

    del(beta_pbfs)

    #next we want to convert the broadening functions to unit area for convolution
    beta_pbf_data_unitarea = np.zeros((np.size(betas), np.size(beta_tau_values), phase_bins))

    data_index1 = 0
    for i in beta_pbf_data_freqscale:
        data_index2 = 0
        for ii in i:
            tsum = trapz(ii)
            beta_pbf_data_unitarea[data_index1][data_index2] = ii/tsum
            data_index2 = data_index2+1
        data_index1 = data_index1+1

    del(beta_pbf_data_freqscale)

    np.savez(f'beta_pbf_data_unitarea|PHASEBINS={phase_bins}', betas = betas, taus_mus = beta_tau_values, pbfs_unitarea = beta_pbf_data_unitarea)

    #now convolve the pbfs with varying gaussians for the final bank of profiles to fit
    beta_convolved_profiles = np.memmap('beta_convolved_profiles_gaussian', dtype='float64', mode='w+', shape=(np.size(betas), np.size(beta_tau_values), np.size(widths_gaussian), phase_bins))

    #indicies of beta, template width, gaussian width, profile data

    data_index0 = 0
    for i in beta_pbf_data_unitarea:
        data_index1 = 0
        for ii in i:
            data_index2 = 0
            for iii in parameters:
                p = iii
                ua_intrinsic_gauss = (p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))\
                / trapz(p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))
                new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
                new_profile = new_profile.real #take real component of convolution
                beta_convolved_profiles[data_index0][data_index1][data_index2] = new_profile
                data_index2 = data_index2+1
            data_index1 = data_index1+1
        data_index0 = data_index0+1

    beta_convolved_profiles.flush()

    #===============================================================================
    # Zeta Varying PBFs
    # ==============================================================================

    #import profiles from professor Cordes
    zeta_pbf_data = np.load('zeta_widths_pbf_data.npz')
    zeta_tau_values = zeta_pbf_data['taus_mus']
    zetas = zeta_pbf_data['zetas']
    zeta_pbfs = zeta_pbf_data['pbfs_unitheight']

    #an array of the broadening functions scaled to however many phase bins data values
    zeta_pbf_data_freqscale = np.zeros((np.size(zetas), np.size(zeta_tau_values), phase_bins))

    data_index1 = 0
    for i in zeta_pbfs:
        data_index2 = 0
        timetofreq_pbfdata = np.zeros((np.size(zeta_tau_values), phase_bins))
        for ii in i:
            interpolate_first_zeta = CubicSpline(times_scaled, ii)
            pbfdata_freqs = interpolate_first_zeta(np.arange(0,phase_bins,1))
            timetofreq_pbfdata[data_index2] = pbfdata_freqs
            data_index2 = data_index2+1
        zeta_pbf_data_freqscale[data_index1] = timetofreq_pbfdata
        data_index1 = data_index1+1

    del(zeta_pbfs)

    #next we want to convert the broadening functions to unit area for convolution
    zeta_pbf_data_unitarea = np.zeros((np.size(zetas), np.size(zeta_tau_values), phase_bins))

    data_index1 = 0
    for i in zeta_pbf_data_freqscale:
        data_index2 = 0
        for ii in i:
            tsum = trapz(ii)
            zeta_pbf_data_unitarea[data_index1][data_index2] = ii/tsum
            data_index2 = data_index2+1
        data_index1 = data_index1+1

    np.savez(f'zeta_pbf_data_unitarea|PHASEBINS={phase_bins}', zetas = zetas, taus_mus = zeta_tau_values, pbfs_unitarea = zeta_pbf_data_unitarea)

    del(zeta_pbf_data_freqscale)

    #now convolve the pbfs with varying gaussians for the final bank of profiles to fit
    zeta_convolved_profiles = np.memmap('zeta_convolved_profiles_gaussian', dtype='float64', mode='w+', shape=(np.size(zetas), np.size(zeta_tau_values), np.size(widths_gaussian), phase_bins))

    #indicies of beta, template width, gaussian width, profile data

    data_index0 = 0
    for i in zeta_pbf_data_unitarea:
        data_index1 = 0
        for ii in i:
            data_index2 = 0
            for iii in parameters:
                p = iii
                ua_intrinsic_gauss = (p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))\
                / trapz(p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))
                new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
                new_profile = new_profile.real #take real component of convolution
                zeta_convolved_profiles[data_index0][data_index1][data_index2] = new_profile
                data_index2 = data_index2+1
            data_index1 = data_index1+1
        data_index0 = data_index0+1

    zeta_convolved_profiles.flush()

    #===============================================================================
    # Exp PBFs
    #===============================================================================

    #an array of the broadening functions scaled to number of phase bins data values

    #import profiles from professor Cordes
    exp_pbf_data = np.load('exp_widths_pbf_data.npz')
    exp_tau_values = exp_pbf_data['taus_mus']
    exp_pbfs = exp_pbf_data['pbfs_unitheight']

    times_scaled = np.zeros(cordes_phase_bins)
    for i in range(cordes_phase_bins):
        times_scaled[i] = phase_bins/cordes_phase_bins*i

    exp_pbf_data_freqscale = np.zeros((np.size(exp_tau_values), phase_bins))

    data_index1 = 0
    for ii in exp_pbfs:
        interpolate_first_beta = CubicSpline(times_scaled, ii)
        pbfdata_freqs = interpolate_first_beta(np.arange(0,phase_bins,1))
        exp_pbf_data_freqscale[data_index1] = pbfdata_freqs
        data_index1 = data_index1+1

    del(exp_pbfs)

    #scale all profiles to unit area
    exp_data_unitarea = np.zeros((np.size(exp_tau_values), phase_bins))

    data_index2 = 0
    for ii in exp_pbf_data_freqscale:
        tsum = trapz(ii)
        exp_data_unitarea[data_index2] = ii/tsum
        data_index2 = data_index2+1

    del(exp_pbf_data_freqscale)

    np.savez(f'exp_pbf_data_unitarea|PHASEBINS={phase_bins}', taus_mus = exp_tau_values, pbfs_unitarea = exp_data_unitarea)

    #convolve the unit area broadening functions with varying gaussians
    exp_convolved_profiles = np.memmap('exp_convolved_profiles_gaussian', dtype='float64', mode='w+', shape=(np.size(exp_tau_values),np.size(widths_gaussian),phase_bins))

    data_index0 = 0
    for ii in exp_data_unitarea:
        data_index2 = 0
        for iii in parameters:
            p = iii
            ua_intrinsic_gauss = (p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2])))) / trapz(p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))
            new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
            new_profile = new_profile.real #take real component of convolution
            exp_convolved_profiles[data_index0][data_index2] = new_profile
            data_index2 = data_index2+1
        data_index0 = data_index0+1

    exp_convolved_profiles.flush()

    np.savez(f'convolved_profiles_intrinsic=gaussian|PHASEBINS={phase_bins}', intrinsic_fwhm_mus = intrinsic_fwhm, betas = betas, beta_taus_mus = beta_tau_values, beta_profiles = beta_convolved_profiles, zetas = zetas, zeta_taus_mus = zeta_tau_values, zeta_profiles = zeta_convolved_profiles, exp_taus_mus = exp_tau_values, exp_profiles = exp_convolved_profiles)

    del(beta_convolved_profiles)
    del(zeta_convolved_profiles)
    del(exp_convolved_profiles)

    os.remove('beta_convolved_profiles_gaussian')
    os.remove('zeta_convolved_profiles_gaussian')
    os.remove('exp_convolved_profiles_gaussian')

    #===============================================================================
    # CONVOLUTION WITH SBAND AVERAGE AS INTRINSIC
    # ==============================================================================


    j1903_intrins = np.load('j1903_high_freq_temp.npy')


    #===============================================================================
    # First rescale J1903 template to varying widths
    #===============================================================================

    times_scaled = np.zeros(np.size(j1903_intrins))
    for i in range(np.size(j1903_intrins)):
        times_scaled[i] = phase_bins/np.size(j1903_intrins)*i

    #an array of the broadening functions scaled to number of phase bins data values
    interpolate_first = CubicSpline(times_scaled, j1903_intrins)
    j1903_intrins = interpolate_first(np.arange(phase_bins))

    sp = SinglePulse(j1903_intrins)
    sband_fwhm = sp.getFWHM()*(j1903_period/phase_bins) #convert to microseconds

    intrins_stretch_factors = (intrinsic_fwhm/sband_fwhm)
    intrinsic_pulses = np.zeros((np.size(intrins_stretch_factors),phase_bins))

    data_index = 0
    for ii in intrins_stretch_factors:

        width_intrins_data = stretch_or_squeeze(j1903_intrins, ii)

        intrinsic_pulses[data_index] = width_intrins_data

        data_index += 1

    #===============================================================================
    # Now convolve with beta pbfs
    #===============================================================================

    b_convolved_w_dataintrins = np.memmap('beta_convolved_profiles_sband_avg', dtype='float64', mode='w+', shape=(np.size(betas), np.size(beta_tau_values), np.size(intrinsic_fwhm), phase_bins))

    data_index0 = 0
    for i in beta_pbf_data_unitarea:
        data_index1 = 0
        for ii in i:
            data_index2 = 0
            for s in intrinsic_pulses:
                ua_intrinsic = s/trapz(s)
                new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic)*np.fft.fft(ii)))
                new_profile = new_profile.real #take real component of convolution
                b_convolved_w_dataintrins[data_index0][data_index1][data_index2] = new_profile
                data_index2 = data_index2+1
            data_index1 = data_index1+1
        data_index0 = data_index0+1

    b_convolved_w_dataintrins.flush()
    del(beta_pbf_data_unitarea)

    #===============================================================================
    #Now convolve with zeta pbfs
    #===============================================================================

    z_convolved_w_dataintrins = np.memmap('zeta_convolved_profiles_sband_avg', dtype='float64', mode='w+', shape=(np.size(zetas), np.size(zeta_tau_values), np.size(intrinsic_fwhm), phase_bins))

    data_index0 = 0
    for i in zeta_pbf_data_unitarea:
        data_index1 = 0
        for ii in i:
            data_index2 = 0
            for s in intrinsic_pulses:
                ua_intrinsic = s/trapz(s)
                new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic)*np.fft.fft(ii)))
                new_profile = new_profile.real #take real component of convolution
                z_convolved_w_dataintrins[data_index0][data_index1][data_index2] = new_profile
                data_index2 = data_index2+1
            data_index1 = data_index1+1
        data_index0 = data_index0+1

    z_convolved_w_dataintrins.flush()

    del(zeta_pbf_data_unitarea)

    #===============================================================================
    #Now convolve with decaying exponential pbfs
    #===============================================================================

    e_convolved_w_dataintrins = np.memmap('exp_convolved_profiles_sband_avg', dtype='float64', mode='w+', shape=(np.size(exp_tau_values), np.size(intrinsic_fwhm), phase_bins))

    data_index0 = 0
    for ii in exp_data_unitarea:
        data_index1 = 0
        for s in intrinsic_pulses:
            ua_intrinsic = s/trapz(s)
            new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic)*np.fft.fft(ii)))
            new_profile = new_profile.real #take real component of convolution
            e_convolved_w_dataintrins[data_index0][data_index1] = new_profile
            data_index1 += 1
        data_index0 += 1

    e_convolved_w_dataintrins.flush()
    del(exp_data_unitarea)


    np.savez(f'convolved_profiles_intrinsic=sband_avg|PHASEBINS={phase_bins}', intrinsic_fwhm_mus = intrinsic_fwhm, betas = betas, beta_taus_mus = beta_tau_values, beta_profiles = b_convolved_w_dataintrins, zetas = zetas, zeta_taus_mus = zeta_tau_values, zeta_profiles = z_convolved_w_dataintrins, exp_taus_mus = exp_tau_values, exp_profiles = e_convolved_w_dataintrins)

    os.remove('beta_convolved_profiles_sband_avg')
    os.remove('zeta_convolved_profiles_sband_avg')
    os.remove('exp_convolved_profiles_sband_avg')
