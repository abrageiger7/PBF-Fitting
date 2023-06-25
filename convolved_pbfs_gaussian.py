"""
Created April 2023
@author: Abra Geiger abrageiger7

Convolving PBFs (beta, zeta, and exponential) with varying gaussians
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline

from fitting_parameters import *

#===============================================================================
# Convolve with varying beta pbfs
#===============================================================================

#import profiles from Professor Cordes
beta_cordes_profs = np.load('beta_widths_pbf_data.npy')

# first want to scale the time to match the phase bins
#this way we have 9549 values and they go up to however many phase bins
times_scaled = np.zeros(cordes_phase_bins)
for i in range(cordes_phase_bins):
    times_scaled[i] = phase_bins/cordes_phase_bins*i

#an array of the broadening functions scaled to number of phase bins data values
beta_pbf_data_freqscale = np.zeros((np.size(betaselect), np.size(widths), phase_bins))

data_index1 = 0
for i in beta_cordes_profs:
    data_index2 = 0
    timetofreq_pbfdata = np.zeros((np.size(widths), phase_bins))
    for ii in i:
        interpolate_first_beta = CubicSpline(times_scaled, ii)
        pbfdata_freqs = interpolate_first_beta(np.arange(0,phase_bins,1))
        timetofreq_pbfdata[data_index2] = pbfdata_freqs
        data_index2 = data_index2+1
    beta_pbf_data_freqscale[data_index1] = timetofreq_pbfdata
    data_index1 = data_index1+1

#next we want to convert the broadening functions to unit area for convolution
beta_pbf_data_unitarea = np.zeros((np.size(betaselect), np.size(widths), phase_bins))

data_index1 = 0
for i in beta_pbf_data_freqscale:
    data_index2 = 0
    for ii in i:
        tsum = trapz(ii)
        beta_pbf_data_unitarea[data_index1][data_index2] = ii/tsum
        data_index2 = data_index2+1
    data_index1 = data_index1+1

#now convolve the pbfs with varying gaussians for the final bank of profiles to fit
beta_convolved_profiles = np.zeros((np.size(betaselect), np.size(widths), \
np.size(parameters[:,0]), phase_bins))
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

#===============================================================================
#Convolve with varying zeta pbfs
#===============================================================================

#import profiles from professor Cordes
zeta_cordes_profs = np.load('zeta_widths_pbf_data.npy')

#an array of the broadening functions scaled to however many phase bins data values
zeta_pbf_data_freqscale = np.zeros((np.size(zetaselect), np.size(widths), phase_bins))

data_index1 = 0
for i in zeta_cordes_profs:
    data_index2 = 0
    timetofreq_pbfdata = np.zeros((np.size(widths), phase_bins))
    for ii in i:
        interpolate_first_zeta = CubicSpline(times_scaled, ii)
        pbfdata_freqs = interpolate_first_zeta(np.arange(0,phase_bins,1))
        timetofreq_pbfdata[data_index2] = pbfdata_freqs
        data_index2 = data_index2+1
    zeta_pbf_data_freqscale[data_index1] = timetofreq_pbfdata
    data_index1 = data_index1+1

#next we want to convert the broadening functions to unit area for convolution
zeta_pbf_data_unitarea = np.zeros((np.size(zetaselect), np.size(widths), phase_bins))

data_index1 = 0
for i in zeta_pbf_data_freqscale:
    data_index2 = 0
    for ii in i:
        tsum = trapz(ii)
        zeta_pbf_data_unitarea[data_index1][data_index2] = ii/tsum
        data_index2 = data_index2+1
    data_index1 = data_index1+1

#now convolve the pbfs with varying gaussians for the final bank of profiles to fit
zeta_convolved_profiles = np.zeros((np.size(zetaselect), np.size(widths), \
np.size(parameters[:,0]), phase_bins))
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


#===============================================================================
#Now convolve with decaying exponential pbfs
#===============================================================================

#an array of the broadening functions scaled to number of phase bins data values

widths_exp_array = np.load('exp_widths_pbf_data.npy')

times_scaled = np.zeros(cordes_phase_bins)
for i in range(cordes_phase_bins):
    times_scaled[i] = phase_bins/cordes_phase_bins*i

exp_pbf_data_freqscale = np.zeros((np.size(widths), phase_bins))

data_index1 = 0
for ii in widths_exp_array:
    interpolate_first_beta = CubicSpline(times_scaled, ii)
    pbfdata_freqs = interpolate_first_beta(np.arange(0,phase_bins,1))
    exp_pbf_data_freqscale[data_index1] = pbfdata_freqs
    data_index1 = data_index1+1


#scale all profiles to unit area
exp_data_unitarea = np.zeros((np.size(widths), phase_bins))

data_index2 = 0
for ii in exp_pbf_data_freqscale:
    tsum = trapz(ii)
    exp_data_unitarea[data_index2] = ii/tsum
    data_index2 = data_index2+1

#convolve the unit area broadening functions with varying gaussians
exp_convolved_profiles = np.zeros((np.size(widths),np.size(parameters[:,2]),phase_bins))
t = np.linspace(0, phase_bins, phase_bins)

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


#===============================================================================
#Now save dictionary of convolved profiles
#===============================================================================

gauss_convolved_profiles = {}
gauss_convolved_profiles['beta'] = beta_convolved_profiles
gauss_convolved_profiles['zeta'] = zeta_convolved_profiles
gauss_convolved_profiles['exp'] = exp_convolved_profiles

np.save('gauss_convolved_profiles', gauss_convolved_profiles)
