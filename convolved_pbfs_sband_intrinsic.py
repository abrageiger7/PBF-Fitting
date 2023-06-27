"""
Created June 2023
Last Edited on Mon May 22 2023

Calculate various fitting profiles from the J1903 intrinsic pulse shape from
average of all s-band
"""

from pypulse.archive import Archive
from pypulse.singlepulse import SinglePulse
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline
from scipy import special
import itertools
import pickle

import convolved_pbfs_gaussian as conv
from fit_functions import *

beta_pbf_data_unitarea = conv.beta_pbf_data_unitarea
exp_data_unitarea = conv.exp_data_unitarea
zeta_pbf_data_unitarea = conv.zeta_pbf_data_unitarea

j1903_intrins = np.load('j1903_high_freq_temp.npy') #2048//8 phase bins

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
i_fwhm = sp.getFWHM()*(j1903_period/phase_bins) #convert to microseconds

intrins_stretch_factors = (gauss_fwhm/i_fwhm)
intrinsic_pulses = np.zeros((np.size(intrins_stretch_factors),phase_bins))

data_index = 0
for ii in intrins_stretch_factors:

    if ii>1:
        times_adjusted = t*ii #1- widen the pulse
        #interpolate the pulse in its broadened state
        interpolate_width = CubicSpline(times_adjusted, j1903_intrins) #2- interpolate to get section of the pulse (extrapolate = True?
        #-> probably don't need because should only be interpolating)
        width_intrins_data = np.zeros(phase_bins)

        #add the intensity that loops around for stretched pulses
        index = 0
        #while(index<(np.max(times_adjusted))):
        while(index<(np.max(times_adjusted)-phase_bins)):
            interp_sect = interpolate_width(np.arange(index,index+phase_bins,1))
            width_intrins_data = np.add(width_intrins_data, interp_sect)
            #plt.plot(interp_sect)
            index = index+phase_bins

        final_interp_sect_array = np.arange(index, int(np.max(times_adjusted))+1, 1)
        final_interp_sect = interpolate_width(final_interp_sect_array)
        final_interp_sect = np.concatenate((final_interp_sect, np.zeros((index + phase_bins - int(np.max(times_adjusted)) - 1))))
        width_intrins_data = np.add(width_intrins_data, final_interp_sect)

    #squeeze narrowed pulses and add section of training zeros onto the end of them
    elif ii<1:
        #lengthen the array of the pulse so the pulse is comparatively narrow, adding zeros to the end
        width_intrins_data = np.zeros(int((1/ii)*phase_bins))
        width_intrins_data[:phase_bins] = j1903_intrins
        times_scaled = np.zeros(int((1/ii)*phase_bins))
        #scale back to an array of size 9549
        for iv in range(int((1/ii)*phase_bins)):
            times_scaled[iv] = phase_bins/(int((1/ii)*phase_bins))*iv
        interpolate_less1 = CubicSpline(times_scaled, width_intrins_data)
        width_intrins_data = interpolate_less1(np.arange(phase_bins))

    #for width values of 1, no alteration necessary
    elif ii == 1:
        width_intrins_data = j1903_intrins

    intrinsic_pulses[data_index] = width_intrins_data

    data_index += 1


intrinss_fwhm = np.zeros(np.size(intrins_stretch_factors))
ii = 0
for i in intrinsic_pulses:

    sp = SinglePulse(i)
    intrinss_fwhm[ii] = sp.getFWHM() * (j1903_period/np.size(i))
    ii += 1

plt.plot(intrinss_fwhm, label = 'intrinsic_s', alpha = 0.5)
plt.plot(gauss_fwhm, label = 'gaussian', alpha = 0.5)
plt.legend()
plt.show()

np.save('sband_intrins_fwhm', intrinss_fwhm)

#===============================================================================
# Now convolved with beta pbfs
#===============================================================================


b_convolved_w_dataintrins = np.zeros((np.size(betaselect), np.size(widths), np.size(gauss_fwhm), phase_bins))

data_index0 = 0
for i in beta_pbf_data_unitarea:
    data_index1 = 0
    for ii in i:
        data_index2 = 0
        for s in intrinsic_pulses:
            ua_intrinsic_gauss = s/trapz(s)
            new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
            new_profile = new_profile.real #take real component of convolution
            b_convolved_w_dataintrins[data_index0][data_index1][data_index2] = new_profile
            data_index2 = data_index2+1
        data_index1 = data_index1+1
    data_index0 = data_index0+1

#===============================================================================
#Now convolve with decaying exponential pbfs
#===============================================================================


e_convolved_w_dataintrins = np.zeros((np.size(widths), np.size(gauss_fwhm), phase_bins))

data_index0 = 0
for ii in exp_data_unitarea:
    data_index1 = 0
    for s in intrinsic_pulses:
        ua_intrinsic_gauss = s/trapz(s)
        new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
        new_profile = new_profile.real #take real component of convolution
        e_convolved_w_dataintrins[data_index0][data_index1] = new_profile
        data_index1 += 1
    data_index0 += 1

#===============================================================================
#Now convolve with zeta pbfs
#===============================================================================


z_convolved_w_dataintrins = np.zeros((np.size(zetaselect), np.size(widths), np.size(gauss_fwhm), phase_bins))

data_index0 = 0
for i in zeta_pbf_data_unitarea:
    data_index1 = 0
    for ii in i:
        data_index2 = 0
        for s in intrinsic_pulses:
            ua_intrinsic_gauss = s/trapz(s)
            new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
            new_profile = new_profile.real #take real component of convolution
            z_convolved_w_dataintrins[data_index0][data_index1][data_index2] = new_profile
            data_index2 = data_index2+1
        data_index1 = data_index1+1
    data_index0 = data_index0+1

#===============================================================================
#Now save dictionary of convolved profiles
#===============================================================================


sband_intrins_convolved_profiles = {}
sband_intrins_convolved_profiles['beta'] = b_convolved_w_dataintrins
sband_intrins_convolved_profiles['zeta'] = z_convolved_w_dataintrins
sband_intrins_convolved_profiles['exp'] = e_convolved_w_dataintrins

with open(f'sband_intrins_convolved_profiles_phasebins={phase_bins}.pkl', 'wb') as fp:
    pickle.dump(sband_intrins_convolved_profiles, fp)
