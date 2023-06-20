"""
Created June 2023
Last Edited on Mon May 22 2023
@author: Abra Geiger abrageiger7

Calculate various fitting profiles from the J1903 intrinsic pulse shape
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
import convolved_pbfs as conv
import convolved_exp as cexp
import zeta_convolved_pbfs as zconv

#imports
widths = conv.widths
betaselect = conv.betaselect
gauss_fwhm = conv.gauss_fwhm
parameters = conv.parameters
zetaselect = zconv.zetaselect
phase_bins = conv.phase_bins
t = conv.t
bpbf_data_unitarea = conv.pbf_data_unitarea
exp_data_unitarea = cexp.exp_data_unitarea
zpbf_data_unitarea = zconv.pbf_data_unitarea
gauss_fwhm = conv.gauss_fwhm

j1903_intrins = np.load('j1903_high_freq_temp.npy')

# first want to rescale the j1903 template
init_phase_bins = 2048
times_scaled = np.zeros(init_phase_bins)
for i in range(init_phase_bins):
    times_scaled[i] = phase_bins/init_phase_bins*i

#an array of the broadening functions scaled to number of phase bins data values
interpolate_first = CubicSpline(times_scaled, j1903_intrins)
intrinsdata = interpolate_first(np.arange(0,phase_bins,1))
j1903_intrins = intrinsdata

sp = SinglePulse(j1903_intrins)
i_fwhm = sp.getFWHM()

intrins_stretch_factors = gauss_fwhm/i_fwhm
intrinsic_pulses = np.zeros((np.size(intrins_stretch_factors),phase_bins))

data_index = 0
for ii in intrins_stretch_factors:

    if ii>1:
        times_adjusted = t*ii #1- widen the pulse
        #interpolate the pulse in its broadened state
        interpolate_width = CubicSpline(times_adjusted, i) #2- interpolate to get section of the pulse (extrapolate = True?
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
        width_intrins_data[:phase_bins] = i
        times_scaled = np.zeros(int((1/ii)*phase_bins))
        #scale back to an array of size 9549
        for iv in range(int((1/ii)*phase_bins)):
            times_scaled[iv] = phase_bins/(int((1/ii)*phase_bins))*iv
        interpolate_less1 = CubicSpline(times_scaled, width_intrins_data)
        width_intrins_data = interpolate_less1(np.arange(phase_bins))

    #for width values of 1, no alteration necessary
    elif ii == 1:
        width_intrins_data = i

    intrinsic_pulses[data_index] = width_intrins_data

    data_index += 1




b_convolved_w_dataintrins = np.zeros((np.size(betaselect), np.size(widths), np.size(gauss_fwhm), phase_bins))

data_index0 = 0
for i in bpbf_data_unitarea:
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

np.save('beta_convolved_intris', b_convolved_w_dataintrins)


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

np.save('exp_convolved_intris', e_convolved_w_dataintrins)


z_convolved_w_dataintrins = np.zeros((np.size(zetaselect), np.size(widths), np.size(gauss_fwhm), phase_bins))

data_index0 = 0
for i in zpbf_data_unitarea:
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

np.save('zeta_convolved_intris', z_convolved_w_dataintrins)
