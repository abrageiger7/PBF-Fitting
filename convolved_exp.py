"""
Created April 2023
Last Edited on Mon May 22 2023
@author: Abra Geiger abrageiger7

Comparing PBFs with decaying exponential
"""


#imports
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline
import convolved_pbfs as conv

#import because same pbf width stretch factors and gaussian widths as convolved pbfs
widths = conv.widths
gauss_widths = conv.widths_gaussian
parameters = conv.parameters

#create varying exponential profiles
widths_exp_array = np.zeros((np.size(widths), 2048))

phase_bins = 2048
len_dec_exp_profile = 45

time_bins = np.linspace(0,len_dec_exp_profile,phase_bins)
exponential = np.exp(-time_bins)
i = exponential

#indexes for stretch width values
data_index2 = 0
for ii in widths:
    #adjust times to this width by multiplying the times by the stretch/squeeze value (the width)
    #for stretch greater than zero, the PBF will broaden
    #for stretch less than zero, the PBF will narrow

    if ii>1:
        #stretching braodening function
        
        times_adjusted = time_bins*ii #1 - widen the pulse
        #interpolate the pulse in its broadened state
        interpolate_width = CubicSpline(times_adjusted, i)
        #2- interpolate to get section of the pulse desired
        width_pbf_data = np.zeros(phase_bins)

        #add the intensity that loops around for stretched pulses
        index = 0
        while(index<(np.max(times_adjusted))-len_dec_exp_profile):
            interp_sect = interpolate_width(np.linspace(index,index+len_dec_exp_profile,phase_bins))
            width_pbf_data = np.add(width_pbf_data, interp_sect)
            index = index+len_dec_exp_profile

        final_interp_sect_array = np.arange(index, int(np.max(times_adjusted))+1, 1)
        final_interp_sect = interpolate_width(final_interp_sect_array)
        final_interp_sect = np.concatenate((final_interp_sect, np.zeros((phase_bins-np.size(final_interp_sect)))))
        width_pbf_data = np.add(width_pbf_data, final_interp_sect)

        #plt.xlabel('Phase Bins')
        #plt.ylabel('Arbitrary Pulse Intensity')
        #plt.title('Broadened Pulse with Stretch Factor of ' + str(ii) + ' Overlaying the Cut-off PBF')
        #plt.show()
        
    elif ii<1:
        #squeezing broadening function
        
        #lengthen the array of the pulse so the pulse is comparatively narrow, adding zeros to the end
        width_pbf_data = np.zeros(int((1/ii)*phase_bins))
        width_pbf_data[:phase_bins] = i
        times_scaled = np.zeros(int((1/ii)*phase_bins))
        #scale back to an array of size 2048
        for iv in range(int((1/ii)*phase_bins)):
            times_scaled[iv] = phase_bins/(int((1/ii)*phase_bins))*iv
        interpolate_less1 = CubicSpline(times_scaled, width_pbf_data)
        width_pbf_data = interpolate_less1(np.arange(phase_bins))

    
    elif ii == 1:
        #for width values of 1, no alteration necessary
        
        width_pbf_data = i

        
    #scale all profiles to unit height
    width_pbf_data = width_pbf_data/np.max(width_pbf_data)

    #plot broadening function
    #plt.xlabel('Phase Bins')
    #plt.ylabel('Arbitrary Pulse Intensity')
    #plt.title('Broadened Pulse with Stretch Factor of ' + str(ii))
    #plt.plot(np.arange(2048), np.roll(width_pbf_data, 5))
    #plt.show()

    #append broadening function to array
    widths_exp_array[data_index2] = width_pbf_data
    data_index2 = data_index2+1

#plt.figure(1)
#for i in range(10):
#    plt.plot(widths_exp_array[i*5])
#plt.show()

#scale all profiles to unit area
exp_data_unitarea = np.zeros((np.size(widths), phase_bins))

data_index2 = 0
for ii in widths_exp_array:
    tsum = trapz(ii)
    exp_data_unitarea[data_index2] = ii/tsum
    data_index2 = data_index2+1

#convolve the unit area broadening functions with varying gaussians
convolved_profiles_exp = np.zeros((np.size(widths),np.size(parameters[:,2]),phase_bins))
t = np.linspace(0, phase_bins, phase_bins)

data_index0 = 0
for ii in exp_data_unitarea:
    data_index2 = 0
    for iii in parameters:
        p = iii
        ua_intrinsic_gauss = (p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2])))) / trapz(p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))
        new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
        new_profile = new_profile.real #take real component of convolution
        convolved_profiles_exp[data_index0][data_index2] = new_profile
        data_index2 = data_index2+1
    data_index0 = data_index0+1

#plt.figure(2)
#for i in range(10):
#    plt.plot(convolved_profiles_exp[i*5][i*5])
#    plt.show()
