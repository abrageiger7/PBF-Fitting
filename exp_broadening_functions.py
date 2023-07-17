"""
Created June 2023
@author: Abra Geiger abrageiger7

Creates array of decaying exponential pbfs varying i pbf width (stretch and
squeeze factor)
"""

from fit_functions import *

import numpy as np
from scipy.interpolate import CubicSpline
import math

len_dec_exp_profile = 45

#create varying exponential profiles
widths_exp_array = np.zeros((np.size(widths), cordes_phase_bins))


time_bins = np.linspace(0,len_dec_exp_profile, cordes_phase_bins)
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
        width_pbf_data = np.zeros(cordes_phase_bins)

        #add the intensity that loops around for stretched pulses
        index = 0
        while(index<(np.max(times_adjusted))-len_dec_exp_profile):
            interp_sect = interpolate_width(np.linspace(index,index+len_dec_exp_profile,cordes_phase_bins))
            width_pbf_data = np.add(width_pbf_data, interp_sect)
            index = index+len_dec_exp_profile

        final_interp_sect_array = np.arange(index, int(np.max(times_adjusted))+1, 1)
        final_interp_sect = interpolate_width(final_interp_sect_array)
        final_interp_sect = np.concatenate((final_interp_sect, np.zeros((cordes_phase_bins-np.size(final_interp_sect)))))
        width_pbf_data = np.add(width_pbf_data, final_interp_sect)


    elif ii<1:
        #squeezing broadening function

        #lengthen the array of the pulse so the pulse is comparatively narrow, adding zeros to the end
        width_pbf_data = np.zeros(int((1/ii)*cordes_phase_bins))
        width_pbf_data[:cordes_phase_bins] = i
        times_scaled = np.zeros(int((1/ii)*cordes_phase_bins))
        #scale back to an array of size number of phase bins
        for iv in range(int((1/ii)*cordes_phase_bins)):
            times_scaled[iv] = cordes_phase_bins/(int((1/ii)*cordes_phase_bins))*iv
        interpolate_less1 = CubicSpline(times_scaled, width_pbf_data)
        width_pbf_data = interpolate_less1(np.arange(cordes_phase_bins))


    elif ii == 1:
        #for width values of 1, no alteration necessary

        width_pbf_data = i


    #scale all profiles to unit height
    width_pbf_data = width_pbf_data/np.max(width_pbf_data)

    #append broadening function to array
    widths_exp_array[data_index2] = width_pbf_data
    data_index2 = data_index2+1


np.save('exp_widths_pbf_data', widths_exp_array)
