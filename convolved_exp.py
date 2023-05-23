#convoled_exp.py

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline
import convolved_pbfs as conv

widths = conv.widths
gauss_widths = conv.widths_gaussian
betaselect = conv.betaselect
parameters = conv.parameters

#comparing PBFs with decaying exponential

#create varying exponential profiles
widths_exp_array = np.zeros((np.size(widths), 2048))

time_bins = np.linspace(0,45,2048)
exponential = np.exp(-time_bins)


i = exponential
#indexes for stretch width values
data_index2 = 0
for ii in widths:
    #adjust times to this width by multiplying the times by the stretch/squeeze value (the width)
    #for stretch greater than zero, the PBF will broaden
    #for stretch less than zero, the PBF will narrow

    if ii>1:
        times_adjusted = time_bins*ii #1- widen the pulse
        #interpolate the pulse in its broadened state
        interpolate_width = CubicSpline(times_adjusted, i)
        #2- interpolate to get section of the pulse desired
        width_pbf_data = np.zeros(2048)

        #add the intensity that loops around for stretched pulses
        index = 0
        while(index<(np.max(times_adjusted))-45):
            interp_sect = interpolate_width(np.linspace(index,index+45,2048))
            width_pbf_data = np.add(width_pbf_data, interp_sect)
            index = index+45

        final_interp_sect_array = np.arange(index, int(np.max(times_adjusted))+1, 1)
        final_interp_sect = interpolate_width(final_interp_sect_array)
        final_interp_sect = np.concatenate((final_interp_sect, np.zeros((2048-np.size(final_interp_sect)))))
        width_pbf_data = np.add(width_pbf_data, final_interp_sect)

        #plt.xlabel('Phase Bins')
        #plt.ylabel('Arbitrary Pulse Intensity')
        #plt.title('Broadened Pulse with Stretch Factor of ' + str(ii) + ' Overlaying the Cut-off PBF')
        #plt.show()

    #squeeze narrowed pulses and add section of training zeros onto the end of them
    elif ii<1:
        #lengthen the array of the pulse so the pulse is comparatively narrow, adding zeros to the end
        width_pbf_data = np.zeros(int((1/ii)*2048))
        width_pbf_data[:2048] = i
        times_scaled = np.zeros(int((1/ii)*2048))
        #scale back to an array of size 2048
        for iv in range(int((1/ii)*2048)):
            times_scaled[iv] = 2048/(int((1/ii)*2048))*iv
        interpolate_less1 = CubicSpline(times_scaled, width_pbf_data)
        width_pbf_data = interpolate_less1(np.arange(2048))

    #for width values of 1, no alteration necessary
    elif ii == 1:
        width_pbf_data = i

    #scale all profiles to unit height
    width_pbf_data = width_pbf_data/np.max(width_pbf_data)

    #plot broadening function
    #plt.xlabel('Phase Bins')
    #plt.ylabel('Arbitrary Pulse Intensity')
    #plt.title('Broadened Pulse with Stretch Factor of ' + str(ii))
    #plt.plot(np.arange(2048), np.roll(width_pbf_data, 5))
    #plt.show()

    widths_exp_array[data_index2] = width_pbf_data
    data_index2 = data_index2+1

#plt.figure(1)
#for i in range(10):
#    plt.plot(widths_exp_array[i*5])
#plt.show()

exp_data_unitarea = np.zeros((np.size(widths), 2048))

data_index2 = 0
for ii in widths_exp_array:
    tsum = trapz(ii)
    exp_data_unitarea[data_index2] = ii/tsum
    data_index2 = data_index2+1

convolved_profiles_exp = np.zeros((np.size(widths),np.size(parameters[:,2]),2048))
t = np.linspace(0, 2048, 2048)

data_index0 = 0
for ii in exp_data_unitarea:
    data_index2 = 0
    for iii in parameters:
        p = iii
        ua_intrinsic_gauss = (p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2])))) / trapz(p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))
        new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
        new_profile = new_profile.real #take real component of convolution
        #new_profile = np.convolve(ii, ua_intrinsic_gauss) #or could do this - np.roll(ii, 122) - to match up centers - no, Dr. Lam says keep broadening function starting at zero
        convolved_profiles_exp[data_index0][data_index2] = new_profile
        data_index2 = data_index2+1
    data_index0 = data_index0+1

plt.figure(2)
for i in range(10):
    plt.plot(convolved_profiles_exp[i*5][i*5])
    plt.show()
