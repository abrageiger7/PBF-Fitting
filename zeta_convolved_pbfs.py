"""
Created June 2023
Last Edited on Tues Jun 13 2023
@author: Abra Geiger abrageiger7

Convolving Zeta-varying PBFs with varying gaussians
"""


#imports
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline
import convolved_pbfs as conv

#import profiles from professor Cordes
cordes_profs = np.load('zeta_widths_pbf_data.npy')

#array of zeta values used
zetaselect = np.array([0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0])

#array of widths used (pbf stretch factors)
widths = conv.widths

#array of gaussian widths (phase bins)
widths_gaussian = conv.widths_gaussian
#gauss widths converted to fwhm microseconds
gauss_fwhm = conv.gauss_fwhm

parameters = conv.parameters

#phase bins
phase_bins = conv.phase_bins
t = conv.t

# first want to scale the time to match the phase bins
#this way we have 9549 values and they go up to however many phase bins
times_scaled = conv.times_scaled

#an array of the broadening functions scaled to however many phase bins data values
pbf_data_freqscale = np.zeros((np.size(zetaselect), np.size(widths), phase_bins))

data_index1 = 0
for i in cordes_profs:
    data_index2 = 0
    timetofreq_pbfdata = np.zeros((np.size(widths), phase_bins))
    for ii in i:
        interpolate_first_zeta = CubicSpline(times_scaled, ii)
        pbfdata_freqs = interpolate_first_zeta(np.arange(0,phase_bins,1))
        timetofreq_pbfdata[data_index2] = pbfdata_freqs
        data_index2 = data_index2+1
    pbf_data_freqscale[data_index1] = timetofreq_pbfdata
    data_index1 = data_index1+1

#next we want to convert the broadening functions to unit area for convolution
pbf_data_unitarea = np.zeros((np.size(zetaselect), np.size(widths), phase_bins))

data_index1 = 0
for i in pbf_data_freqscale:
    data_index2 = 0
    for ii in i:
        tsum = trapz(ii)
        pbf_data_unitarea[data_index1][data_index2] = ii/tsum
        data_index2 = data_index2+1
    data_index1 = data_index1+1

#now convolve the pbfs with varying gaussians for the final bank of profiles to fit
convolved_profiles = np.zeros((np.size(zetaselect), np.size(widths), \
np.size(parameters[:,0]), phase_bins))
#indicies of beta, template width, gaussian width, profile data

data_index0 = 0
for i in pbf_data_unitarea:
    data_index1 = 0
    for ii in i:
        data_index2 = 0
        for iii in parameters:
            p = iii
            ua_intrinsic_gauss = (p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))\
            / trapz(p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))
            new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
            new_profile = new_profile.real #take real component of convolution
            convolved_profiles[data_index0][data_index1][data_index2] = new_profile
            data_index2 = data_index2+1
        data_index1 = data_index1+1
    data_index0 = data_index0+1

time = conv.time #milliseconds

np.save('zeta_convolved_profs', convolved_profiles)

#for i in range(10):
#    plt.plot(convolved_profiles[0][i*14][0])
#    plt.show()
