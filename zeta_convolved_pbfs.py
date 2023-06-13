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

#import profiles from professor Cordes
cordes_profs = np.load('zeta_widths_pbf_data.npy')

#array of zeta values used
zetaselect = np.array([0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0])

#array of widths used (pbf stretch factors)
widths = np.concatenate((np.linspace(0.1, 1.0, 40), np.linspace(1.1, 42.0, 160)))

#array of gaussian widths (phase bins)
widths_gaussian = np.linspace(0.1, 250.0, 50)
#gauss widths converted to fwhm microseconds
gauss_fwhm = widths_gaussian * ((0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))

#gaussian parameters in phase bins and arbitrary intensity comparitive to data
parameters = np.zeros((50, 3))
parameters[:,0] = 0.3619 #general amplitude to be scaled
parameters[:,1] = 1025.0 #general mean
parameters[:,2] = widths_gaussian #independent variable

#phase bins
phase_bins = 2048
t = np.linspace(0, phase_bins, phase_bins)

# first want to scale the time to match the phase bins
#this way we have 9549 values and they go up to 2048s
cordes_phase_bins = 9549
times_scaled = np.zeros(cordes_phase_bins)
for i in range(cordes_phase_bins):
    times_scaled[i] = phase_bins/cordes_phase_bins*i

#an array of the broadening functions scaled to 2048 data values
pbf_data_freqscale = np.zeros((np.size(zetaselect), np.size(widths), 2048))

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

sec_pulse_per = 0.0021499
s_to_ms_conv = 1e3
time = np.arange(0,phase_bins,1) * (sec_pulse_per/phase_bins) * s_to_ms_conv #milliseconds

np.save('zeta_convolved_profs', convolved_profiles)

#for i in range(10):
#    plt.plot(convolved_profiles[0][i*14][0])
#    plt.show()