#convolved_pbfs.py
#includes both exponential and pbf broadened profile templates
#Last edited by Abra Geiger 4/24/23

#cordes PBFs
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline

cordes_profs = np.load('widths_pbf_data.npy')

betaselect = np.array([3.1, 3.5, 3.667, 3.8, 3.9, 3.95, 3.975, 3.99, 3.995, 3.9975, 3.999, 3.99999])

widths = np.concatenate((np.linspace(0.1, 1.0, 10), np.linspace(1.1, 42.0, 40)))

widths_gaussian = np.linspace(0.1, 250.0, 50)

parameters = np.zeros((50, 3))
parameters[:,0] = 0.3619 #general amplitude to be scaled
parameters[:,1] = 1025.0 #general mean
parameters[:,2] = widths_gaussian #independent variable

#phase bins
t = np.linspace(0, 2048, 2048)

# first want to scale the time to match the phase bins
#this way we have 9549 values and they go up to 2048s
times_scaled = np.zeros(9549)
for i in range(9549):
    times_scaled[i] = 2048/9549*i

pbf_data_freqscale = np.zeros((np.size(betaselect), np.size(widths), 2048))

data_index1 = 0
for i in cordes_profs:
    data_index2 = 0
    timetofreq_pbfdata = np.zeros((np.size(widths), 2048))
    for ii in i:
        interpolate_first_beta = CubicSpline(times_scaled, ii)
        pbfdata_freqs = interpolate_first_beta(np.arange(0,2048,1))
        timetofreq_pbfdata[data_index2] = pbfdata_freqs
        data_index2 = data_index2+1
    pbf_data_freqscale[data_index1] = timetofreq_pbfdata
    data_index1 = data_index1+1

#next we want to convert the broadening functions to unit area for convolution
pbf_data_unitarea = np.zeros((np.size(betaselect), np.size(widths), 2048))

data_index1 = 0
for i in pbf_data_freqscale:
    data_index2 = 0
    for ii in i:
        tsum = trapz(ii)
        pbf_data_unitarea[data_index1][data_index2] = ii/tsum
        data_index2 = data_index2+1
    data_index1 = data_index1+1

#convolve the pbfs with varying gaussians for the final back of profiles to fit
convolved_profiles = np.zeros((np.size(betaselect), np.size(widths), np.size(parameters[:,0]), 2048))
#indicies of beta, template width, gaussian width, profile data

data_index0 = 0
for i in pbf_data_unitarea:
    data_index1 = 0
    for ii in i:
        data_index2 = 0
        for iii in parameters:
            p = iii
            ua_intrinsic_gauss = (p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2])))) / trapz(p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))
            new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
            new_profile = new_profile.real #take real component of convolution
            convolved_profiles[data_index0][data_index1][data_index2] = new_profile
            data_index2 = data_index2+1
        data_index1 = data_index1+1
    data_index0 = data_index0+1

time = np.arange(0,2048,1) * (0.0021499/2048) * 1e3 #milliseconds

np.save('convolved_profs', convolved_profiles)

#for i in range(10):
#    plt.plot(convolved_profiles[0][0][i*4])
#    plt.show()
