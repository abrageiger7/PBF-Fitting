"""
Created June 2023
@author: Abra Geiger abrageiger7

Calculate the time delay due to PBf for many different types of PBFs

3 Overall PBF Cases - Beta, Zeta, DecE
Each with different stretch factors and corresponding taus
3 Overall Intrinsic Pulse Cases - Gaussian, Double Gaussian, Nanograv Templates
"""

from pypulse.singlepulse import SinglePulse
from pypulse.singlepulse import SinglePulse
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline
from scipy import special
import itertools


#intrinsic pulse shapes - guassians
ii = 0
intrinsic_gaussians = np.zeros((20,phase_bins))
for i in parameters:
    p = i
    ua_intrinsic_gauss = (p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))\
    / trapz(p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))
    intrinsic_gaussians[ii] = ua_intrinsic_gauss
    ii+=1

#intrinsic pulse shapes - double gaussians
num_opts = 8
doubleg_amp = np.linspace(0,.32,num_opts)
doubleg_mean = np.linspace((200/2048)*phase_bins,(1500.0/2048)*phase_bins,num_opts)
doubleg_widths = np.linspace(0,(150/2048)*phase_bins,num_opts)

v = 0
intrinsic_gaussians_dg = np.zeros((20,num_opts,num_opts,num_opts,phase_bins))
for i in parameters:
    p = i
    for ii in np.arange(num_opts):
        for iii in np.arange(num_opts):
            for iv in np.arange(num_opts):

                ii1 = doubleg_amp[ii]
                iii1 = doubleg_mean[iii]
                iv1 = doubleg_widths[iv]
                double_gauss = (p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2])))) \
                + (ii1*np.exp((-1.0/2.0)*(((t-iii1)/iv1)*((t-iii1)/iv1))))
                ua_double_gauss = double_gauss/trapz(double_gauss)
                intrinsic_gaussians_dg[v][ii][iii][iv] = ua_double_gauss

    v+=1

toa_delays = np.zeros((np.size(betaselect), np.size(widths[::20])))
#for varying beta and pbf widths
for i in range(np.size(betaselect)):
    for ii in range(np.size(widths[::20])):
        template = bpbf_data_unitarea[i][ii]
        tau = tau.tau_values[i][ii]

        intrinsic = intrinsic_gaussians_dg[5][4][3][4]

        conv_temp = np.fft.ifft(np.fft.fft(template)*np.fft.fft(intrinsic))

        #Calculates mode of data profile to shift template to
        x = np.max(template)
        xind = np.where(template == x)[0][0]

        profile = intrinsic / np.max(intrinsic) #fitPulse requires template height of one
        z = np.max(profile)
        zind = np.where(profile == z)[0][0]
        ind_diff = xind-zind
        profile = np.roll(profile, ind_diff)
        sp = SinglePulse(conv_temp)
        fitting = sp.fitPulse(profile, rms_baseline = 0.1) #TOA cross-correlation, TOA template
        #matching, scale factor, TOA error, scale factor error, signal to noise
        #ratio, cross-correlation coefficient
        #based on the fitPulse fitting, scale and shift the profile to best fit
        #the inputted data
        #fitPulse figures out the best amplitude itself
        spt = SinglePulse(profile*fitting[2])
        fitted_template = spt.shiftit(fitting[1])

        max1 = np.where((fitted_template == np.max(fitted_template)))[0][0]
        max2 = np.where((conv_temp == np.max(conv_temp)))[0][0]
        toa_delays[i][ii] = max2-max1 #in phase bins

plt.imshow(toa_delays)
plt.title('Beta')
plt.colorbar()
plt.show()
print(toa_delays)


toa_delays = np.zeros((np.size(zetaselect), np.size(widths[::20])))
#for varying zeta and pbf widths
for i in range(np.size(zetaselect)):
    for ii in range(np.size(widths[::20])):
        template = zpbf_data_unitarea[i][ii]
        tau = tau.zeta_tau_values[i][ii]

        intrinsic = intrinsic_gaussians_dg[5][4][3][4]

        conv_temp = np.fft.ifft(np.fft.fft(template)*np.fft.fft(intrinsic))

        #Calculates mode of data profile to shift template to
        x = np.max(template)
        xind = np.where(template == x)[0][0]

        profile = intrinsic / np.max(intrinsic) #fitPulse requires template height of one
        z = np.max(profile)
        zind = np.where(profile == z)[0][0]
        ind_diff = xind-zind
        profile = np.roll(profile, ind_diff)
        sp = SinglePulse(conv_temp)
        fitting = sp.fitPulse(profile, rms_baseline = 0.1) #TOA cross-correlation, TOA template
        #matching, scale factor, TOA error, scale factor error, signal to noise
        #ratio, cross-correlation coefficient
        #based on the fitPulse fitting, scale and shift the profile to best fit
        #the inputted data
        #fitPulse figures out the best amplitude itself
        spt = SinglePulse(profile*fitting[2])
        fitted_template = spt.shiftit(fitting[1])

        max1 = np.where((fitted_template == np.max(fitted_template)))[0][0]
        max2 = np.where((conv_temp == np.max(conv_temp)))[0][0]
        toa_delays[i][ii] = max2-max1 #in phase bins

plt.imshow(toa_delays)
plt.title('Zeta')
plt.colorbar()
plt.show()
print(toa_delays)


toa_delays = np.zeros(np.size(widths[::20]))
#for exp and varying pbf widths
for ii in range(np.size(widths[::20])):
    template = exp_data_unitarea[i][ii]
    tau = tau.tau_values_exp[i][ii]

    intrinsic = intrinsic_gaussians_dg[5][4][3][4]

    conv_temp = np.fft.ifft(np.fft.fft(template)*np.fft.fft(intrinsic))

    #Calculates mode of data profile to shift template to
    x = np.max(template)
    xind = np.where(template == x)[0][0]

    profile = intrinsic / np.max(intrinsic) #fitPulse requires template height of one
    z = np.max(profile)
    zind = np.where(profile == z)[0][0]
    ind_diff = xind-zind
    profile = np.roll(profile, ind_diff)
    sp = SinglePulse(conv_temp)
    fitting = sp.fitPulse(profile, rms_baseline = 0.1) #TOA cross-correlation, TOA template
    #matching, scale factor, TOA error, scale factor error, signal to noise
    #ratio, cross-correlation coefficient
    #based on the fitPulse fitting, scale and shift the profile to best fit
    #the inputted data
    #fitPulse figures out the best amplitude itself
    spt = SinglePulse(profile*fitting[2])
    fitted_template = spt.shiftit(fitting[1])

    max1 = np.where((fitted_template == np.max(fitted_template)))[0][0]
    max2 = np.where((conv_temp == np.max(conv_temp)))[0][0]
    toa_delays[ii] = max2-max1 #in phase bins

plt.plot(toa_delays)
plt.title('Exp')
plt.colorbar()
plt.show()
print(toa_delays)
