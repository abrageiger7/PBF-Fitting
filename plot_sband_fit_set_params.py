import pickle
import numpy as np
import matplotlib.pyplot as plt
from pypulse.singlepulse import SinglePulse
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline
import math
from pypulse.singlepulse import SinglePulse
from fit_functions import triple_gauss, convolve, stretch_or_squeeze, calculate_tau
import sys


'''Plots J1903+0327 Sband Average with Fitted Triple Gaussian intrinsic pulse
shape convolved with an extended medium pulse broadening function with zeta = 0
and beta = 11/3.'''

comp1 = [0.04, 0.42, 0.01]
comp2 = [1.0, 0.48, 0.04]
comp3 = [0.3, 0.56, 0.025]
tau = 11.2

print('Component 1 ' + str(comp1))
print('Component 2 ' + str(comp2))
print('Component 3 ' + str(comp3))

sband = np.load('j1903_high_freq_temp_unsmoothed.npy')
sband = sband / trapz(sband)
phase = np.linspace(0,1,np.size(sband))
t_phasebins = np.arange(np.size(sband))

#import the pbf and rescale it to have 2048 phasebins like the data
pbf = np.load(f'zeta_0_beta_11_3_pbf.npy')
cordes_phase_bins = np.size(pbf)
phase_bins = np.size(sband)
subs_time_avg = np.zeros(phase_bins)

for ii in range(np.size(subs_time_avg)):
        subs_time_avg[ii] = np.average(pbf[((cordes_phase_bins//phase_bins)*ii):((cordes_phase_bins//phase_bins)*(ii+1))])
subs_time_avg = subs_time_avg / trapz(subs_time_avg)

tau_subs_time_avg = calculate_tau(subs_time_avg)[0]

profile_fitted = convolve(triple_gauss(comp1, comp2, comp3, t_phasebins)[0], stretch_or_squeeze(subs_time_avg, tau/tau_subs_time_avg))

sp = SinglePulse(sband)
fitting = sp.fitPulse(profile_fitted)

sps = SinglePulse(profile_fitted*fitting[2])
fitted_template = sps.shiftit(fitting[1])

plt.figure(1)
plt.plot(phase, sband, color = 'lightsteelblue', lw = 2.8)
plt.plot(phase, fitted_template, color = 'midnightblue')
plt.xlabel('Pulse Phase')
plt.title(f'J1903 Modeled S-band 2200 [MHz]')
plt.show()
plt.close('all')
