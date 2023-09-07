import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline
import itertools
import pickle
from pypulse.singlepulse import SinglePulse
#import sys

from generic_2048_funcs import *

'''NOT UPDATED: Better version in toa_delay.ipynb notebooks file'''

#sys.path.insert(0, '/Users/abrageiger/Documents/research/pbf_fitting/')

print(phase_bins)
#gaussian convolved_profiles
with open(f'gauss_convolved_profiles_phasebins={phase_bins}.pkl', 'rb') as fp:
    gauss_convolved_profiles = pickle.load(fp)

#sband intrinsic convolved_profiles
with open(f'sband_intrins_convolved_profiles_phasebins={phase_bins}.pkl', 'rb') as fp:
    sband_intrins_convolved_profiles = pickle.load(fp)

#pbfs of unitarea and 2048 phase bins
with open(f'unitarea_pbfs_phasebins={phase_bins}.pkl', 'rb') as fp:
    unitarea_pbfs = pickle.load(fp)

beta_pbf_data_unitarea = unitarea_pbfs['beta']
zeta_pbf_data_unitarea = unitarea_pbfs['zeta']
exp_pbf_data_unitarea = unitarea_pbfs['exp']

#tau values corresponding to above templates in phase bins
with open(f'tau_values_phasebins={phase_bins}.pkl', 'rb') as fp:
    tau_values = pickle.load(fp)

beta_tau_values = tau_values['beta']
exp_tau_values = tau_values['exp']
zeta_tau_values = tau_values['zeta']


# ==============================================================================
# Case where fitting to profiles of beta = 3.99999 and varying tau values
# Fitting with decaying exponential pbf and freedom in tau and intrinsic gaussian width
# ==============================================================================

# Convolve with beta = 3.99999 pbfs

# beta_ind = 11
#
# gauss_param_ind = 40
#
# num_pbfwidths = 40
#
# profiles_fit_to = np.zeros((num_pbfwidths,2048))
#
# tau_correct = np.zeros(num_pbfwidths)
#
# for i in range(num_pbfwidths):
#
#     profiles_fit_to[i] = gauss_convolved_profiles['beta'][beta_ind,i*(num_pbfwidth//num_pbfwidths),gauss_param_ind]
#     tau_correct[i] = beta_tau_values[beta_ind,i*(num_pbfwidth//num_pbfwidths)]
#     plt.plot(gauss_convolved_profiles['beta'][beta_ind,i*(num_pbfwidth//num_pbfwidths),gauss_param_ind])
#
# plt.show()
#
# # now fit each convolved with beta convolved
#
# tau_fitted = np.zeros(num_pbfwidths)
#
# toa_offset = np.zeros(num_pbfwidths)
#
# gwidth_fitted = np.zeros(num_pbfwidths)
#
# for i in range(num_pbfwidths):
#
#     results = find_best_template_3D(profiles_fit_to[i],gauss_convolved_profiles['exp'][:,:])
#
#     tau_fitted[i] = exp_tau_values[results[0][0][0]]
#
#     toa_offset[i] = results[1]
#
#     gwidth_fitted[i] = parameters[results[0][1][0],2] * (2.0*math.sqrt(2*math.log(2)))
#
# plt.title(f'Fitting with Decaying Exponential when Beta = {betaselect[beta_ind]} is correct.')
# plt.plot(tau_correct, tau_fitted, color = 'k')
# plt.plot(tau_correct, tau_correct, ls = '--', color = 'grey')
# plt.xlabel(r'Correct $\tau$ [phasebins]')
# plt.ylabel(r'Fitted $\tau$ [phasebins]')
# plt.show()
#
# plt.title(f'Fitting with Decaying Exponential when Beta = {betaselect[beta_ind]} is correct.')
# plt.plot(tau_correct, toa_offset, color = 'k')
# plt.xlabel(r'Correct $\tau$ [phasebins]')
# plt.ylabel(r'TOA Delay [phasebins]')
# plt.show()
#
#
# plt.title(f'Fitting with Decaying Exponential when Beta = {betaselect[beta_ind]} is correct.')
# plt.plot(tau_correct, gwidth_fitted, color = 'k')
# plt.xlabel(r'Correct $\tau$ [phasebins]')
# plt.ylabel(r'Fitted Intrinsic Gaussian FWHM [phasebins]')
# plt.show()

# ==============================================================================
# Now case where vary tau and fit gaussian width (but in this case actually use
# three component gaussian as the intrinsic shape) and beta where beta is correct;
# this assumes know correct pbf shape
# ==============================================================================

# Convolve with beta = 3.99999 pbfs
# Has the same fwhm as gaussian of same index


beta_ind = 11

gauss_param_ind = 40

num_pbfwidths = 40

profiles_fit_to = np.zeros((num_pbfwidths,2048))

tau_correct = np.zeros(num_pbfwidths)

plt.figure(1)

for i in range(num_pbfwidths):

    profiles_fit_to[i] = sband_intrins_convolved_profiles['beta'][beta_ind,i*(num_pbfwidth//num_pbfwidths),gauss_param_ind]
    tau_correct[i] = beta_tau_values[beta_ind,i*(num_pbfwidth//num_pbfwidths)]
    plt.plot(sband_intrins_convolved_profiles['beta'][beta_ind,i*(num_pbfwidth//num_pbfwidths),gauss_param_ind])

plt.show()
plt.close('all')

results = find_best_template_3D(profiles_fit_to[10],gauss_convolved_profiles['beta'][beta_ind,::10,::20])

print(exp_tau_values[results[0][0][0]])

print(results[1])

print(parameters[results[0][1][0],2] * (2.0*math.sqrt(2*math.log(2))))


# now fit each convolved with beta convolved, assuming know gwidth

# tau_fitted = np.zeros(num_pbfwidths)
#
# toa_offset = np.zeros(num_pbfwidths)
#
# gwidth_fitted = np.zeros(num_pbfwidths)
#
# for i in range(num_pbfwidths):
#
#     results = find_best_template_3D(profiles_fit_to[i],gauss_convolved_profiles['beta'][beta_ind,:,:])
#
#     tau_fitted[i] = exp_tau_values[results[0][0][0]]
#
#     toa_offset[i] = results[1]
#
#     gwidth_fitted[i] = parameters[results[0][1][0],2] * (2.0*math.sqrt(2*math.log(2)))
#
# plt.title(f'Fitting with Single Gaussian and Beta = {betaselect[beta_ind]} when more complicated intrinsic pulse.')
# plt.plot(tau_correct, tau_fitted, color = 'k')
# plt.plot(tau_correct, tau_correct, ls = '--', color = 'grey')
# plt.xlabel(r'Correct $\tau$ [phasebins]')
# plt.ylabel(r'Fitted $\tau$ [phasebins]')
# plt.show()
#
# plt.title(f'Fitting with Single Gaussian and Beta = {betaselect[beta_ind]} when more complicated intrinsic pulse.')
# plt.plot(tau_correct, toa_offset, color = 'k')
# plt.xlabel(r'Correct $\tau$ [phasebins]')
# plt.ylabel(r'TOA Delay [phasebins]')
# plt.show()
#
# plt.title(f'Fitting with Single Gaussian and Beta = {betaselect[beta_ind]} when more complicated intrinsic pulse.')
# plt.plot(tau_correct, gwidth_fitted, color = 'k')
# plt.xlabel(r'Correct $\tau$ [phasebins]')
# plt.ylabel(r'Fitted Intrinsic Gaussian FWHM [phasebins]')
# plt.show()


# ==============================================================================
# Now same as above, but also fit with wrong pbf - exponential instead of beta
# ==============================================================================

# Convolve with beta = 3.99999 pbfs
# Has the same fwhm as gaussian of same index

# profiles_fit_to = np.zeros((num_pbfwidths,2048))
#
# tau_correct = np.zeros(num_pbfwidths)
#
# for i in range(num_pbfwidths):
#
#     profiles_fit_to[i] = sband_intrins_convolved_profiles['beta'][beta_ind,i*(num_pbfwidth//num_pbfwidths),gauss_param_ind]
#     tau_correct[i] = beta_tau_values[beta_ind,i*(num_pbfwidth//num_pbfwidths)]
#     plt.plot(sband_intrins_convolved_profiles['beta'][beta_ind,i*(num_pbfwidth//num_pbfwidths),gauss_param_ind])
#
# plt.show()
#
#
# # now fit each convolved with beta convolved, assuming know gwidth
#
# tau_fitted = np.zeros(num_pbfwidths)
#
# toa_offset = np.zeros(num_pbfwidths)
#
# gwidth_fitted = np.zeros(num_pbfwidths)
#
# for i in range(num_pbfwidths):
#
#     results = find_best_template(profiles_fit_to[i],gauss_convolved_profiles['exp'][beta_ind,:,gauss_param_ind])
#
#     results = find_best_template_3D(profiles_fit_to[i],gauss_convolved_profiles['exp'][beta_ind,::])
#
#     tau_fitted[i] = exp_tau_values[results[0][0][0]]
#
#     toa_offset[i] = results[1]
#
#     gwidth_fitted[i] = parameters[results[0][1][0],2] * (2.0*math.sqrt(2*math.log(2)))
#
# plt.title(f'Fitting with Single Gaussian and Exp when actually Beta = {betaselect[beta_ind]} and more complicated intrinsic pulse.')
# plt.plot(tau_correct, tau_fitted, color = 'k')
# plt.plot(tau_correct, tau_correct, ls = '--', color = 'grey')
# plt.xlabel(r'Correct $\tau$ [phasebins]')
# plt.ylabel(r'Fitted $\tau$ [phasebins]')
# plt.show()
#
# plt.title(f'Fitting with Single Gaussian and Exp when actually Beta = {betaselect[beta_ind]} and more complicated intrinsic pulse.')
# plt.plot(tau_correct, toa_offset, color = 'k')
# plt.xlabel(r'Correct $\tau$ [phasebins]')
# plt.ylabel(r'TOA Delay [phasebins]')
# plt.show()
#
# plt.title(f'Fitting with Single Gaussian and Exp when actually Beta = {betaselect[beta_ind]} and more complicated intrinsic pulse.')
# plt.plot(tau_correct, gwidth_fitted, color = 'k')
# plt.xlabel(r'Correct $\tau$ [phasebins]')
# plt.ylabel(r'Fitted Intrinsic Gaussian FWHM [phasebins]')
# plt.show()
