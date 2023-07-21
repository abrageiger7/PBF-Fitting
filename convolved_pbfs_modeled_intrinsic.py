"""
Created June 2023
Last Edited on Mon May 22 2023

Calculate various fitting profiles from the J1903 intrinsic pulse shape from
average of all s-band
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
import pickle

from fit_functions import *

#===============================================================================
# First model the intrinsic shape over frequency
#===============================================================================



#===============================================================================
# Now convolved with beta pbfs
#===============================================================================

beta_pbf_data_unitarea = np.load(f'beta_pbf_data_unitarea_phasebins={phase_bins}.npy')

b_convolved_w_dataintrins = np.memmap('b_convolved_w_dataintrins', dtype='float64', mode='w+', shape=(np.size(betaselect), np.size(widths), np.size(gauss_fwhm), phase_bins))

data_index0 = 0
for i in beta_pbf_data_unitarea:
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

b_convolved_w_dataintrins.flush()
del(beta_pbf_data_unitarea)

#===============================================================================
#Now convolve with decaying exponential pbfs
#===============================================================================

exp_data_unitarea = np.load(f'exp_data_unitarea_phasebins={phase_bins}.npy')

e_convolved_w_dataintrins = np.memmap('e_convolved_w_dataintrins', dtype='float64', mode='w+', shape=(np.size(widths), np.size(gauss_fwhm), phase_bins))

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

e_convolved_w_dataintrins.flush()
del(exp_data_unitarea)

#===============================================================================
#Now convolve with zeta pbfs
#===============================================================================

zeta_pbf_data_unitarea = np.load(f'zeta_pbf_data_unitarea_phasebins={phase_bins}.npy')

z_convolved_w_dataintrins = np.memmap('z_convolved_w_dataintrins', dtype='float64', mode='w+', shape=(np.size(zetaselect), np.size(widths), np.size(gauss_fwhm), phase_bins))

data_index0 = 0
for i in zeta_pbf_data_unitarea:
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

z_convolved_w_dataintrins.flush()

del(zeta_pbf_data_unitarea)

#===============================================================================
#Now save dictionary of convolved profiles
#===============================================================================


sband_intrins_convolved_profiles = {}
sband_intrins_convolved_profiles['beta'] = b_convolved_w_dataintrins
sband_intrins_convolved_profiles['zeta'] = z_convolved_w_dataintrins
sband_intrins_convolved_profiles['exp'] = e_convolved_w_dataintrins

with open(f'sband_intrins_convolved_profiles_phasebins={phase_bins}.pkl', 'wb') as fp:
    pickle.dump(sband_intrins_convolved_profiles, fp)
