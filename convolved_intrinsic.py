"""
Created June 2023
Last Edited on Mon May 22 2023
@author: Abra Geiger abrageiger7

Calculate various fitting profiles from the J1903 intrinsic pulse shape
"""

from pypulse.archive import Archive
from pypulse.singlepulse import SinglePulse
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import trapz
#from scipy.interpolate import CubicSpline
from scipy import special
import itertools
import convolved_pbfs as conv
import convolved_exp as cexp
import zeta_convolved_pbfs as zconv

#imports
widths = conv.widths
betaselect = conv.betaselect
gauss_fwhm = conv.gauss_fwhm
parameters = conv.parameters
zetaselect = zconv.zetaselect
phase_bins = conv.phase_bins
t = conv.t
bpbf_data_unitarea = conv.pbf_data_unitarea
exp_data_unitarea = cexp.exp_data_unitarea
zpbf_data_unitarea = zconv.pbf_data_unitarea

j1903_intrins = np.load('j1903_high_freq_temp.npy')
sp = SinglePulse(j1903_intrins)
i_fwhm = sp.getFWHM()

b_convolved_w_dataintrins = np.zeros((np.size(betaselect), np.size(widths), phase_bins))

data_index0 = 0
for i in bpbf_data_unitarea:
    data_index1 = 0
    for ii in i:
        ua_intrinsic_gauss = j1903_intrins
        new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
        new_profile = new_profile.real #take real component of convolution
        b_convolved_w_dataintrins[data_index0][data_index1] = new_profile
        data_index1 = data_index1+1
    data_index0 = data_index0+1

e_convolved_w_dataintrins = np.zeros((np.size(widths), phase_bins))

data_index0 = 0
for ii in exp_data_unitarea:
    ua_intrinsic_gauss = j1903_intrins
    new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
    new_profile = new_profile.real #take real component of convolution
    e_convolved_w_dataintrins[data_index0] = new_profile
    data_index0 = data_index0+1

z_convolved_w_dataintrins = np.zeros((np.size(zetaselect), np.size(widths), phase_bins))

data_index0 = 0
for i in zpbf_data_unitarea:
    data_index1 = 0
    for ii in i:
        ua_intrinsic_gauss = j1903_intrins
        new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
        new_profile = new_profile.real #take real component of convolution
        z_convolved_w_dataintrins[data_index0][data_index1] = new_profile
        data_index1 = data_index1+1
    data_index0 = data_index0+1
