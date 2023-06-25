"""
Created April 2023
@author: Abra Geiger abrageiger7

Calculating Tau Values for Varying PBFs (varying in type and width)
"""

import convolved_pbfs as conv
import numpy as np
import matplotlib.pyplot as plt
import math

from fit_functions import *
from fitting_params import *

#tau values are the amount of time covered between the mode of the pbf and
#the mode/e value of the pbf

tau_values = {}

#===============================================================================
#Beta
#===============================================================================

#cordes beta pbfs
beta_cordes_profs = np.load('beta_widths_pbf_data.npy')

beta_tau_values = np.zeros((np.size(betaselect), np.size(widths)))

data_index = 0
for i in beta_cordes_profs:
    data_index2 = 0
    for ii in i:
        tau_ii = calculate_tau(ii)
        beta_tau_values[data_index][data_index2] = tau_ii[0]
        data_index2 = data_index2+1
    data_index = data_index+1

tau_values['beta'] = beta_tau_values

#===============================================================================
#Zeta
#===============================================================================

#cordes zeta pbfs
zeta_cordes_profs = np.load('zeta_widths_pbf_data.npy')

zeta_tau_values = np.zeros((np.size(zetaselect), np.size(widths)))

data_index = 0
for i in zeta_cordes_profs:
    data_index2 = 0
    for ii in i:
        tau_ii = calculate_tau(ii)
        zeta_tau_values[data_index][data_index2] = tau_ii[0]
        data_index2 = data_index2+1
    data_index = data_index+1

tau_values['zeta'] = zeta_tau_values

#===============================================================================
#Exponential
#===============================================================================

#long in time exp pbfs
widths_exp_array = np.load('exp_widths_pbf_data.npy')

exp_tau_values = np.zeros(np.size(widths))

data_index = 0
for i in widths_exp_array:
    exp_tau_values[data_index] = calculate_tau(i)[0]
    data_index = data_index+1

tau_values['exp'] = exp_tau_values

#===============================================================================

np.save('tau_values', tau_values)
