import numpy as np
import matplotlib.pyplot as plt
import math
from fit_functions import calculate_tau, stretch_or_squeeze, init_data_phase_bins, phase_bins, find_nearest, single_pbf_thin_screen
from scipy.interpolate import CubicSpline
from scipy.integrate import trapz
import os

phase_bins = init_data_phase_bins
print(phase_bins)

# Cordes thin screen pbfs
# CAREFUL: time steps spaced logarithmically
pbf_data = np.load('generate_pbf_sets_Nbeta_12_Nli_10.npz', allow_pickle=True)
zetas = pbf_data['livec']
betas = pbf_data['betavec']

print(zetas)
print(betas)

num_pbfwidth = 400

# for each pbf, calculate a number of tau values (microseconds)
tau_values_set = np.linspace(0.1,500,num_pbfwidth)

# parameters to collect for each pbf
tau_values_collect = np.zeros((np.size(betas), np.size(zetas), num_pbfwidth))
thin_screen_pbfs = np.zeros((np.size(betas), np.size(zetas), num_pbfwidth, phase_bins))

for i in range(np.size(betas)): #beta
     for ii in range(np.size(zetas)): #zeta

        beta = betas[i]
        zeta = zetas[ii]

        if (beta!=3.667 and zeta==0.01) or (beta==3.667 and zeta!=0.01) or (beta==3.667 and zeta==0.01):

            print(f'Beta = {beta} at index {i}')
            print(f'Zeta = {zeta} at index {ii}')

            for iii in range(np.size(tau_values_set)):

                print(iii)

                results = single_pbf_thin_screen(betas[i], zetas[ii], tau_values_set[iii], pbf_data)

                thin_screen_pbfs[i][ii][iii] = results[0]
                tau_values_collect[i][ii][iii] = results[1]

np.savez(f'thin_screen_pbfs|PHASEBINS={phase_bins}', pbfs_unitheight = thin_screen_pbfs, betas = betas, zetas = zetas, tau_mus = tau_values_collect)
