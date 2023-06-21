"""
Created Jun 2023
Last Edited on Mon May 22 2023
@author: Abra Geiger abrageiger7

Debugging and testing profile fitting
"""

#imports
import numpy as np
import matplotlib.pyplot as plt
import fit_functions as fittin
import convolved_pbfs as conv
#import intrinsic_pbfs as intrins
import math
from profile_class import Profile
import zeta_convolved_pbfs as zconv


#import the parameter bank for reference, comparing, and plotting
convolved_profiles = conv.convolved_profiles
widths = conv.widths
gauss_widths = conv.widths_gaussian
betaselect = conv.betaselect
time = conv.time
zetaselect = zconv.zetaselect

#import data
data = np.load("J1903_data.npy")
freq = np.load("J1903_freqs.npy")
mjds = np.load("J1903_mjds.npy")
chan = np.load("J1903_numchan.npy")
dur = np.load("J1903_dur.npy")

#===============================================================================
# Testing the intrinsic profile fitting
# Results - Intrins does a bit better - 501.5 versus 459.3
# ==============================================================================
# low_chig = 0
# low_chii = 0
#
# for i in range(10):
#
#     num_chan = int(chan[i*5])
#     datas = data[i*5][:num_chan]
#     freqs = freq[i*5][:num_chan]
#
#     p = Profile(mjds[i*5], datas, freqs, dur[i*5])
#
#     for ii in range(p.num_sub):
#
#         datafitb = p.fit(ii, beta_ind = 11, gwidth_ind = 2, intrins = True)
#         datafite = p.fit(ii, dec_exp = True, gwidth_pwr_law = True, intrins = True)
#         #datafitz = p.fit(ii, zind = 6, intrins = True)
#
#         low_chii += datafitb[0]
#         low_chii += datafite[0]
#
#         datafitb = p.fit(ii, beta_ind = 11, gwidth_ind = 2)
#         datafite = p.fit(ii, dec_exp = True, gwidth_pwr_law = True)
#
#         low_chig += datafitb[0]
#         low_chig += datafite[0]
#
# print(low_chig)
# print(low_chii)

#===============================================================================
# Testing the best fit beta gwidth for intrinsic s-band fitting
# ==============================================================================

best_fit_widths = np.zeros((10,7,6,2))

for i in range(10):

    num_chan = int(chan[i*5])
    datas = data[i*5][:num_chan]
    freqs = freq[i*5][:num_chan]

    p = Profile(mjds[i*5], datas, freqs, dur[i*5])

    for ii in range(p.num_sub//2):

        for iii in range(6):

            datafitb = p.fit(ii*2, beta_ind = 11, gwidth_ind = iii, intrins = True)

            best_fit_widths[i][ii][iii][0] = datafitb[0]
            best_fit_widths[i][ii][iii][1] = datafitb[5]

print(best_fit_widths)
np.save('test_of_best_beta_gwidth_intrins', best_fit_widths)



#===============================================================================
# Testing the gwidth power law
# ==============================================================================

# for i in range(10):
#
#     num_chan = int(chan[i*5])
#     datas = data[i*5][:num_chan]
#     freqs = freq[i*5][:num_chan]
#
#     p = Profile(mjds[i*5], datas, freqs, dur[i*5])
#     pwr_ind = p.fit(0, dec_exp = True, intrins = True)
#
#     print(f'Best Fit Gauss Width = {pwr_ind[3]}')
#
#     pwr_ind = p.fit_pwr_law_g(intrins = True)
#
#     print(pwr_ind)

#===============================================================================
# Collecting best fit gaussian widths for highest frequency pulse in order to
# set a reference frequency and gaussian width for the gwidth pwr law

# **Results** Average best fit gaussian width for 1742 (highest freq) was about
# 70 microseconds - starting the gwidth power laws from there now
# ==============================================================================

# ii = 0
#
# high_freq_gwidth_test = np.zeros((56,2))
#
# print(conv.gauss_fwhm)
#
# for i in range(56):
#
#     num_chan = int(chan[i])
#     datas = data[i][:num_chan]
#     freqs = freq[i][:num_chan]
#
#     p = Profile(mjds[i], datas, freqs, dur[i])
#
#     dataret = p.fit(ii, dec_exp=True)
#
#     high_freq_gwidth_test[i][0] = p.freq_suba #frequency
#     high_freq_gwidth_test[i][1] = dataret[1] #gaussian width
#
#     print(f'Frequency = {p.freq_round} MHz')
#     print(fr'Gaussian Width = {dataret[1]} \mu s')
#
# np.save('high_freq_gwidth_test', high_freq_gwidth_test)


#===============================================================================
# Printing lowest chi-squared values to get a sense of the best gaussian width
# ==============================================================================

# dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 3)
# print(dataeh[0])
#
# dataem = p.fit(iv, dec_exp=True, gwidth_ind = 3)
# print(dataem[0])
#
# datael = p.fit(iii, dec_exp=True, gwidth_ind = 3)
# print(datael[0])
#
# dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 4)
# print(dataeh[0])
#
# dataem = p.fit(iv, dec_exp=True, gwidth_ind = 4)
# print(dataem[0])
#
# datael = p.fit(iii, dec_exp=True, gwidth_ind = 4)
# print(datael[0])
#
# dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 5)
# print(dataeh[0])
#
# dataem = p.fit(iv, dec_exp=True, gwidth_ind = 5)
# print(dataem[0])
#
# datael = p.fit(iii, dec_exp=True, gwidth_ind = 5)
# print(datael[0])
#
# dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 6)
# print(dataeh[0])
#
# dataem = p.fit(iv, dec_exp=True, gwidth_ind = 6)
# print(dataem[0])
#
# datael = p.fit(iii, dec_exp=True, gwidth_ind = 6)
# print(datael[0])
#
# dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 7)
# print(dataeh[0])
#
# dataem = p.fit(iv, dec_exp=True, gwidth_ind = 7)
# print(dataem[0])
#
# datael = p.fit(iii, dec_exp=True, gwidth_ind = 7)
# print(datael[0])
#
# dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 8)
# print(dataeh[0])
#
# dataem = p.fit(iv, dec_exp=True, gwidth_ind = 8)
# print(dataem[0])
#
# datael = p.fit(iii, dec_exp=True, gwidth_ind = 8)
# print(datael[0])
#
# dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 9)
# print(dataeh[0])
#
# dataem = p.fit(iv, dec_exp=True, gwidth_ind = 9)
# print(dataem[0])
#
# datael = p.fit(iii, dec_exp=True, gwidth_ind = 9)
# print(datael[0])
#
# dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 10)
# print(dataeh[0])
#
# dataem = p.fit(iv, dec_exp=True, gwidth_ind = 10)
# print(dataem[0])
#
# datael = p.fit(iii, dec_exp=True, gwidth_ind = 10)
# print(datael[0])
#
# dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 11)
# print(dataeh[0])
#
# dataem = p.fit(iv, dec_exp=True, gwidth_ind = 11)
# print(dataem[0])
#
# datael = p.fit(iii, dec_exp=True, gwidth_ind = 11)
# print(datael[0])
#
# dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 12)
# print(dataeh[0])
#
# dataem = p.fit(iv, dec_exp=True, gwidth_ind = 12)
# print(dataem[0])
#
# datael = p.fit(iii, dec_exp=True, gwidth_ind = 12)
# print(datael[0])

#===============================================================================
# testing that no tau values are the same to confirm that grid spacing is resolved enough
# ==============================================================================

# import tau
# for i in range(len(betaselect)):
#     for ii in range(np.size(tau.tau_values[0])-1):
#         if tau.tau_values[i][ii] == tau.tau_values[i][ii+1]:
#             print('MORE RESOLUTION NECESSARY')
# for ii in range(np.size(tau.tau_values_exp)-1):
#     if tau.tau_values_exp[ii] == tau.tau_values_exp[ii+1]:
#         print('MORE RESOLUTION NECESSARY')
# for i in range(len(zetaselect)):
#     for ii in range(np.size(tau.zeta_tau_values[0])-1):
#         if tau.zeta_tau_values[i][ii] == tau.zeta_tau_values[i][ii+1]:
#             print('MORE RESOLUTION NECESSARY')
