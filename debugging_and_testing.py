"""
Created Jun 2023
@author: Abra Geiger abrageiger7

Debugging and testing profile fitting
"""

#imports
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

from profile_class_gaussian import Profile_Gauss as pcg
from profile_class_sband_intrinsic import Profile_Intrinss as pcs

#import data
with open('j1903_data.pkl', 'rb') as fp:
    data_dict = pickle.load(fp)

mjd_strings = list(data_dict.keys())

mjd0 = data_dict[mjd_strings[0]]['mjd']
data0 = data_dict[mjd_strings[0]]['data']
freqs0 = data_dict[mjd_strings[0]]['freqs']
dur0 = data_dict[mjd_strings[0]]['dur']

#===============================================================================
# Test of best fit widths for number of different mjds at lower freq
# ==============================================================================
# for i in range(len(mjd_strings)//5):
#
#     mjd0 = data_dict[mjd_strings[i*5]]['mjd']
#     data0 = data_dict[mjd_strings[i*5]]['data']
#     freqs0 = data_dict[mjd_strings[i*5]]['freqs']
#     dur0 = data_dict[mjd_strings[i*5]]['dur']
#
#     p = pcs(mjd0,data0,freqs0,dur0)
#     p.fit(p.num_sub-1, 'beta', bzeta_ind=11)
#     p.fit(p.num_sub-1, 'exp')

#===============================================================================
# Test of best fit widths for number of different mjds at highest freq
# ==============================================================================
# for i in range(len(mjd_strings)//5):
#
#     mjd0 = data_dict[mjd_strings[i*5]]['mjd']
#     data0 = data_dict[mjd_strings[i*5]]['data']
#     freqs0 = data_dict[mjd_strings[i*5]]['freqs']
#     dur0 = data_dict[mjd_strings[i*5]]['dur']
#
#     p = pcs(mjd0,data0,freqs0,dur0)
#     p.fit(0, 'beta', bzeta_ind=11)
#     p.fit(0, 'exp')

#===============================================================================
# Testing new organized code
# ==============================================================================

#test gaussian fitting
p = pcg(mjd0,data0,freqs0,dur0)
p.fit(0, 'beta')
p.fit(0, 'beta', bzeta_ind = 11)
p.fit(0, 'beta', bzeta_ind = 6, gwidth_ind = 25)
#dont actually ever set pbf width yet, so no function for it
p.fit(0, 'zeta')
p.fit(0, 'zeta', bzeta_ind = 0)
p.fit(0, 'zeta', bzeta_ind = 6, gwidth_ind = 25)
#dont actually ever set pbf width yet, so no function for it
p.fit(0, 'exp')
p.fit(0, 'exp', gwidth_ind = 25)
#dont actually ever set pbf width yet, so no function for it

#test sband intrinsic fitting
p = pcs(mjd0,data0,freqs0,dur0)
p.fit(0, 'beta')
p.fit(0, 'beta', bzeta_ind = 11)
p.fit(0, 'beta', bzeta_ind = 6, iwidth_ind = 25)
#dont actually ever set pbf width yet, so no function for it
p.fit(0, 'zeta')
p.fit(0, 'zeta', bzeta_ind = 0)
p.fit(0, 'zeta', bzeta_ind = 6, iwidth_ind = 25)
#dont actually ever set pbf width yet, so no function for it
p.fit(0, 'exp')
p.fit(0, 'exp', iwidth_ind = 25)
#dont actually ever set pbf width yet, so no function for it

#===============================================================================
# BELOW CODE FROM BEFORE REORGANIZE CLASS ON 6/25
# THIS REORGANIZED INTO TWO CLASSES - ONE FOR EACH INTRINSIC SHAPE
# ==============================================================================




#===============================================================================
# Testing power law for s-band intrinsic shape
# ==============================================================================


#===============================================================================
# Test of best intrinsic pulse shape - gaussian or s-band for beta
# It seems that the s-band profile is overall a better fitting intrinsic pulse shape
# ==============================================================================

# chi_tests_b = np.zeros((10,7,7)) #for each mjd and frequency, record frequency, low_chi, and gwidth
# mjder = np.zeros(10)
#
# for i in range(10):
#
#     num_chan = int(chan[i*5])
#     datas = data[i*5][:num_chan]
#     freqs = freq[i*5][:num_chan]
#
#     mjder[i] = mjds[i*5]
#
#     p = Profile(mjds[i*5], datas, freqs, dur[i*5])
#
#     for ii in range(p.num_sub//2):
#
#         ii = ii*2
#
#         datafitbi = p.fit(ii, beta_ind = 11, intrins = True)
#         chi_tests_b[i][ii//2][1] = datafitbi[0]
#         chi_tests_b[i][ii//2][2] = datafitbi[3]
#         chi_tests_b[i][ii//2][3] = datafitbi[4]
#
#         chi_tests_b[i][ii//2][0] = p.freq_suba
#
#         datafitb = p.fit(ii, beta_ind = 11, gwidth_ind = 27)
#         chi_tests_b[i][ii//2][4] = datafitb[0]
#         chi_tests_b[i][ii//2][5] = datafitb[3]
#         chi_tests_b[i][ii//2][6] = datafitb[4]
#
# np.save('mjds_intrinss_vs_gauss', mjder)
# np.save('intrinss_vs_gauss_b', chi_tests_b)

#===============================================================================
# Test of best intrinsic pulse shape - gaussian or s-band for dec exp
# ==============================================================================

# chi_tests_e = np.zeros((10,7,7)) #for each mjd and frequency, record frequency, low_chi, and gwidth
#
# for i in range(10):
#
#     num_chan = int(chan[i*5])
#     datas = data[i*5][:num_chan]
#     freqs = freq[i*5][:num_chan]
#
#     p = Profile(mjds[i*5], datas, freqs, dur[i*5])
#
#     for ii in range(p.num_sub//2):
#
#         ii = ii*2
#
#         datafitei = p.fit(ii, dec_exp = True, intrins = True)
#         chi_tests_e[i][ii//2][1] = datafitei[0]
#         chi_tests_e[i][ii//2][2] = datafitei[3]
#         chi_tests_e[i][ii//2][3] = datafitei[4]
#
#         chi_tests_e[i][ii//2][0] = p.freq_suba
#
#         datafite = p.fit(ii, dec_exp = True)
#         chi_tests_e[i][ii//2][4] = datafitb[0]
#         chi_tests_e[i][ii//2][5] = datafite[3]
#         chi_tests_e[i][ii//2][6] = datafite[4]
#
# np.save('intrinss_vs_gauss_e', chi_tests_e)

#===============================================================================
# Testing the intrinsic profile fitting again
# Set non intrins beta gwidth to index 27 - about 50 microseconds
# Set intirnsic widths based on best fits below -
# Results ******
#==============================================================================
# 6/22/23 have not run yet with new parameters and beta
#
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
#         datafitb = p.fit(ii, beta_ind = 11, gwidth_pwr_law = True, intrins = True)
#         datafite = p.fit(ii, dec_exp = True, gwidth_pwr_law = True, intrins = True)
#
#         low_chii += datafitb[0]
#         low_chii += datafite[0]
#
#         datafitb = p.fit(ii, beta_ind = 11, gwidth_ind = 27)
#         datafite = p.fit(ii, dec_exp = True, gwidth_pwr_law = True)
#
#         low_chig += datafitb[0]
#         low_chig += datafite[0]
#
# print(low_chig)
# print(low_chii)

#===============================================================================
# BETA: Testing the gwidth power law for intrinsic
# results: Mostly zeros, but a couple of mjds favored between 1 and 2 for power law
# ==============================================================================

# for i in range(10):
#
#     num_chan = int(chan[i*5])
#     datas = data[i*5][:num_chan]
#     freqs = freq[i*5][:num_chan]
#
#     p = Profile(mjds[i*5], datas, freqs, dur[i*5])
#     pwr_ind = p.fit_pwr_law_g(beta_ind = 11, intrins = True)
#
#     print(pwr_ind)

#===============================================================================
# BETA: Collecting best fit gaussian widths for highest frequency pulse in order to
# set a reference frequency and gaussian width for the gwidth pwr law; this is for
# the intrinsic this time
# results: same as dec exp - 11.7 ish
# ==============================================================================
#
# ii = 0
#
# high_freq_gwidth_test = np.zeros((56,2))
#
# for i in range(56):
#
#     num_chan = int(chan[i])
#     datas = data[i][:num_chan]
#     freqs = freq[i][:num_chan]
#
#     p = Profile(mjds[i], datas, freqs, dur[i])
#
#     dataret = p.fit(ii, beta_ind=11, intrins=True)
#
#     high_freq_gwidth_test[i][0] = p.freq_suba #frequency
#     high_freq_gwidth_test[i][1] = dataret[3] #gaussian width
#
#     print(f'Frequency = {p.freq_round} MHz')
#     print(fr'Gaussian Width = {dataret[3]} \mu s')
#
# np.save('high_freq_gwidth_test_intrins_beta', high_freq_gwidth_test)

#===============================================================================
# Testing the intrinsic profile fitting again
# Set non intrins beta gwidth to index 27 - about 50 microseconds
# Set intirnsic widths based on best fit below -
# Results of 458.88256969082136 versus 466.9777182135979
#==============================================================================
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
#         datafitb = p.fit(ii, beta_ind = 11, gwidth_ind = 1, intrins = True)
#         datafite = p.fit(ii, dec_exp = True, gwidth_pwr_law = True, intrins = True)
#
#         low_chii += datafitb[0]
#         low_chii += datafite[0]
#
#         datafitb = p.fit(ii, beta_ind = 11, gwidth_ind = 27)
#         datafite = p.fit(ii, dec_exp = True, gwidth_pwr_law = True)
#
#         low_chig += datafitb[0]
#         low_chig += datafite[0]
#
# print(low_chig)
# print(low_chii)


#===============================================================================
# Collecting best fit gaussian widths for highest frequency pulse in order to
# set a reference frequency and gaussian width for the gwidth pwr law; this is for
# the intrinsic this time
# results: favors highest freq intrinsic width of about 11
# ==============================================================================

# ii = 0
#
# high_freq_gwidth_test = np.zeros((56,2))
#
# for i in range(56):
#
#     num_chan = int(chan[i])
#     datas = data[i][:num_chan]
#     freqs = freq[i][:num_chan]
#
#     p = Profile(mjds[i], datas, freqs, dur[i])
#
#     dataret = p.fit(ii, dec_exp=True, intrins=True)
#
#     high_freq_gwidth_test[i][0] = p.freq_suba #frequency
#     high_freq_gwidth_test[i][1] = dataret[3] #gaussian width
#
#     print(f'Frequency = {p.freq_round} MHz')
#     print(fr'Gaussian Width = {dataret[3]} \mu s')
#
# np.save('high_freq_gwidth_test_intrins', high_freq_gwidth_test)

#===============================================================================
# Testing the intrinsic profile fitting
# Results - Intrins does a bit better - 501.5 versus 459.3, but also wrong pbf width for beta set...
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
# Ran with more narrow gwidth range up to 30 and was favored at about 2 microseconds
# ==============================================================================
#
# best_fit_widths = np.zeros((10,7,20,2))
#
# for i in range(10):
#
#     num_chan = int(chan[i*5])
#     datas = data[i*5][:num_chan]
#     freqs = freq[i*5][:num_chan]
#
#     p = Profile(mjds[i*5], datas, freqs, dur[i*5])
#
#     for ii in range(p.num_sub//2):
#
#         for iii in range(20):
#
#             datafitb = p.fit(ii*2, beta_ind = 11, gwidth_ind = iii, intrins = True)
#
#             best_fit_widths[i][ii][iii][0] = datafitb[0]
#             best_fit_widths[i][ii][iii][1] = datafitb[5]
#
# print(best_fit_widths)
# np.save('test_of_best_beta_gwidth_intrins', best_fit_widths)

#===============================================================================
# Testing the gwidth power law for intrinsic
# Results favor power law index of 0.0
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
