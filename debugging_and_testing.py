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

num_chan0 = int(chan[0])
data0 = data[0][:num_chan0]
freq0 = freq[0][:num_chan0]


p = Profile(mjds[0], data0, freq0, dur[0])

ii = 0
iii = 11
iv = 6

datab = p.fit(ii, beta_ind = 11, gwidth_ind = 4)

print(mjds[0])

print('low_chi, tau_fin, tau_low, tau_up, self.comp_fse(tau_fin), gwidth, pbf_width_fin)')

print(datab)

dataeh = p.fit(ii, dec_exp=True)
print(dataeh)

dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 3)
print(dataeh[0])

dataem = p.fit(iv, dec_exp=True, gwidth_ind = 3)
print(dataem[0])

datael = p.fit(iii, dec_exp=True, gwidth_ind = 3)
print(datael[0])

dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 4)
print(dataeh[0])

dataem = p.fit(iv, dec_exp=True, gwidth_ind = 4)
print(dataem[0])

datael = p.fit(iii, dec_exp=True, gwidth_ind = 4)
print(datael[0])

dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 5)
print(dataeh[0])

dataem = p.fit(iv, dec_exp=True, gwidth_ind = 5)
print(dataem[0])

datael = p.fit(iii, dec_exp=True, gwidth_ind = 5)
print(datael[0])

dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 6)
print(dataeh[0])

dataem = p.fit(iv, dec_exp=True, gwidth_ind = 6)
print(dataem[0])

datael = p.fit(iii, dec_exp=True, gwidth_ind = 6)
print(datael[0])


dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 7)
print(dataeh[0])


dataem = p.fit(iv, dec_exp=True, gwidth_ind = 7)
print(dataem[0])


datael = p.fit(iii, dec_exp=True, gwidth_ind = 7)
print(datael[0])


dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 8)
print(dataeh[0])


dataem = p.fit(iv, dec_exp=True, gwidth_ind = 8)
print(dataem[0])


datael = p.fit(iii, dec_exp=True, gwidth_ind = 8)
print(datael[0])


dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 9)
print(dataeh[0])


dataem = p.fit(iv, dec_exp=True, gwidth_ind = 9)
print(dataem[0])


datael = p.fit(iii, dec_exp=True, gwidth_ind = 9)
print(datael[0])


dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 10)
print(dataeh[0])


dataem = p.fit(iv, dec_exp=True, gwidth_ind = 10)
print(dataem[0])


datael = p.fit(iii, dec_exp=True, gwidth_ind = 10)
print(datael[0])


dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 11)
print(dataeh[0])


dataem = p.fit(iv, dec_exp=True, gwidth_ind = 11)
print(dataem[0])


datael = p.fit(iii, dec_exp=True, gwidth_ind = 11)
print(datael[0])


dataeh = p.fit(ii, dec_exp=True, gwidth_ind = 12)
print(dataeh[0])


dataem = p.fit(iv, dec_exp=True, gwidth_ind = 12)
print(dataem[0])


datael = p.fit(iii, dec_exp=True, gwidth_ind = 12)
print(datael[0])



# import tau
# #testing that no tau values are the same to confirm that grid spacing is resolved enough
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
