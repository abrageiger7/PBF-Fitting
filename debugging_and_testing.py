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

ii = p.num_sub - 1


p = Profile(mjds[0], data0, freq0, dur[0])

datab = p.fit(ii, beta_ind = 11, gwidth_ind = 4)

print('low_chi, tau_fin, tau_low, tau_up, self.comp_fse(tau_fin), gwidth, pbf_width_fin)')

print(datab)
