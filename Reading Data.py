# -*- coding: utf-8 -*-
"""
Created on Wed May 3 2023
Last Edited on Mon May 22 2023
@author: Abra Geiger abrageiger7
"""

data = np.load("J1903_data.npy")
freq = np.load("J1903_freqs.npy")
mjds = np.load("J1903_mjds.npy")
chan = np.load("J1903_numchan.npy")

print(data)
print(freq)
print(mjds)
print(chan)
