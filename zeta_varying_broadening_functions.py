"""
Created June 2023
@author: Abra Geiger abrageiger7

Creates array of pbfs varying in zeta and pbf width (stretch and squeeze factor)
"""

# Reference commentary regarding pbfs and their generation in
# beta_varying_broadening_functions.py

import numpy as np
from numpy import zeros, size
from numpy import interp
from scipy.interpolate import CubicSpline
import math
import glob

from fitting_params import *

# Read in the PBF files

#collecting broadening functions

pbf_array = np.zeros((np.size(zetaselect), cordes_phase_bins))
time_array = np.zeros((np.size(zetaselect), cordes_phase_bins))

# Select only beta = 3.66667 files:
npzfiles = glob.glob('PBF*3.66667*npz')
nfiles = np.size(npzfiles)

for n, npzfile in enumerate(npzfiles):
    indata = np.load(npzfile, allow_pickle=True)    # True needed to get dicts

    tvec = indata['tvec']
    pbf = indata['PBF']
    GDnu2 = indata['GDnu2']
    bbeta = indata['bb']
    dvs = indata['dvs']
    vsvec = indata['vsvec']
    inargs = indata['args']
    zeta = inargs.all().zeta
    sftype = inargs.all().sftype

    if n==0:
        tvecarray = zeros((nfiles, size(tvec)))
        pbfarray = zeros((nfiles, size(pbf)))
        GDnu2array = zeros((nfiles, size(vsvec)))
        betavec = zeros(nfiles)
        zetavec = zeros(nfiles)
        sftypevec = zeros(nfiles, dtype=str)

    # Recalculate tvec because earlier versions of gammaD_from_pde_sw.py
    # calculated it differently and seems to be off by x2 (too large)
    dt_spline = 1 / (2.*vsvec.max())
    # factor of 2\pi makes widths in vs-space and t-space in bin units reasonable

    dt_spline *= 2*np.pi
    Nt2 = vsvec.size
    tvec = np.linspace(0, 2*Nt2-1,2*Nt2-1, endpoint=True) * dt_spline

    tvecarray[n] = tvec
    pbfarray[n] = pbf
    GDnu2array[n] = GDnu2[0:size(vsvec)]
    betavec[n] = bbeta
    zetavec[n] = zeta
    sftypevec[n] = sftype

# indices to sort in zeta:
zeta_inds = np.argsort(zetavec)

zeta_sort = zetavec[zeta_inds]
pbf_sort = pbfarray[zeta_inds]
tvec_sort = tvecarray[zeta_inds]

tvecplotmax = 30

fig = plt.figure()
ax = fig.add_subplot(111)

z_ind = 0

for n, tvec in enumerate(tvec_sort):
    zeta = zeta_sort[n]
    if zeta in zetaselect:
        pbf = pbf_sort[n]
        pbf /= pbf.max()                # unit maximum

        tinds = np.where(tvec < tvecplotmax)

        pbf_array[z_ind] = pbf[tinds]
        time_array[z_ind] = tvec[tinds]

        z_ind += 1

time_bins = np.arange(cordes_phase_bins)

#an array of data for varying beta values and varying width values
widths_pbf_array = np.zeros((np.size(zetaselect), np.size(widths), 9549))

#indexes for beta values
data_index1 = 0
#loop through each value of beta
for i in pbf_array:
    #different widths data for the beta
    widths_data_1 = np.zeros((np.size(widths), 9549))

    #indexes for width values
    data_index2 = 0
    for ii in widths:
        #adjust times to this width
        #multiply the times by the stretch/squeeze value (the width)
        #for stretch greater than zero, the PBF will broaden
        #for stretch less than zero, the PBF will narrow

        if ii>1:
            times_adjusted = time_bins*ii #1- widen the pulse
            #interpolate the pulse in its broadened state
            interpolate_width = CubicSpline(times_adjusted, i) #2- interpolate to get section of the pulse (extrapolate = True?
            #-> probably don't need because should only be interpolating)
            width_pbf_data = np.zeros(9549)

            #add the intensity that loops around for stretched pulses
            index = 0
            #while(index<(np.max(times_adjusted))):
            while(index<(np.max(times_adjusted)-9549)):
                interp_sect = interpolate_width(np.arange(index,index+9549,1))
                width_pbf_data = np.add(width_pbf_data, interp_sect)
                #plt.plot(interp_sect)
                index = index+9549

            final_interp_sect_array = np.arange(index, int(np.max(times_adjusted))+1, 1)
            final_interp_sect = interpolate_width(final_interp_sect_array)
            final_interp_sect = np.concatenate((final_interp_sect, np.zeros((index + 9549 - int(np.max(times_adjusted)) - 1))))
            width_pbf_data = np.add(width_pbf_data, final_interp_sect)


        #squeeze narrowed pulses and add section of training zeros onto the end of them
        elif ii<1:
            #lengthen the array of the pulse so the pulse is comparatively narrow, adding zeros to the end
            width_pbf_data = np.zeros(int((1/ii)*9549))
            width_pbf_data[:9549] = i
            times_scaled = np.zeros(int((1/ii)*9549))
            #scale back to an array of size 9549
            for iv in range(int((1/ii)*9549)):
                times_scaled[iv] = 9549/(int((1/ii)*9549))*iv
            interpolate_less1 = CubicSpline(times_scaled, width_pbf_data)
            width_pbf_data = interpolate_less1(np.arange(9549))

        #for width values of 1, no alteration necessary
        elif ii == 1:
            width_pbf_data = i

        #put all profiles to unit height
        width_pbf_data = width_pbf_data/np.max(width_pbf_data)

        widths_data_1[data_index2] = width_pbf_data
        data_index2 = data_index2+1

    widths_pbf_array[data_index1] = widths_data_1
    data_index1 = data_index1+1

np.save("zeta_widths_pbf_data", widths_pbf_array)
