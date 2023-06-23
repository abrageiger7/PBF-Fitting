"""
Created June 2023
@author: Abra Geiger abrageiger7

Creates array of pbfs varying in beta and pbf width (stretch and squeeze factor)
"""

# Provided by Professor Cordes, here is some information on the pbfs

# Reads and plots pulse broadening functions calculated numerically using
# gammaD_from_pde_sw.py [JMC code]

# The PBFs are computed from the partial differential equation for the field
# second moment.   The field originates from a point source embedded in the
# scattering medium.

# The scattering medium is statistically uniform (same scattering strength
# everywhere).

# There is one npz file for each value of $\beta$ = spectral index of wavenumber
# spectrum for the electron density.

# Extracts the pulse broadening function and metadata from each file and plots
# all on the same frame.

# Initial version JMC 2021 March

# Suggested additions:

# 1. Generate fake pulsar pulses with different shapes
        # e.g. single Gaussian component, two and three Gaussian components

# 2. Convolve with PBFs that scale the numerical PBFs to make them wider or
# narrower. This can be done by multipling the FFTs of the PBF and pulse and
# then transforming back with an IFFT.

# 3. Estimate the arrival time using the appropriate function in PyPulse

# 4. Quantify how the shift in TOA (before and after convolution) depends on
# $\beta$ and on the width of the PBF relative to the pulse width before
# scattering

# 5. End result: we'll want curves showing the TOA shift vs. $\beta$ and for
# something like the characteristic pulse broadening time $\tau_d(\beta) / W$
# where $W$ = pulse width before scattering (e.g. FWHM).

import numpy as np
from numpy import zeros, size
from numpy import interp
from scipy.interpolate import CubicSpline
import math
import glob

from fitting_params import *


# Read in the PBF files

npzfiles = glob.glob('PBF*zeta_0.000*.npz')
nfiles = size(npzfiles)

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
        # use lists to handle input that has different length tvec's:
        tvecarray2 = []
        pbfarray2 = []
        GDnu2array2 = []
        betavec = zeros(nfiles)
        dvsvec = zeros(nfiles)
        zetavec = zeros(nfiles)
        sftypevec = zeros(nfiles, dtype=str)

    # Recalculate tvec because earlier versions of gammaD_from_pde_sw.py
    # calculated it differently and seems to be off by x2 (too large)
    dt_spline = 1 / (2.*vsvec.max())
    # factor of 2\pi makes widths in vs-space and t-space in bin units reasonable

    dt_spline *= 2*np.pi
    Nt2 = vsvec.size
    tvec = np.linspace(0, 2*Nt2-1,2*Nt2-1, endpoint=True) * dt_spline


    tvecarray2.append(tvec)
    pbfarray2.append(pbf)
    GDnu2array2.append(GDnu2[0:size(vsvec)])
    betavec[n] = bbeta
    dvsvec[n] = dvs
    zetavec[n] = zeta
    sftypevec[n] = sftype

# indices to sort in beta:
beta_inds = np.argsort(betavec)
beta_sort = betavec[beta_inds]
zeta_sort = zetavec[beta_inds]

tvecplotmax = 30

#the below function gets the pbf data for a certain value of beta

def pbf_data(beta_input, plot = False, data_info = False):

    for n, beta in enumerate(beta_sort):

        #gives the PBF data from the npz files
        if beta == beta_input:
            pbf = pbfarray2[beta_inds[n]]
            pbf /= pbf.max()                # unit maximum
            tvec = tvecarray2[beta_inds[n]]

            tinds = np.where(tvec < tvecplotmax)
            times = tvec[tinds]
            pbfs = pbf[tinds]

            #plots the PBF
            if plot:

                #establishes graph label for the certain beta value
                if beta == 3.667:
                    label = r'$\rm 11/3$'
                elif beta == 3.95:
                    label = r'$\rm %4.2f$'%(beta)
                elif beta == 3.975:
                    label = r'$\rm %5.3f$'%(beta)
                elif beta == 3.99:
                    label = r'$\rm %4.2f$'%(beta)
                elif beta == 3.995:
                    label = r'$\rm %5.3f$'%(beta)
                elif beta == 3.9975:
                    label = r'$\rm %5.4f$'%(beta)
                elif beta ==  3.999:
                    label = r'$\rm %5.3f$'%(beta)
                elif beta < 3.99:
                    label = r'$\rm %4.2f$'%(beta)
                else:
                    label = r'$\rm %3.0f$'%(beta)

                plt.plot(tvec[tinds], pbf[tinds], '-', lw=2, label=label)
                ndecades = 6
                plt.yscale('log')
                ylocs, ytics = calcticks_log(tlog1=-ndecades, tlog2=0, dt=1)
                plt.axis(ymin=0.8*10**(-ndecades))
                plt.yticks(ylocs, ytics)
                plt.xlabel(r'$\rm t \ / \ t_c$', fontsize=16)
                plt.ylabel(r'$\rm PBF(t)$', fontsize=16)
                plt.legend(loc=(0.175,0.75), title=r'$\rm \beta$', ncol=4)
                plt.annotate(plotstamp, xy=(0.6, 0.02), xycoords='figure fraction', ha='left', va='center', fontsize=5)
                plt.show()

            #this is extra data that can be given if called
            if data_info:

                ntmax = pbf.argmax()
                pbfmax = pbf.max()
                pbfinds = pbf[ntmax:].argsort()
                pbfsort = pbf[ntmax:][pbfinds]
                tvecsort = tvec[ntmax:][pbfinds]
                t_e = interp((pbfmax/np.e,), pbfsort, tvecsort)[0]

                return([times, pbfs,('%3d  %5.2f   %6.3f   %6.3f   %6.3f'%(n, zeta,
                    tvec[ntmax], t_e, t_e/tvec[ntmax]))])

            #returns the times (x-data) and pbf (y-data)
            return([times, pbfs])

# now collect the pbf vectors for varying beta

pbf_array = zeros((size(betaselect), cordes_phase_bins))

for n,i in enumerate(betaselect):
    pbf_array[n] = pbf_data(i)[1]


# now collect an array of pbfs varying in beta and pbf width, so must stretch
# and squeeze the pbfs

time_bins = np.arange(cordes_phase_bins)

#an array of data for varying beta values and varying width values
widths_pbf_array = zeros((size(betaselect), size(widths), cordes_phase_bins))

#indexes for beta values
data_index1 = 0
#loop through each value of beta
for i in pbf_array:
    #different widths data for the beta
    widths_data_1 = zeros((size(widths), cordes_phase_bins))

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
            width_pbf_data = zeros(cordes_phase_bins)

            #add the intensity that loops around for stretched pulses
            index = 0
            #while(index<(np.max(times_adjusted))):
            while(index<(np.max(times_adjusted)-cordes_phase_bins)):
                interp_sect = interpolate_width(np.arange(index,index+cordes_phase_bins,1))
                width_pbf_data = np.add(width_pbf_data, interp_sect)
                index = index+cordes_phase_bins

            final_interp_sect_array = np.arange(index, int(np.max(times_adjusted))+1, 1)
            final_interp_sect = interpolate_width(final_interp_sect_array)
            final_interp_sect = np.concatenate((final_interp_sect, zeros((index + cordes_phase_bins - int(np.max(times_adjusted)) - 1))))
            width_pbf_data = np.add(width_pbf_data, final_interp_sect)

        #squeeze narrowed pulses and add section of training zeros onto the end of them
        elif ii<1:
            #lengthen the array of the pulse so the pulse is comparatively narrow, adding zeros to the end
            width_pbf_data = zeros(int((1/ii)*cordes_phase_bins))
            width_pbf_data[:cordes_phase_bins] = i
            times_scaled = zeros(int((1/ii)*cordes_phase_bins))
            #scale back to an array of size cordes_phase_bins
            for iv in range(int((1/ii)*cordes_phase_bins)):
                times_scaled[iv] = cordes_phase_bins/(int((1/ii)*cordes_phase_bins))*iv
            interpolate_less1 = CubicSpline(times_scaled, width_pbf_data)
            width_pbf_data = interpolate_less1(np.arange(cordes_phase_bins))

        #for width values of 1, no alteration necessary
        elif ii == 1:
            width_pbf_data = i

        #put all profiles to unit height
        width_pbf_data = width_pbf_data/np.max(width_pbf_data)

        widths_data_1[data_index2] = width_pbf_data
        data_index2 = data_index2+1

    widths_pbf_array[data_index1] = widths_data_1
    data_index1 = data_index1+1


#the final array of broadening functions is widths_pbf_array
np.save("beta_widths_pbf_data", widths_pbf_array)
