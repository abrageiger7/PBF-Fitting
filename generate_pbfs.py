"""
Created June 2023
@author: Abra Geiger abrageiger7

Creates array of pbfs varying in beta and pbf width (stretch and squeeze factor)
"""

# Provided by Professor Cordes, here is some information on the pbfs
#===============================================================================

# Reads and plots pulse broadening functions calculated numerically using
# gammaD_from_pde_sw.py [JMC code]

# The PBFs are computed from the partial differential equation for the field
# second moment. The field originates from a point source embedded in the
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

#===============================================================================
import numpy as np
from numpy import zeros, size
from numpy import interp
from scipy.interpolate import CubicSpline
import math
import glob
import sys

import matplotlib.pyplot as plt

from fit_functions import cordes_phase_bins, j1903_period, find_nearest, calculate_tau, stretch_or_squeeze
#import the period and necessary functions to calculate the pbf tau value
#in units of microseconds
#also, for having the number of cordes phasebins in one place
#also, function for stretching and squeezing - ends with unit height stretched
#or squeezed array


#===============================================================================
# Define Functions
# ==============================================================================

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

if __name__ == '__main__':

    #array of widths used (pbf stretch factors)
    #previously as low as .1 and as high as 42
    num_pbfwidth = 400

    #looking to get tau values on this range for
    tau_values = np.linspace(0.1,500,399)

    #===============================================================================
    # Generate Beta and Tau Varying PBFs
    # ===============================================================================

    #selected values of beta
    betaselect = np.array([3.1, 3.5, 3.667, 3.8, 3.9, 3.95, 3.975, 3.99, 3.995, 3.9975, 3.999, 3.99999])

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

    # now collect the pbf vectors for varying beta

    pbf_array = zeros((size(betaselect), cordes_phase_bins))

    for n,i in enumerate(betaselect):
        pbf_array[n] = pbf_data(i)[1]

    starting_tau_values = np.zeros(np.size(betaselect))

    for i in range(np.size(betaselect)):
        starting_tau_values[i] = calculate_tau(pbf_array[i])[0]

    # now collect an array of pbfs varying in beta and pbf width, so must stretch
    # and squeeze the pbfs

    time_bins = np.arange(cordes_phase_bins)

    #an array of data for varying beta values and varying width values
    widths_pbf_array = zeros((size(betaselect), size(tau_values), cordes_phase_bins))

    #indexes for beta values
    data_index1 = 0
    #loop through each value of beta
    for i in pbf_array:

        widths = tau_values/starting_tau_values[data_index1]
        #different widths data for the beta
        widths_data_1 = zeros((size(widths), cordes_phase_bins))

        #indexes for width values
        data_index2 = 0
        for ii in widths:
            #adjust times to this width
            #multiply the times by the stretch/squeeze value (the width)
            #for stretch greater than zero, the PBF will broaden
            #for stretch less than zero, the PBF will narrow

            width_pbf_data = stretch_or_squeeze(i,ii)

            widths_data_1[data_index2] = width_pbf_data
            data_index2 = data_index2+1

        widths_pbf_array[data_index1] = widths_data_1
        data_index1 = data_index1+1


    # calculate tau values for the pbfs
    beta_tau_values = np.zeros((np.size(betaselect), np.size(tau_values)))

    data_index = 0
    for i in widths_pbf_array:
        data_index2 = 0
        for ii in i:
            tau_ii = calculate_tau(ii)
            beta_tau_values[data_index][data_index2] = tau_ii[0]
            data_index2 = data_index2+1
        data_index = data_index+1

    for i in range(10):

        plt.plot(widths_pbf_array[7][i*35])
    plt.show()

    #the final array of broadening functions is widths_pbf_array
    np.savez("beta_widths_pbf_data", betas = betaselect, taus_mus = beta_tau_values, pbfs_unitheight = widths_pbf_array)

    #===============================================================================
    # Generate Zeta and Tau Varying PBFs
    # ===============================================================================

    #selected values of zeta
    zetaselect = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0])

    pbf_array = np.zeros((np.size(zetaselect), cordes_phase_bins))
    time_array = np.zeros((np.size(zetaselect), cordes_phase_bins))

    # Select only beta = 3.66667 files:
    npzfiles = glob.glob('PBF*3.66*npz')
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

    starting_tau_values = np.zeros(np.size(zetaselect))

    for i in range(np.size(zetaselect)):
        starting_tau_values[i] = calculate_tau(pbf_array[i])[0]

    #an array of data for varying beta values and varying width values
    widths_pbf_array = np.zeros((np.size(zetaselect), np.size(tau_values), 9549))


    #indexes for beta values
    data_index1 = 0
    #loop through each value of beta
    for i in pbf_array:
        #different widths data for the beta

        widths = tau_values/starting_tau_values[data_index1]

        widths_data_1 = np.zeros((np.size(widths), 9549))

        #indexes for width values
        data_index2 = 0
        for ii in widths:
            #adjust times to this width
            #multiply the times by the stretch/squeeze value (the width)
            #for stretch greater than zero, the PBF will broaden
            #for stretch less than zero, the PBF will narrow

            width_pbf_data = stretch_or_squeeze(i,ii)

            widths_data_1[data_index2] = width_pbf_data
            data_index2 = data_index2+1

        widths_pbf_array[data_index1] = widths_data_1
        data_index1 = data_index1+1


    # calculate tau values for the pbfs
    zeta_tau_values = np.zeros((np.size(zetaselect), np.size(tau_values)))

    data_index = 0
    for i in widths_pbf_array:
        data_index2 = 0
        for ii in i:
            tau_ii = calculate_tau(ii)
            zeta_tau_values[data_index][data_index2] = tau_ii[0]
            data_index2 = data_index2+1
        data_index = data_index+1

    for i in range(10):

        plt.plot(widths_pbf_array[4][i*35])

    plt.show()

    #the final array of broadening functions is widths_pbf_array
    np.savez("zeta_widths_pbf_data", zetas = zetaselect, taus_mus = zeta_tau_values, pbfs_unitheight = widths_pbf_array)


    #===============================================================================
    # Generate Tau Varying Exponential PBFs
    # ===============================================================================

    len_dec_exp_profile = 45

    #create varying exponential profiles
    widths_exp_array = np.zeros((np.size(tau_values), cordes_phase_bins))


    time_bins = np.linspace(0,len_dec_exp_profile, cordes_phase_bins)
    exponential = np.exp(-time_bins)
    i = exponential

    starting_tau_value = calculate_tau(i)[0]

    widths = tau_values/starting_tau_value

    #indexes for stretch width values
    data_index2 = 0
    for ii in widths:
        #adjust times to this width by multiplying the times by the stretch/squeeze value (the width)
        #for stretch greater than zero, the PBF will broaden
        #for stretch less than zero, the PBF will narrow

        width_pbf_data = stretch_or_squeeze(i,ii)

        #append broadening function to array
        widths_exp_array[data_index2] = width_pbf_data
        data_index2 = data_index2+1

    #now collect tau values for the broadening functions
    exp_tau_values = np.zeros(np.size(tau_values))

    data_index = 0
    for i in widths_exp_array:
        exp_tau_values[data_index] = calculate_tau(i)[0]
        data_index = data_index+1

    for i in range(10):

        plt.plot(widths_exp_array[i*35])
    plt.show()

    np.savez('exp_widths_pbf_data', taus_mus = exp_tau_values, pbfs_unitheight = widths_exp_array)
