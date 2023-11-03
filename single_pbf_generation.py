import numpy as np
from numpy import zeros, size
from numpy import interp
from scipy.interpolate import CubicSpline
import math
import glob
import sys

import matplotlib.pyplot as plt

from fit_functions import cordes_phase_bins, j1903_period, find_nearest, calculate_tau, stretch_or_squeeze, init_data_phase_bins

#the below function gets the pbf data for a certain value of beta

beta = float(sys.argv[1]) # set spectral index of wavenumber spectrum
zeta = float(sys.argv[2]) # set inner scale of waveumber spectrum
screen = str(sys.argv[3]) # 4 'thin' or 'thick' medium pbf


def pbf_data_beta(beta_input):

    #selected values of beta
    betaselect = np.array([3.1, 3.5, 3.667, 3.8, 3.9, 3.95, 3.975, 3.99, 3.995, 3.9975, 3.999, 3.99999])

    beta_care = find_nearest(betaselect, beta_input)[0]

    # Read in the PBF files
    npzfiles = glob.glob(f'PBF*beta_{beta_care}*zeta_0.000*.npz')
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

        GDnu2array2.append(GDnu2[0:size(vsvec)])
        betavec[n] = bbeta
        dvsvec[n] = dvs
        zetavec[n] = zeta
        sftypevec[n] = sftype

        tvecplotmax = 30

        pbf /= pbf.max()                # unit maximum
        tinds = np.where(tvec < tvecplotmax)
        time = tvec[tinds]

        spline = CubicSpline(np.linspace(0,np.size(pbf)-1,np.size(pbf)), pbf)
        new_pbf = spline(np.linspace(0,np.size(pbf)-1,init_data_phase_bins))

        #returns the times (x-data) and pbf (y-data)
        np.save(f'zeta_0_beta_{beta_care}_pbf.npy', new_pbf/new_pbf.max())


def pbf_data_zeta(zeta_input):

    #selected values of zeta
    zetaselect = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0])

    zeta_care = find_nearest(zetaselect, zeta_input)[0]

    # Select only beta = 3.66667 files:
    npzfiles = glob.glob(f'PBF*3.66**npz')
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

        GDnu2array2.append(GDnu2[0:size(vsvec)])
        betavec[n] = bbeta
        dvsvec[n] = dvs
        zetavec[n] = zeta
        sftypevec[n] = sftype

        tvecplotmax = 30

        pbf /= pbf.max()                # unit maximum
        tinds = np.where(tvec < tvecplotmax)
        time = tvec[tinds]

        spline = CubicSpline(np.linspace(0,np.size(pbf)-1,np.size(pbf)), pbf)
        new_pbf = spline(np.linspace(0,np.size(pbf)-1,init_data_phase_bins))

        #returns the times (x-data) and pbf (y-data)
        np.save(f'zeta_{zeta_care}_beta_11_3_pbf.npy', new_pbf/new_pbf.max())


def save_single_pbf_extended_medium(beta, zeta):

    '''Requires that zeta is 0 is beta is not 11/3. Also, requires that beta
    and zeta values already have pbfs calculated.'''

    if zeta == 0:

        pbf_data_beta(beta)

    else:

        pbf_data_zeta(zeta)


def save_single_pbf_thin_screen(beta, zeta):

    '''Calculate and save 2048 phase bin thin screen pbf for the inputted values
    of beta and zeta (values must be contained in betas and zetas respectively)
    '''

    # Cordes thin screen pbfs
    # CAREFUL: time steps spaced logarithmically
    pbf_data = np.load('generate_pbf_sets_Nbeta_12_Nli_10.npz', allow_pickle=True)

    inner_scale = pbf_data['livec']
    beta_values = pbf_data['betavec']
    time_values = pbf_data['tvecarray']
    pbf_array = pbf_data['pbfarray']
    one_over_e_values = pbf_data['tevec']
    pdf_tau_values = pbf_data['tavevec']

    # because the time steps are spaced logarithmically, start with lots of points
    large_phase_bins = 5000000

    # parameters to collect for each pbf
    betas = beta_values
    zetas = inner_scale

    reference_tau_scale = 50000.0

    i = find_nearest(betas, beta)[1][0][0]
    ii = find_nearest(zetas, zeta)[1][0][0]

    time = time_values[i][ii]
    pbf = pbf_array[i][ii]

    tau = one_over_e_values[i][ii]

    plt.figure(1)
    plt.plot(time, pbf, label=f'Original Logarithmically Spaced', color='blue')

    spline = CubicSpline(time, pbf)

    # large number of phase bins to accomodate log time spacing
    time_linear = np.linspace(time[0], time[-1], large_phase_bins)
    pbf_linear = spline(time_linear)
    plt.plot(time_linear, pbf_linear, label=f'{large_phase_bins} Linearly Interpolated Time Steps', color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'$\beta$'+ f' = {betas[i]},' + r' $\zeta$' + f' = {zetas[ii]},' + r' $\tau$' + f' = {np.round(tau,2)}')
    plt.legend()
    plt.savefig(f'zeta_{zetas[ii]}_beta_{betas[i]}_thin_screen_pbf_linearly_interpolated.pdf')
    plt.show()
    plt.close('all')

    print(f"Scale factor = {reference_tau_scale/tau}")

    rescaled_pbf = stretch_or_squeeze(pbf_linear, reference_tau_scale/tau)
    plt.figure(1)
    plt.plot(time_linear, rescaled_pbf, label=f'{large_phase_bins} Linearly Interpolated Time Steps', color='blue')

    del(pbf_linear)
    del(tau)
    spline = CubicSpline(np.linspace(0,np.size(rescaled_pbf)-1,np.size(rescaled_pbf)), rescaled_pbf)
    new_pbf = spline(np.linspace(0,np.size(rescaled_pbf)-1,init_data_phase_bins))
    plt.plot(np.linspace(time[0], time[-1], init_data_phase_bins), new_pbf, label='Interpolated', color='red')
    plt.title(r'$\beta$'+ f' = {betas[i]},' + r' $\zeta$' + f' = {zetas[ii]},' + r' $\tau$' + f' = {np.round(tau,2)}')
    plt.legend()
    plt.savefig(f'zeta_{zetas[ii]}_beta_{betas[i]}_thin_screen_pbf_stretched.pdf')
    plt.show()
    plt.close('all')

    np.save(f'zeta_{zetas[ii]}_beta_{betas[i]}_thin_screen_pbf.npy', new_pbf/np.max(new_pbf))


if __name__ == '__main__':

    if screen == 'thin':

        save_single_pbf_thin_screen(beta,zeta)

    elif screen == 'thick':

        save_single_pbf_extended_medium(beta,zeta)

    else:

        print('Choose valid screen: thin or thick')
