"""
Created April 2023
@author: Abra Geiger abrageiger7

File of Constants and Functions for Fitting Broadening Functions
"""

# Functions contained:
# find_nearest(tau helper function)
# ecdf and pdf_to_cdf (likelihood helper functions), likelihood_evaluator
# chi2_distance (helper function)
# subaverages4
# calculate_tau
# triple_gauss

#imports
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline
from scipy import special

#===============================================================================
# Phase bins and timing
# ==============================================================================

#number of phase bins that j1903 data comes with
init_data_phase_bins = 2048

#number of phase bins for a profile - time average every 8 of the original data
#2048

#phase_bins = 2048
phase_bins = init_data_phase_bins//8

# number of time bins for original cordes pbfs
cordes_phase_bins = 9549

#phase bins
t = np.linspace(0, phase_bins, phase_bins)

#seconds per pulse period
sec_pulse_per = 0.0021499

#seconds to milliseconds conversion
s_to_ms_conv = 1e3

#time phase bins in milliseconds
time = np.arange(0,phase_bins,1) * (sec_pulse_per/phase_bins) * s_to_ms_conv #milliseconds

opr_size = int((500/2048)*phase_bins) #number of phase bins for offpulse noise calculation

j1903_period = sec_pulse_per * 1e6 #microseconds

#===============================================================================
# Finite Scintle Effect
# ==============================================================================

B = 60.0 #approximate frequency range for each channel in MHz - subaverage 4 of about 12.5 MHz each
D = 0.64 #approximate distance to the pulsar in kpc
nt = 0.2 #filling factor over time
nv = 0.2 #filling factor over bandwidth
c_1 = 1.16 #constant for uniform scattering media in fse
vel_cons = 2.53e4 #km/sec velocity constant in relationship between pulsar
#perpendicular velocity and delta tau_d
V = 18.9 #km/sec perpendicular velocity of the pulsar

#===============================================================================
# Functions
# ==============================================================================

def find_nearest(a, a0):
    '''Element in nd array `a` closest to the scalar value `a0`

    Preconditions: a is an n dimensional array and a0 is a scalar value

    Returns index, value'''
    idx = np.abs(a - a0).argmin()
    return a.flat[idx], np.where((a == a.flat[idx]))

#Prof Lam's code for likelihood evaluator
#Creates empirical cdf
def ecdf(values, sort=True):
    if sort:
        values = np.sort(values)
    return values, np.linspace(0, 1, len(values))

EPS = special.erf(1.0/np.sqrt(2))/2.0

def pdf_to_cdf(pdf, dt=1):
    return np.cumsum(pdf)*dt

def likelihood_evaluator(x, y, cdf=False, median=False, pm=True, values=None):
    """
    cdf: if True, x,y describe the cdf
    median: if True, use the median value, otherwise the peak of the pdf (assuming cdf=False
    pm: xminus and xplus are the plus/minus range, not the actual values
    Future: give it values to grab off the CDF (e.g. 2 sigma, 99%, etc)
    values: use this array
    """
    if not cdf:
        y = y/np.trapz(y, x=x)
        ycdf = pdf_to_cdf(y, dt=(x[1]-x[0]))
    else: #else given a cdf
        ycdf = y

    if not values:
        if median:
            yb = 0.50   #Now take the median!
        else:
            indb = np.argmax(y)
            yb = ycdf[indb]
        ya = yb - EPS
        yc = yb + EPS
        yd = 0.95

        inda = np.argmin(np.abs(ycdf - ya))
        if median:
            indb = np.argmin(np.abs(ycdf - yb))
        indc = np.argmin(np.abs(ycdf - yc))
        indd = np.argmin(np.abs(ycdf - yd))

        inds = np.arange(inda, indc+1) #including indc
        #print indc-inda, np.trapz(L[inds], x=Vrs[inds])
        xval = x[indb]
        if pm:
            xminus = x[indb] - x[inda]
            xplus = x[indc] - x[indb]
        else:
            xminus = x[inda]
            xplus = x[indc]
        x95 = x[indd]

        return xval, xminus, xplus, x95
    else:
        retval = np.zeros_like(values)
        for i, v in enumerate(values):
            indv = np.argmin(np.abs(ycdf - v))
            retval[i] = x[indv]
        return retval

def stretch_or_squeeze(i, ii):

    '''i is profile stretching and squeezing (numpy array), ii is stretch or squeeze factor (float or int)
    Returns unit height stretched or squeezed array'''

    cordes_phase_bins = np.size(i)

    ii = np.abs(ii)

    time_bins = np.arange(cordes_phase_bins)

    #adjust times to this width
    #multiply the times by the stretch/squeeze value (the width)
    #for stretch greater than zero, the PBF will broaden
    #for stretch less than zero, the PBF will narrow

    if ii>1:
        times_adjusted = time_bins*ii #1- widen the pulse
        #interpolate the pulse in its broadened state
        interpolate_width = CubicSpline(times_adjusted, i) #2- interpolate to get section of the pulse (extrapolate = True?
        #-> probably don't need because should only be interpolating)
        width_pbf_data = np.zeros(cordes_phase_bins)

        #add the intensity that loops around for stretched pulses
        index = 0
        #while(index<(np.max(times_adjusted))):
        while(index<(np.max(times_adjusted)-cordes_phase_bins)):
            interp_sect = interpolate_width(np.arange(index,index+cordes_phase_bins,1))
            width_pbf_data = np.add(width_pbf_data, interp_sect)
            index = index+cordes_phase_bins

        final_interp_sect_array = np.arange(index, int(np.max(times_adjusted))+1, 1)
        final_interp_sect = interpolate_width(final_interp_sect_array)
        final_interp_sect = np.concatenate((final_interp_sect, np.zeros((index + cordes_phase_bins - int(np.max(times_adjusted)) - 1))))
        width_pbf_data = np.add(width_pbf_data, final_interp_sect)

    #squeeze narrowed pulses and add section of training zeros onto the end of them
    elif ii<1:
        #lengthen the array of the pulse so the pulse is comparatively narrow, adding zeros to the end
        width_pbf_data = np.zeros(int((1/ii)*cordes_phase_bins))
        width_pbf_data[:cordes_phase_bins] = i
        times_scaled = np.zeros(int((1/ii)*cordes_phase_bins))
        #scale back to an array of size cordes_phase_bins
        for iv in range(int((1/ii)*cordes_phase_bins)):
            times_scaled[iv] = cordes_phase_bins/(int((1/ii)*cordes_phase_bins))*iv

        interpolate_less1 = CubicSpline(times_scaled, width_pbf_data)
        width_pbf_data = interpolate_less1(np.arange(cordes_phase_bins))

    #for width values of 1, no alteration necessary
    else:
        width_pbf_data = i

    #unit area
    width_pbf_data = width_pbf_data / np.max(width_pbf_data)

    return(width_pbf_data)


def chi2_distance(A, B, std, subt_deg_of_freedom):

    '''Takes two vectors and calculates their comparative chi-squared value

    Pre-condition: A and B are 2 vectors of the same length and num_fitted (int) is
    the number of fitted parameters for dividing by the number of degrees of freedom.
    std is the standard error to divide through by, and subt_deg_of_freedom is
    the amount of degrees of freedom subtracted from the length of the compared
    data arrays.

    Returns chi-squared value'''

    squared_residuals = []
    for (a, b) in zip(A, B):
        sq_res = (a-b)**2
        squared_residuals.append(sq_res)

    squared_residuals = np.array(squared_residuals)
    chi_squared = np.sum(squared_residuals/(std**2)) / (len(squared_residuals) - subt_deg_of_freedom)

    return(chi_squared)

def subaverages4(mjdi, data, freqsm, final_phase_bins, plot = False):
    '''Takes an epoch of pulsar data and subaverages every four frequency
    channels

    Pre-condition:
    mjdi (float): the epoch mjd
    data (numpy array): a 2D array of the epoch data overfrequency and time
    freqsm (list): the 1D frequency array corresponding the channels within the data
    plot (bool): if True, will plot the data in frequency and time and plot the
    four highest frequency channels

    Returns the subaveraged data (numpy array), the average frequencies for
    this subaveraged data (list), and mjdi (float). Also subaverages in time if
    phase bins is different from 2048 as in the data.'''

    if plot == True:
        #plots the pulse over time and frequency
        plt.imshow(data, aspect='26.0', origin='lower')
        plt.ylabel('Frequency (MHz)')
        plt.xlabel('Pulse Period (ms)')
        plt.title('J1903+0327 Observation on MJD ' + str(mjdi)[:5])
        xlabels_start = np.linspace(0, 2.15, 10)
        xlabels = np.zeros(10)
        for i in range(10):
            xlabels[i] = str(xlabels_start[i])[:4]
        ylabels_start = np.linspace(max(freqsm), min(freqsm), 10)
        ylabels = np.zeros(10)
        for i in range(10):
            ylabels[i] = str(ylabels_start[i])[:4]
        plt.xticks(ticks = np.linspace(0,phase_bins,10), labels = xlabels)
        plt.yticks(ticks = np.linspace(0,len(freqsm),10), labels = ylabels)
        plt.colorbar().set_label('Pulse Intensity')
        plt.show()

    subs = np.zeros((len(freqsm)//4,init_data_phase_bins))
    center_freqs = np.zeros(len(freqsm)//4)

    #floor division for subintegrations all of 4 frequency channels
    #also compute the average frequencies for each subintegration
    for i in range(len(freqsm)//4):
        datad = data[4*i:(4*i)+4]
        dataf = freqsm[4*i:(4*i)+4]
        subs[i] = np.average(datad, axis = 0)
        center_freqs[i] = np.average(dataf)

    #ignores any excess beyond the highest multiple of 4 frequency channels

    #now subaveraging in time
    if final_phase_bins != init_data_phase_bins:
        subs_time_avg = np.zeros((len(freqsm)//4,phase_bins))

        for i in range(len(freqsm)//4):
            for ii in range(phase_bins):
                subs_time_avg[i][ii] = np.average(subs[i][((init_data_phase_bins//phase_bins)*ii):((init_data_phase_bins//phase_bins)*ii)+(init_data_phase_bins//phase_bins)])
        subs = subs_time_avg

    if plot == True:
        fig, ax = plt.subplots(2,2)
        fig.suptitle('MJD 57537 High Frequency Pulses')
        fig.size = (16,24)
        title = 'Pulse at Frequency ' + str(np.round(freqsm[0])) + 'MHz'
        ax[0,0].plot(time, data[0])
        ax[0,0].set_title(title)

        title = 'Pulse at Frequency ' + str(np.round(freqsm[1])) + 'MHz'
        ax[0,1].plot(time, data[1])
        ax[0,1].set_title(title)

        title = 'Pulse at Frequency ' + str(np.round(freqsm[2])) + 'MHz'
        ax[1,0].plot(time, data[2])
        ax[1,0].set_title(title)

        title = 'Pulse at Frequency ' + str(np.round(freqsm[3])) + 'MHz'
        ax[1,1].plot(time, data[3])
        ax[1,1].set_title(title)

        for ax1 in ax.flat:
            ax1.set(xlabel='Pulse Phase (ms)', ylabel='Pulse Phase (ms)')

        plt.tight_layout()
        plt.show()

        for i in range(np.size(center_freqs)):
            plt.xlabel('Pulse Phase')
            plt.ylabel('Pulse Intensity')
            plt.title('Subintegration at ' + str(center_freqs[i]) + 'MHz')
            plt.plot(subs[i])
            plt.show()

    return(subs, center_freqs, mjdi)

def subaverage(data, freqsm, final_phase_bins, subaverage_by):
    '''Takes an epoch of pulsar data and subaverages every subaverage_by
    frequency channels

    Pre-condition:
    data (numpy array): a 2D array of the epoch data overfrequency and time
    freqsm (list): the 1D frequency array corresponding the channels within the data
    final_phase_bins (int): number of phase bins to time average to
    subaverage_by (int): number of channels to average by

    Returns the subaveraged data (numpy array) and the average frequencies for
    this subaveraged data (list). Also subaverages in time if
    phase bins is different from 2048 as in the data.'''

    subs = np.zeros((len(freqsm)//subaverage_by,init_data_phase_bins))
    center_freqs = np.zeros(len(freqsm)//subaverage_by)

    #floor division for subintegrations all of 4 frequency channels
    #also compute the average frequencies for each subintegration
    for i in range(len(freqsm)//subaverage_by):
        datad = data[subaverage_by*i:(subaverage_by*i)+subaverage_by]
        dataf = freqsm[subaverage_by*i:(subaverage_by*i)+subaverage_by]
        subs[i] = np.average(datad, axis = 0)
        center_freqs[i] = np.average(dataf)

    #ignores any excess beyond the highest multiple of 4 frequency channels
    #now subaveraging in time
    if final_phase_bins != init_data_phase_bins:
        subs_time_avg = np.zeros((len(freqsm)//subaverage_by,phase_bins))

        for i in range(len(freqsm)//subaverage_by):
            for ii in range(phase_bins):
                subs_time_avg[i][ii] = np.average(subs[i]\
                [((init_data_phase_bins//phase_bins)*ii):\
                ((init_data_phase_bins//phase_bins)*ii)\
                +(init_data_phase_bins//phase_bins)])
        subs = subs_time_avg

    return(subs, center_freqs)

def calculate_tau(profile):
    '''Calculates tau value of J1903 profile by calculating where it decays to
    the value of its max divided by e. Microseconds

    Preconditions: profile is a 1 dimensional array of the length of one J1903
    period'''

    iii = np.copy(profile)
    #isolate decaying part of pbf
    maxim = np.where((iii == np.max(iii)))[0][0]
    iii[:maxim] = 0.0
    near = find_nearest(iii, np.max(iii)/math.e)
    tau = (near[1][0][0] - maxim) * j1903_period / np.size(profile) #microseconds
    tau_index = near[1][0][0]
    tau_unconvert = near[0]
    return(tau, tau_index, tau_unconvert)

def single_gauss(p, t, unit_area = True):
    '''Input of this array for the 3 Gaussian components: amp, mean, width
    Returns unit area 1 component Gaussian'''
    gauss = (p[0]*np.exp((-1.0/2.0)*(((t-(p[1]* phase_bins))/(p[2]* phase_bins))*((t-(p[1]* phase_bins))/(p[2]* phase_bins)))))
    #unit area
    max_gauss = np.max(gauss)
    area_gauss = trapz(gauss)
    if unit_area == False:
        return(gauss)
    gauss = gauss / area_gauss
    return(gauss, max_gauss/area_gauss)

def triple_gauss(p, g, q, t, unit_area = True):
    '''Input of three arrays for the 3 Gaussian components: amp, mean, width
    Returns unit area 3 component Gaussian'''
    phase_bins = np.size(t)
    gauss = (p[0]*np.exp((-1.0/2.0)*(((t-(p[1]* phase_bins))/(p[2]* phase_bins))*((t-(p[1]* phase_bins))/(p[2]* phase_bins))))) + (g[0]*np.exp((-1.0/2.0)*(((t-(g[1]* phase_bins))/(g[2]* phase_bins))*((t-(g[1]* phase_bins))/(g[2]* phase_bins))))) + (q[0]*np.exp((-1.0/2.0)*(((t-(q[1]* phase_bins))/(q[2]* phase_bins))*((t-(q[1]* phase_bins))/(q[2]* phase_bins)))))
    #unit area
    max_gauss = np.max(gauss)
    area_gauss = trapz(gauss)
    if unit_area == False:
        return(gauss)
    gauss = gauss / area_gauss
    return(gauss, max_gauss/area_gauss)

def convolve(arr1, arr2):
    '''Input of two 1D arrays and returns them convolved and normalized to
    unit height'''
    if np.size(arr1) != np.size(arr2):
        print('Input arrays must be the same size.')
    arr1_unita = arr1/trapz(arr1)
    arr2_unita = arr2/trapz(arr2)
    convolved = np.fft.ifft(np.fft.fft(arr1_unita)*np.fft.fft(arr2)).real
    convolved = convolved / np.max(convolved)
    return(convolved)

def profile_fscrunch(data):

    '''Returns the frequency average of the input data array.

    2D numpy array data: data array to average. Requires axes of frequency and
    time respectively'''

    return(np.average(data, axis = 0))

def calculate_rms(profile):

    '''Calculates the RMS of the noise of the first fifth of phase for pulse
    data profile 'profile'.

    1D numpy array profile: pulse data profile'''

    opr_size = np.size(profile)//5

    rms_collect = 0
    for i in range(opr_size):
        rms_collect += profile[i]**2

    rms = math.sqrt(rms_collect/opr_size)
    return(rms)

def time_average(profile, new_phase_bins):

    '''Averages the given profile in time resulting in a time averaged profile
    with new_phase_bins number of phase bins.

    numpy array profile: the data profile to subaverage
    int new_phase_bins: the number of data points to subaverage to'''

    if (np.size(profile)%new_phase_bins != 0):
        raise Exception('Must time average by an integer factor!')

    number_to_average = np.size(profile)//new_phase_bins

    averaged = np.zeros(new_phase_bins)

    for i in range(np.size(averaged)):
        averaged[i] = np.average(profile[number_to_average*i:number_to_average*(i+1)])

    return(averaged)
