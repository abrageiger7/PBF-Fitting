"""
Created April 2023
@author: Abra Geiger abrageiger7

File of Functions for Fitting Broadening Functions
"""

# Functions contained:
# find_nearest(tau helper function)
# ecdf and pdf_to_cdf (likelihood helper functions), likelihood_evaluator
# chi2_distance (helper function)
# subaverages4
# calculate_tau

#imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz
from scipy import special

from fitting_params import *

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

def subaverages4(mjdi, data, freqsm, plot = False):
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

    subs = np.zeros((len(freqsm)//4,2048))
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
    if phase_bins != 2048:
        subs_time_avg = np.zeros((len(freqsm)//4,phase_bins))

        for i in range(len(freqsm)//4):
            for ii in range(phase_bins):
                subs_time_avg[i][ii] = np.average(subs[i][((2048//phase_bins)*ii):((2048//phase_bins)*ii)+(2048//phase_bins)])
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
