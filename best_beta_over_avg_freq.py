"""
Created July 2023
@author: Abra Geiger abrageiger7

Calculations of Best Fit Beta for each frequency Over all MJD averaged

Only 200 pbf to choose from when ran
- set fitting_params.py in this way

Set gwidth to 91 microseconds (index 12 when 50 gwidths)
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
import matplotlib.ticker as tick
from astropy.time import Time
import pickle
from pypulse.singlepulse import SinglePulse


from profile_class_sband_intrinsic import Profile_Intrinss as pcs
from fit_functions import *


#sband intrinsic convolved_profiles
with open(f'sband_intrins_convolved_profiles_phasebins={phase_bins}.pkl', 'rb') as fp:
    sband_intrins_convolved_profiles = pickle.load(fp)


def chi2_distance(A, B, subt_deg_of_freedom):

    '''Takes two vectors and calculates their comparative chi-squared value

    Pre-condition: A and B are 2 vectors of the same length and num_fitted (int) is
    the number of fitted parameters for dividing by the number of degrees of freedom.
    subt_deg_of_freedom is the amount of degrees of freedom subtracted from the length
    of the compared data arrays.

    Returns chi-squared value'''

    squared_residuals = []
    for (a, b) in zip(A, B):
        sq_res = (a-b)**2
        squared_residuals.append(sq_res)

    squared_residuals = np.array(squared_residuals)
    chi_squared = np.sum(squared_residuals) / (len(squared_residuals) - subt_deg_of_freedom)

    return(chi_squared)

def fit_sing(profile, template, num_par, plot = False):
    '''Fits a data profile to a template

    Pre-conditions:
    profile (numpy array): the template
    num_par (int): the number of fitted parameters

    Returns the fit chi-squared value (float)'''

    xind = np.where(profile == np.max(profile))[0][0]

    templatef = template / np.max(template) #fitPulse requires template height of one
    z = np.max(templatef)
    zind = np.where(templatef == z)[0][0]
    ind_diff = xind-zind
    #this lines the profiles up approximately so that Single Pulse finds the
    #true minimum, not just a local min
    templatef = np.roll(templatef, ind_diff)

    sp = SinglePulse(profile)
    fitting = sp.fitPulse(templatef) #TOA cross-correlation, TOA template
    #matching, scale factor, TOA error, scale factor error, signal to noise
    #ratio, cross-correlation coefficient
    #based on the fitPulse fitting, scale and shift the profile to best fit
    #the inputted data
    #fitPulse figures out the best amplitude itself
    spt = SinglePulse(templatef*fitting[2])
    fitted_template = spt.shiftit(fitting[1])

    if plot == True:

        plt.plot(profile, label = 'data fitted to')
        plt.plot(fitted_template, label = 'best fit template')
        plt.legend()
        plt.title('after scaling amplitude and center')
        plt.show()

    chi_sq_measure = chi2_distance(profile, fitted_template, num_par)

    return(chi_sq_measure, fitting[1]) # chi-sq measure of the fitting and the fitting toa offset * important to
    # see how the different pbfs and intrinsic shapes correspond to intrinsic gaussians at different spots and
    # therefore with different toas

def find_best_template_3D(profile, templates, num_par = 4):
    '''Returns index of the best fit template to the
    profile for the array of templates as arg templates.
    This should predominantly be used for fitting for
    varying pbf widths and intrinsic widths.

    num par of 3 means only fitting for one param in this
    case'''

    chi_sqs_collect = np.zeros(np.shape(templates[:,:,0]))

    ind1 = 0
    for i in templates:

        ind2 = 0
        for ii in i:

            chi_sq = fit_sing(profile,ii,num_par)[0]
            chi_sqs_collect[ind1][ind2] = chi_sq
            ind2+=1

        ind1+=1

    plt.figure(1)
    plt.imshow(chi_sqs_collect)
    plt.colorbar()
    plt.title('$\chi^2$ versus fitting templates')
    plt.ylabel('$\chi^2$')
    plt.show()
    plt.close('all')

    low_chi = find_nearest(chi_sqs_collect,0)[0]
    lsqs_index_1 = np.where((chi_sqs_collect == np.min(chi_sqs_collect)))[0][0]
    lsqs_index_2 = np.where((chi_sqs_collect == np.min(chi_sqs_collect)))[1][0]

    plt.figure(2)
    plt.plot(profile, label = 'data fitted to')
    plt.plot(templates[lsqs_index_1][lsqs_index_2], label = 'best fit template')
    plt.legend()
    plt.title('before scaling amplitude and center')
    plt.show()
    plt.close('all')

    chi_sq, toa_offset = fit_sing(profile, templates[lsqs_index_1][lsqs_index_2], num_par, plot = True)

    return(lsqs_index_1, lsqs_index_2, toa_offset)

#import data
with open('j1903_average_profiles_lband.pkl', 'rb') as fp:
    data = pickle.load(fp)

data_collect = {}

mjd = 0.0
freqs = np.array([1175,1275,1375,1475,1575,1675,1775])
dur = 0.0

beta_collect = np.zeros(np.size(freqs))

pbfwidth_collect = np.zeros(np.size(freqs))

iwidth = 101.0

iwidth_ind = find_nearest(gauss_fwhm, iwidth)[1][0][0]

for i in range(np.size(freqs)):

    subs_time_avg = np.zeros(phase_bins)

    for ii in range(np.size(subs_time_avg)):
            subs_time_avg[ii] = np.average(data[i][((2048//phase_bins)*ii):((2048//phase_bins)*(ii+1))])

    plt.figure(1)
    plt.plot(subs_time_avg)
    plt.show()
    plt.close('all')
    lsqs_index_1, lsqs_index_2, toa_offset = find_best_template_3D(subs_time_avg, sband_intrins_convolved_profiles['beta'][:,:,iwidth_ind], num_par = 4)

    beta = betaselect[lsqs_index_1]
    beta_collect[i] = beta

    pbfwidth = widths[lsqs_index_2]
    pbfwidth_collect[i] = pbfwidth

plt.figure(1)

plt.plot(freqs, beta_collect, '.')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Best Fit Beta')
plt.title('Best Beta for All Data Freq Avg')
plt.savefig(f'best_beta_for_all_lband_data_averaged_per_freq_setiwidth={int(np.round(gauss_fwhm[iwidth_ind]))}.pdf')
plt.show()

plt.close('all')

plt.figure(2)

plt.plot(freqs, pbfwidth_collect, '.')
plt.xlabel('Frequency [MHz]')
plt.ylabel('Best Fit PBF Width')
plt.title('Best PBF Width for All Data Freq Avg')
plt.savefig(f'best_pbfwidth_for_all_lband_data_averaged_per_freq_setiwidth={int(np.round(gauss_fwhm[iwidth_ind]))}.pdf')
plt.show()

plt.close('all')
