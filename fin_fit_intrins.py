"""
Created April 2023
@author: Abra Geiger abrageiger7

Time and Frequency Varying Calculations of Fit Parameters
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
import matplotlib.ticker as tick
from astropy.time import Time
import pickle

from profile_class_sband_intrinsic import Profile_Intrinss as pcs
from fit_functions import *


#import data
with open('j1903_data.pkl', 'rb') as fp:
    data_dict = pickle.load(fp)

mjd_strings = list(data_dict.keys())
mjds = np.zeros(np.size(mjd_strings))
for i in range(np.size(mjd_strings)):
    mjds[i] = data_dict[mjd_strings[i]]['mjd']

#TO DO: intrinsic pulse shape convolution -> not really valid because even highest
#frequency is scattered

#TO DO: fix in previous code difference between beta power law index and beta for pbfs

#Below are various calculations of fit parameters using the Profile class and
#functions from fit_functions.py

def power_laws_and_plots(beta_ind, beta_gwidth_ind, exp_gwidth_ind):
    '''Calculates tau with error for each pulse for J1903 L-band data given the
    specified beta_ind and gwidth_ind. Then calculates the best fit tau versus
    frequency power law and plots. Does this for both dec exp pbfs and beta pbfs.

    Other plots include:
    - tau_0 versus mjd, alpha (index of tau versus freq power law) versus mjd
    - histograms and summed likelihoods (essentially histograms with error)
      of the tau_0 and alpha values
    - autocorrelation plots of tau_0 and alpha over mjd
    - tau_0 versus alpha with correlation coefficient'''

    #set the tested slope and yint vals:
    num_slope = 700
    num_yint = 700

    ref_freq = 1.5

    slope_test = np.linspace(-8.0, -0.5, num = num_slope)
    yint_test = np.linspace(30.0, 120.0, num = num_yint)

    #to collect the likelihood distributions for each mjd's fit
    likelihood_slopeb = np.zeros((np.size(mjds),num_slope))
    likelihood_yintb = np.zeros((np.size(mjds),num_yint))

    likelihood_slopee = np.zeros((np.size(mjds),num_slope))
    likelihood_yinte = np.zeros((np.size(mjds),num_yint))

    #to collect the power law data - slopes and yints with error
    plaw_datab = np.zeros((np.size(mjds),6)) #slope, low err, high err, yint, low err, high err
    plaw_datae = np.zeros((np.size(mjds),6)) #slope, low err, high err, yint, low err, high err

    plt.figure(1)
    figb, axsb = plt.subplots(nrows=7, ncols=4, sharex=True, sharey=True, figsize = (8.27,11.69))

    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    #dece_gwidth_pwr_ind = 0.9 #WILL HAVE TO CHANGE

    for i in range(28):

        '''First calculate and collect tau values with errors'''
        print(f'MJD {i}')
        mjd = data_dict[mjd_strings[i]]['mjd']
        data = data_dict[mjd_strings[i]]['data']
        freqs = data_dict[mjd_strings[i]]['freqs']
        dur = data_dict[mjd_strings[i]]['dur']

        p = pcs(mjd, data, freqs, dur)

        freq_list = np.zeros(p.num_sub)

        tau_listb = np.zeros(p.num_sub)
        tau_low_listb = np.zeros(p.num_sub)
        tau_high_listb = np.zeros(p.num_sub)
        fse_listb = np.zeros(p.num_sub)

        for ii in range(p.num_sub):

            print(f'Frequency {ii}')

            datab = p.fit(ii, 'beta', bzeta_ind = beta_ind, iwidth_ind = beta_gwidth_ind)
            tau_listb[ii] = datab['tau_fin']
            tau_low_listb[ii] = datab['tau_low']
            tau_high_listb[ii] = datab['tau_up']
            fse_listb[ii] = datab['fse_effect']

            freq_list[ii] = p.freq_suba

        tau_low_listb = np.sqrt(np.array(tau_low_listb)**2+np.array(fse_listb)**2)
        tau_high_listb = np.sqrt(np.array(tau_high_listb)**2+np.array(fse_listb)**2)

        #convert errors to log space
        total_low_erra = tau_low_listb / (np.array(tau_listb)*math.log(10.0))
        total_high_erra = tau_high_listb / (np.array(tau_listb)*math.log(10.0))

        #convert freqs and taus to log space
        logfreqs = []
        for d in freq_list:
            logfreqs.append(math.log10(d/1000.0)) #convert freqs to GHz
        logtaus = []
        for d in tau_listb:
            logtaus.append(math.log10(d))
        logfreqs = np.array(logfreqs)
        logtaus = np.array(logtaus)

        #calculate the chi-squared surface for the varying slopes and yints
        chisqs = np.zeros((len(slope_test), len(yint_test)))
        for ii, n in enumerate(yint_test):
            for iz, w in enumerate(slope_test):
                error_above_vs_below = np.zeros(np.size(logtaus))
                for iii in np.arange(np.size(logtaus)):
                    if logtaus[iii] > (w * (np.subtract(logfreqs[iii],math.log10(ref_freq))) + math.log10(n)):
                        error_above_vs_below[iii] = total_low_erra[iii]
                    elif logtaus[iii] < (w * (np.subtract(logfreqs[iii],math.log10(ref_freq))) + math.log10(n)):
                        error_above_vs_below[iii] = total_high_erra[iii]
                yphi = (logtaus - (w * (np.subtract(logfreqs,math.log10(ref_freq))) + math.log10(n))) /  error_above_vs_below # subtract so ref freq GHz y-int
                yphisq = yphi ** 2
                yphisq2sum = sum(yphisq)
                chisqs[iz,ii] = yphisq2sum

        chisqs = chisqs - np.amin(chisqs)
        chisqs = np.exp((-0.5)*chisqs)

        probabilitiesx = np.sum(chisqs, axis=1)
        probabilitiesy = np.sum(chisqs, axis=0)
        likelihoodx = likelihood_evaluator(slope_test, probabilitiesx)
        likelihoody = likelihood_evaluator(yint_test, probabilitiesy)

        likelihood_slopeb[i] = probabilitiesx
        likelihood_yintb[i] = probabilitiesy

        slope = likelihoodx[0]
        yint = likelihoody[0]

        plaw_datab[i][0] = slope
        plaw_datab[i][1] = likelihoodx[1]
        plaw_datab[i][2] = likelihoodx[2]
        plaw_datab[i][3] = yint
        plaw_datab[i][4] = likelihoody[1]
        plaw_datab[i][5] = likelihoody[2]

        axsb.flat[i].loglog()
        y = ((np.subtract(logfreqs,math.log10(ref_freq)))*likelihoodx[0]) + math.log10(likelihoody[0])
        axsb.flat[i].errorbar(x = freq_list/1000.0, y = tau_listb, yerr = [total_low_erra, total_high_erra], fmt = '.', color = '0.50', elinewidth = 0.78, ms = 4.5)
        textstr = '\n'.join((
        r'$\mathrm{MJD}=%.0f$' % (int(mjd), ),
        r'$\tau_0=%.1f$' % (yint, ),
        r'$\alpha=%.2f$' % (slope, )))
        axsb.flat[i].text(0.65, 0.95, textstr, fontsize=5, verticalalignment='top', transform=axsb.flat[i].transAxes)
        axsb.flat[i].plot(freq_list/1000.0, 10**y, color = 'dimgrey', linewidth = .8)
        axsb.flat[i].tick_params(axis='x', labelsize='x-small')
        axsb.flat[i].tick_params(axis='y', labelsize='x-small')
        axsb.flat[i].xaxis.set_minor_formatter(tick.ScalarFormatter())
        axsb.flat[i].yaxis.set_minor_formatter(tick.ScalarFormatter())
        plt.rc('font', family = 'serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')

    ax = figb.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel(r'$\nu$ (GHz)')
    ax.set_ylabel(r'$\tau$ ($\mu$s)')
    ax.set_title('Beta = 3.99999 J1903+0327 Scattering vs. Epoch')
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    title = f'timea_power_laws_intrinss_betaind={beta_ind}_gwidth=ind{beta_gwidth_ind}_1.pdf'
    figb.savefig(title)
    figb.show()
    plt.close('all')

    #now do for the same range with dec exp
    plt.figure(2)
    fige, axse = plt.subplots(nrows=7, ncols=4, sharex=True, sharey=True, figsize = (8.27,11.69))

    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    for i in range(28):

        '''First calculate and collect tau values with errors.'''
        print(f'MJD {i}')
        mjd = data_dict[mjd_strings[i]]['mjd']
        data = data_dict[mjd_strings[i]]['data']
        freqs = data_dict[mjd_strings[i]]['freqs']
        dur = data_dict[mjd_strings[i]]['dur']

        p = pcs(mjd, data, freqs, dur)

        freq_list = np.zeros(p.num_sub)

        tau_liste = np.zeros(p.num_sub)
        tau_low_liste = np.zeros(p.num_sub)
        tau_high_liste = np.zeros(p.num_sub)
        fse_liste = np.zeros(p.num_sub)

        for ii in range(p.num_sub):

            print(f'Frequency {ii}')

            datae = p.fit(ii, 'exp', iwidth_ind = beta_gwidth_ind)
            tau_liste[ii] = datae['tau_fin']
            tau_low_liste[ii] = datae['tau_low']
            tau_high_liste[ii] = datae['tau_up']
            fse_liste[ii] = datae['fse_effect']

            freq_list[ii] = p.freq_suba

        tau_low_liste = np.sqrt(tau_low_liste**2+fse_liste**2)
        tau_high_liste = np.sqrt(tau_high_liste**2+fse_liste**2)

        #convert errors to log space
        total_low_erra = tau_low_liste / (np.array(tau_liste)*math.log(10.0))
        total_high_erra = tau_high_liste / (np.array(tau_liste)*math.log(10.0))

        #convert freqs and taus to log space
        logfreqs = []
        for d in freq_list:
            logfreqs.append(math.log10(d/1000.0)) #convert freqs to GHz
        logtaus = []
        for d in tau_liste:
            logtaus.append(math.log10(d))
        logfreqs = np.array(logfreqs)
        logtaus = np.array(logtaus)

        #calculate the chi-squared surface for the varying slopes and yints
        chisqs = np.zeros((len(slope_test), len(yint_test)))
        for ii, n in enumerate(yint_test):
            for iz, w in enumerate(slope_test):
                error_above_vs_below = np.zeros(np.size(logtaus))
                for iii in np.arange(np.size(logtaus)):
                    if logtaus[iii] > (w * (np.subtract(logfreqs[iii],math.log10(ref_freq))) + math.log10(n)):
                        error_above_vs_below[iii] = total_low_erra[iii]
                    elif logtaus[iii] < (w * (np.subtract(logfreqs[iii],math.log10(ref_freq))) + math.log10(n)):
                        error_above_vs_below[iii] = total_high_erra[iii]
                yphi = (logtaus - (w * (np.subtract(logfreqs,math.log10(ref_freq))) + math.log10(n))) / error_above_vs_below # subtract so ref_freq GHz y-int
                yphisq = yphi ** 2
                yphisq2sum = sum(yphisq)
                chisqs[iz,ii] = yphisq2sum

        chisqs = chisqs - np.amin(chisqs)
        chisqs = np.exp((-0.5)*chisqs)

        probabilitiesx = np.sum(chisqs, axis=1)
        probabilitiesy = np.sum(chisqs, axis=0)
        likelihoodx = likelihood_evaluator(slope_test, probabilitiesx)
        likelihoody = likelihood_evaluator(yint_test, probabilitiesy)

        likelihood_slopee[i] = probabilitiesx
        likelihood_yinte[i] = probabilitiesy

        slope = likelihoodx[0]
        yint = likelihoody[0]

        plaw_datae[i][0] = slope
        plaw_datae[i][1] = likelihoodx[1]
        plaw_datae[i][2] = likelihoodx[2]
        plaw_datae[i][3] = yint
        plaw_datae[i][4] = likelihoody[1]
        plaw_datae[i][5] = likelihoody[2]

        axse.flat[i].loglog()
        y = ((np.subtract(logfreqs,math.log10(ref_freq)))*likelihoodx[0]) + math.log10(likelihoody[0])
        axse.flat[i].errorbar(x = freq_list/1000.0, y = tau_liste, yerr = [total_low_erra, total_high_erra], fmt = '.', color = '0.50', elinewidth = 0.78, ms = 4.5)
        textstr = '\n'.join((
        r'$\mathrm{MJD}=%.0f$' % (int(mjd), ),
        r'$\tau_0=%.1f$' % (yint, ),
        r'$\alpha=%.2f$' % (slope, )))
        axse.flat[i].text(0.65, 0.95, textstr, fontsize=5, verticalalignment='top', transform=axse.flat[i].transAxes)
        axse.flat[i].plot(freq_list/1000.0, 10**y, color = 'dimgrey', linewidth = .8)
        axse.flat[i].tick_params(axis='x', labelsize='x-small')
        axse.flat[i].tick_params(axis='y', labelsize='x-small')
        axse.flat[i].xaxis.set_minor_formatter(tick.ScalarFormatter())
        axse.flat[i].yaxis.set_minor_formatter(tick.ScalarFormatter())
        plt.rc('font', family = 'serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')

    ax = fige.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel(r'$\nu$ (GHz)')
    ax.set_ylabel(r'$\tau$ ($\mu$s)')
    ax.set_title('Decaying Exponential J1903+0327 Scattering vs. Epoch')
    title = f'timea_power_laws_intrinss_exponential_gwidth=ind{exp_gwidth_ind}_1.pdf'
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fige.savefig(title)
    plt.close('all')


    #now do again for the remaining 28 mjds
    plt.figure(3)
    figb2, axsb2 = plt.subplots(nrows=7, ncols=4, sharex=True, sharey=True, figsize = (8.27,11.69))

    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    for i in range(28,56):

        '''First calculate and collect tau values with errors.'''
        print(f'MJD {i}')
        mjd = data_dict[mjd_strings[i]]['mjd']
        data = data_dict[mjd_strings[i]]['data']
        freqs = data_dict[mjd_strings[i]]['freqs']
        dur = data_dict[mjd_strings[i]]['dur']

        p = pcs(mjd, data, freqs, dur)

        freq_list = np.zeros(p.num_sub)

        tau_listb = np.zeros(p.num_sub)
        tau_low_listb = np.zeros(p.num_sub)
        tau_high_listb = np.zeros(p.num_sub)
        fse_listb = np.zeros(p.num_sub)

        for ii in range(p.num_sub):

            print(f'Frequency {ii}')

            datab = p.fit(ii, 'beta', bzeta_ind = beta_ind, iwidth_ind = beta_gwidth_ind)
            tau_listb[ii] = datab['tau_fin']
            tau_low_listb[ii] = datab['tau_low']
            tau_high_listb[ii] = datab['tau_up']
            fse_listb[ii] = datab['fse_effect']

            freq_list[ii] = p.freq_suba

        tau_low_listb = np.sqrt(np.array(tau_low_listb)**2+np.array(fse_listb)**2)
        tau_high_listb = np.sqrt(np.array(tau_high_listb)**2+np.array(fse_listb)**2)

        #convert errors to log space
        total_low_erra = tau_low_listb / (np.array(tau_listb)*math.log(10.0))
        total_high_erra = tau_high_listb / (np.array(tau_listb)*math.log(10.0))

        #convert freqs and taus to log space
        logfreqs = []
        for d in freq_list:
            logfreqs.append(math.log10(d/1000.0)) #convert freqs to GHz
        logtaus = []
        for d in tau_listb:
            logtaus.append(math.log10(d))
        logfreqs = np.array(logfreqs)
        logtaus = np.array(logtaus)

        #calculate the chi-squared surface for the varying slopes and yints
        chisqs = np.zeros((len(slope_test), len(yint_test)))
        for ii, n in enumerate(yint_test):
            for iz, w in enumerate(slope_test):
                error_above_vs_below = np.zeros(np.size(logtaus))
                for iii in np.arange(np.size(logtaus)):
                    if logtaus[iii] > (w * (np.subtract(logfreqs[iii],math.log10(ref_freq))) + math.log10(n)):
                        error_above_vs_below[iii] = total_low_erra[iii]
                    elif logtaus[iii] < (w * (np.subtract(logfreqs[iii],math.log10(ref_freq))) + math.log10(n)):
                        error_above_vs_below[iii] = total_high_erra[iii]
                yphi = (logtaus - (w * (np.subtract(logfreqs,math.log10(ref_freq))) + math.log10(n))) /  error_above_vs_below # subtract so ref_freq GHz y-int
                yphisq = yphi ** 2
                yphisq2sum = sum(yphisq)
                chisqs[iz,ii] = yphisq2sum

        chisqs = chisqs - np.amin(chisqs)
        chisqs = np.exp((-0.5)*chisqs)

        probabilitiesx = np.sum(chisqs, axis=1)
        probabilitiesy = np.sum(chisqs, axis=0)
        likelihoodx = likelihood_evaluator(slope_test, probabilitiesx)
        likelihoody = likelihood_evaluator(yint_test, probabilitiesy)

        likelihood_slopeb[i] = probabilitiesx
        likelihood_yintb[i] = probabilitiesy

        slope = likelihoodx[0]
        yint = likelihoody[0]

        plaw_datab[i][0] = slope
        plaw_datab[i][1] = likelihoodx[1]
        plaw_datab[i][2] = likelihoodx[2]
        plaw_datab[i][3] = yint
        plaw_datab[i][4] = likelihoody[1]
        plaw_datab[i][5] = likelihoody[2]

        i -= 28
        axsb2.flat[i].loglog()
        y = ((np.subtract(logfreqs,math.log10(ref_freq)))*likelihoodx[0]) + math.log10(likelihoody[0])
        axsb2.flat[i].errorbar(x = freq_list/1000.0, y = tau_listb, yerr = [total_low_erra, total_high_erra], fmt = '.', color = '0.50', elinewidth = 0.78, ms = 4.5)
        textstr = '\n'.join((
        r'$\mathrm{MJD}=%.0f$' % (int(mjd), ),
        r'$\tau_0=%.1f$' % (yint, ),
        r'$\alpha=%.2f$' % (slope, )))
        axsb2.flat[i].text(0.65, 0.95, textstr, fontsize=5, verticalalignment='top', transform=axsb2.flat[i].transAxes)
        axsb2.flat[i].plot(freq_list/1000.0, 10**y, color = 'dimgrey', linewidth = .8)
        axsb2.flat[i].tick_params(axis='x', labelsize='x-small')
        axsb2.flat[i].tick_params(axis='y', labelsize='x-small')
        axsb2.flat[i].xaxis.set_minor_formatter(tick.ScalarFormatter())
        axsb2.flat[i].yaxis.set_minor_formatter(tick.ScalarFormatter())
        plt.rc('font', family = 'serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')


    ax = figb2.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel(r'$\nu$ (GHz)')
    ax.set_ylabel(r'$\tau$ ($\mu$s)')
    ax.set_title('Beta = 3.99999 J1903+0327 Scattering vs. Epoch')
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    title = f'timea_power_laws_intrinss_betaind={beta_ind}_gwidth=ind{beta_gwidth_ind}_2.pdf'
    figb2.savefig(title)
    plt.close('all')


    #now do for the same range with dec exp
    plt.figure(4)
    fige2, axse2 = plt.subplots(nrows=7, ncols=4, sharex=True, sharey=True, figsize = (8.27,11.69))

    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    for i in range(28,56):

        '''First calculate and collect tau values with errors.'''
        print(f'MJD {i}')
        mjd = data_dict[mjd_strings[i]]['mjd']
        data = data_dict[mjd_strings[i]]['data']
        freqs = data_dict[mjd_strings[i]]['freqs']
        dur = data_dict[mjd_strings[i]]['dur']

        p = pcs(mjd, data, freqs, dur)

        freq_list = np.zeros(p.num_sub)

        tau_liste = np.zeros(p.num_sub)
        tau_low_liste = np.zeros(p.num_sub)
        tau_high_liste = np.zeros(p.num_sub)
        fse_liste = np.zeros(p.num_sub)

        for ii in range(p.num_sub):

            print(f'Frequency {ii}')

            datae = p.fit(ii, 'exp', iwidth_ind = beta_gwidth_ind)
            tau_liste[ii] = datae['tau_fin']
            tau_low_liste[ii] = datae['tau_low']
            tau_high_liste[ii] = datae['tau_up']
            fse_liste[ii] = datae['fse_effect']

            freq_list[ii] = p.freq_suba

        tau_low_liste = np.sqrt(tau_low_liste**2+fse_liste**2)
        tau_high_liste = np.sqrt(tau_high_liste**2+fse_liste**2)

        #convert errors to log space
        total_low_erra = tau_low_liste / (np.array(tau_liste)*math.log(10.0))
        total_high_erra = tau_high_liste / (np.array(tau_liste)*math.log(10.0))

        #convert freqs and taus to log space
        logfreqs = []
        for d in freq_list:
            logfreqs.append(math.log10(d/1000.0)) #convert freqs to GHz
        logtaus = []
        for d in tau_liste:
            logtaus.append(math.log10(d))
        logfreqs = np.array(logfreqs)
        logtaus = np.array(logtaus)

        #calculate the chi-squared surface for the varying slopes and yints
        chisqs = np.zeros((len(slope_test), len(yint_test)))
        for ii, n in enumerate(yint_test):
            for iz, w in enumerate(slope_test):
                error_above_vs_below = np.zeros(np.size(logtaus))
                for iii in np.arange(np.size(logtaus)):
                    if logtaus[iii] > (w * (np.subtract(logfreqs[iii],math.log10(ref_freq))) + math.log10(n)):
                        error_above_vs_below[iii] = total_low_erra[iii]
                    elif logtaus[iii] < (w * (np.subtract(logfreqs[iii],math.log10(ref_freq))) + math.log10(n)):
                        error_above_vs_below[iii] = total_high_erra[iii]
                yphi = (logtaus - (w * (np.subtract(logfreqs,math.log10(ref_freq))) + math.log10(n))) / error_above_vs_below # subtract so ref_freq GHz y-int
                yphisq = yphi ** 2
                yphisq2sum = sum(yphisq)
                chisqs[iz,ii] = yphisq2sum

        chisqs = chisqs - np.amin(chisqs)
        chisqs = np.exp((-0.5)*chisqs)

        probabilitiesx = np.sum(chisqs, axis=1)
        probabilitiesy = np.sum(chisqs, axis=0)
        likelihoodx = likelihood_evaluator(slope_test, probabilitiesx)
        likelihoody = likelihood_evaluator(yint_test, probabilitiesy)

        likelihood_slopee[i] = probabilitiesx
        likelihood_yinte[i] = probabilitiesy

        slope = likelihoodx[0]
        yint = likelihoody[0]

        plaw_datae[i][0] = slope
        plaw_datae[i][1] = likelihoodx[1]
        plaw_datae[i][2] = likelihoodx[2]
        plaw_datae[i][3] = yint
        plaw_datae[i][4] = likelihoody[1]
        plaw_datae[i][5] = likelihoody[2]

        i -= 28
        axse2.flat[i].loglog()
        y = ((np.subtract(logfreqs,math.log10(ref_freq)))*likelihoodx[0]) + math.log10(likelihoody[0])
        axse2.flat[i].errorbar(x = freq_list/1000.0, y = tau_liste, yerr = [total_low_erra, total_high_erra], fmt = '.', color = '0.50', elinewidth = 0.78, ms = 4.5)
        textstr = '\n'.join((
        r'$\mathrm{MJD}=%.0f$' % (int(mjd), ),
        r'$\tau_0=%.1f$' % (yint, ),
        r'$\alpha=%.2f$' % (slope, )))
        axse2.flat[i].text(0.65, 0.95, textstr, fontsize=5, verticalalignment='top', transform=axse2.flat[i].transAxes)
        axse2.flat[i].plot(freq_list/1000.0, 10**y, color = 'dimgrey', linewidth = .8)
        axse2.flat[i].tick_params(axis='x', labelsize='x-small')
        axse2.flat[i].tick_params(axis='y', labelsize='x-small')
        axse2.flat[i].xaxis.set_minor_formatter(tick.ScalarFormatter())
        axse2.flat[i].yaxis.set_minor_formatter(tick.ScalarFormatter())
        plt.rc('font', family = 'serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')


    ax = fige2.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel(r'$\nu$ (GHz)')
    ax.set_ylabel(r'$\tau$ ($\mu$s)')
    ax.set_title('Decaying Exponential J1903+0327 Scattering vs. Epoch')
    title = f'timea_power_laws_intrinss_exponential_gwidth=ind{exp_gwidth_ind}_2.pdf'
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    fige2.savefig(title)
    plt.close('all')

    np.save(f'timea_powerlaw_data_intrinss_betaind{beta_ind}_betagwidthind{beta_gwidth_ind}_expgwidthind{exp_gwidth_ind}', [plaw_datab,plaw_datae])

    #now plot slopes and yints over time

    plt.figure(5)

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize = (11,5), sharex = True)
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=15)

    markers, caps, bars = axs.flat[0].errorbar(x = mjds, y = plaw_datae[:,0], yerr = [plaw_datae[:,1], plaw_datae[:,2]], fmt = 'o', ms = 5, color = 'g', capsize = 2, label = 'Exponential PBF')

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    markers, caps, bars = axs.flat[0].errorbar(x = mjds, y = plaw_datab[:,0], yerr = [plaw_datab[:,1], plaw_datab[:,2]], fmt = 's', color = 'dimgrey', capsize = 2, label = r'$\beta = 3.99999$ PBF')
    axs.flat[0].set_ylabel(r'$\alpha$', fontsize = 14)

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    axs.flat[0].legend(loc = 'upper right', fontsize = 14, bbox_to_anchor = (1.2,1))
    axis2 = axs.flat[0].twiny()
    XLIM = axs.flat[0].get_xlim()
    XLIM = list(map(lambda x: Time(x,format='mjd',scale='utc').decimalyear,XLIM))
    axis2.set_xlim(XLIM)
    axis2.set_xlabel('Years', fontsize = 14)
    axis2.tick_params(axis='x')

    axs.flat[1].set_ylabel(r'$\tau_0$ ($\mu$s)', fontsize = 14)
    axs.flat[1].set_xlabel('MJD', fontsize = 14)
    markers, caps, bars = axs.flat[1].errorbar(x = mjds, y = plaw_datae[:,3], yerr = [plaw_datae[:,4], plaw_datae[:,5]], fmt = 'o', ms = 5, color = 'g', capsize = 2, label = 'Exponential PBF')

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    markers, caps, bars = axs.flat[1].errorbar(x = mjds, y = plaw_datab[:,3], yerr = [plaw_datab[:,4], plaw_datab[:,5]], fmt = 's', color = 'dimgrey', capsize = 2, label = r'$\beta$ PBF')

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    fig.tight_layout()
    plt.savefig(f'timea_intrinss_time_scales_intrinss_betaind{beta_ind}_betagwidthind{beta_gwidth_ind}_expgwidthind{exp_gwidth_ind}.pdf')
    plt.close(5)


    #now plot seperate panels for dece and beta
    plt.figure(6)

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize = (12,6), sharex = True)
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=10)

    markers, caps, bars = axs.flat[0].errorbar(x = mjds, y = plaw_datae[:,0], yerr = [plaw_datae[:,1], plaw_datae[:,2]], fmt = 'o', ms = 5, color = 'g', capsize = 2, label = 'Exponential PBF')

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    axs.flat[0].set_ylabel(r'$\alpha$', fontsize = 14)

    axs.flat[0].legend()

    axis2 = axs.flat[0].twiny()
    XLIM = axs.flat[0].get_xlim()
    XLIM = list(map(lambda x: Time(x,format='mjd',scale='utc').decimalyear,XLIM))
    axis2.set_xlim(XLIM)
    axis2.set_xlabel('Years', fontsize = 14)
    axis2.tick_params(axis='x')

    axs.flat[2].set_ylabel(r'$\tau_0$ ($\mu$s)', fontsize = 14)

    markers, caps, bars = axs.flat[2].errorbar(x = mjds, y = plaw_datae[:,3], yerr = [plaw_datae[:,4], plaw_datae[:,5]], fmt = 'o', ms = 5, color = 'g', capsize = 2, label = 'Exponential PBF')

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    markers, caps, bars = axs.flat[1].errorbar(x = mjds, y = plaw_datab[:,0], yerr = [plaw_datab[:,1], plaw_datab[:,2]], fmt = 's', color = 'dimgrey', capsize = 2, label = r'$\beta = 3.99999$ PBF')
    axs.flat[1].set_ylabel(r'$\alpha$', fontsize = 14)

    axs.flat[1].legend()

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    axs.flat[3].set_ylabel(r'$\tau_0$ ($\mu$s)', fontsize = 14)
    markers, caps, bars = axs.flat[3].errorbar(x = mjds, y = plaw_datab[:,3], yerr = [plaw_datab[:,4], plaw_datab[:,5]], fmt = 's', color = 'dimgrey', capsize = 2, label = r'$\beta$ PBF')

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    axs.flat[3].set_xlabel('MJD', fontsize = 14)

    fig.tight_layout()
    plt.savefig(f'timea_intrinss_time_scales_seperate_panels_intrinss_betaind{beta_ind}_betagwidthind{beta_gwidth_ind}_expgwidthind{exp_gwidth_ind}.pdf')
    plt.close(6)

    #autocorrelation
    plt.figure(11)

    #must make sure data is sorted by mjd for autocorrelation
    arr1inds = mjds.argsort()

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize = (12,6), sharex = True)
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=10)

    acor0 = axs.flat[0].acorr(plaw_datae[:,0][arr1inds], label = r'Exponential PBF $\alpha$', maxlags = 55)
    acor1 = axs.flat[1].acorr(plaw_datab[:,0][arr1inds], label = r'Exponential PBF $\tau_0$', maxlags = 55)
    acor2 = axs.flat[2].acorr(plaw_datae[:,3][arr1inds], label = r'Beta = 3.99999 PBF $\alpha$', maxlags = 55)
    acor3 = axs.flat[3].acorr(plaw_datab[:,3][arr1inds], label = r'Beta = 3.99999 PBF $\tau_0$', maxlags = 55)

    fig.tight_layout()
    plt.savefig(f'timea_intrinss_autocorrelation_intrinss_betaind{beta_ind}_betagwidthind{beta_gwidth_ind}_expgwidthind{exp_gwidth_ind}.pdf')
    plt.close(11)


    plt.figure(7)

    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=13)

    #now plot the histograms and summed likeihoods (essentially the histograms w error)
    fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (10, 14), sharex = 'col', sharey = 'row')

    axs.flat[0].hist(plaw_datab[:,0], color = 'dimgrey', alpha = 0.8, bins = 10, label = 'Beta = 3.99999 PBF')
    axs.flat[0].set_ylabel('Counts', fontsize=14)

    axs.flat[2].hist(plaw_datae[:,0], color = 'green', bins = 10, label = 'Exponential PBF')
    axs.flat[2].set_ylabel('Counts', fontsize=14)

    axs.flat[1].hist(plaw_datab[:,3], color = 'dimgrey', alpha = 0.8, bins = 10, label = 'Beta = 3.99999 PBF')

    axs.flat[3].hist(plaw_datae[:,3], color = 'green', bins = 10, label = 'Exponential PBF')
    axs.flat[3].sharex(axs.flat[1])

    fig.tight_layout()

    norm_likelihood_slopeb = np.zeros(np.shape(likelihood_slopeb))
    for i in range(np.size(likelihood_slopeb[:,0])):
        norm_likelihood_slopeb[i] = likelihood_slopeb[i]/ trapz(likelihood_slopeb[i])

    norm_likelihood_yintb = np.zeros(np.shape(likelihood_yintb))
    for i in range(np.size(likelihood_yintb[:,0])):
        norm_likelihood_yintb[i] = likelihood_yintb[i]/ trapz(likelihood_yintb[i])

    norm_likelihood_yinte = np.zeros(np.shape(likelihood_yinte))
    for i in range(np.size(likelihood_yinte[:,0])):
        norm_likelihood_yinte[i] = likelihood_yinte[i]/ trapz(likelihood_yinte[i])

    norm_likelihood_slopee = np.zeros(np.shape(likelihood_slopee))
    for i in range(np.size(likelihood_slopee[:,0])):
        norm_likelihood_slopee[i] = likelihood_slopee[i]/ trapz(likelihood_slopee[i])

    axs.flat[4].plot(slope_test, np.sum(norm_likelihood_slopeb, axis = 0)/trapz(np.sum(norm_likelihood_slopeb, axis = 0)), color = 'dimgrey', alpha = 0.5, label = 'Beta = 3.99999 PBF')
    axs.flat[4].set_ylabel(r'Normalized Integrated Likelihood', fontsize=14)

    axs.flat[5].plot(yint_test, np.sum(norm_likelihood_yintb, axis = 0)/trapz(np.sum(norm_likelihood_yintb, axis = 0)), color = 'dimgrey', alpha = 0.5, label = 'Beta = 3.99999 PBF')

    axs.flat[6].plot(slope_test, np.sum(norm_likelihood_slopee, axis = 0)/trapz(np.sum(norm_likelihood_slopee, axis = 0)), color = 'g', label = 'Exponential PBF')
    axs.flat[6].set_xlabel(r'$\alpha$', fontsize=16)
    axs.flat[6].set_ylabel(r'Normalized Integrated Likelihood', fontsize=14)

    axs.flat[7].plot(yint_test, np.sum(norm_likelihood_yinte, axis = 0)/trapz(np.sum(norm_likelihood_yinte, axis = 0)), color = 'g', label = 'Exponential PBF')
    axs.flat[7].set_xlabel(r'$\tau_0$ ($\mu$s)', fontsize=16)

    axs.flat[0].set_yticks(np.linspace(0,12,5))
    axs.flat[2].set_yticks(np.linspace(0,12,5))

    axs.flat[4].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    axs.flat[6].set_yticks(np.linspace(0,.01,4))
    axs.flat[4].set_yticks(np.linspace(0,.01,4))
    axs.flat[6].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    axs.flat[6].sharey(axs.flat[4])
    axs.flat[0].sharey(axs.flat[2])

    axs.flat[0].legend(loc = 'upper left', fontsize = 8)
    axs.flat[4].legend(loc = 'upper left', fontsize = 8)
    axs.flat[2].legend(loc = 'upper left', fontsize = 8)
    axs.flat[6].legend(loc = 'upper left', fontsize = 8)

    fig.tight_layout()
    plt.savefig(f'timea_intrinss_alpha_tau_hist_intrinss_betaind{beta_ind}_betagwidthind{beta_gwidth_ind}_expgwidthind{exp_gwidth_ind}.pdf')
    plt.close(7)

    #now plot tau_0 versus alpha
    plt.figure(8)
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=10)

    markers, caps, bars = plt.errorbar(x = plaw_datae[:,3], y = plaw_datae[:,0], xerr = [plaw_datae[:,4], plaw_datae[:,5]], yerr = [plaw_datae[:,1], plaw_datae[:,2]], fmt = 'o', ms = 5, color = 'g', capsize = 2, label = 'Exponential PBF')

    [bar.set_alpha(0.2) for bar in bars]
    [cap.set_alpha(0.2) for cap in caps]

    markers, caps, bars = plt.errorbar(x = plaw_datab[:,3], y = plaw_datab[:,0], xerr = [plaw_datab[:,4], plaw_datab[:,5]], yerr = [plaw_datab[:,1], plaw_datab[:,2]], fmt = 's', ms = 5, color = 'dimgrey', capsize = 2, label = 'Beta = 3.99999 PBF')
    plt.xlabel(r'$\tau_0$ ($\mu$s)', fontsize = 14)
    plt.ylabel(r'$\alpha$', fontsize = 14)

    corrb, _ = pearsonr(plaw_datab[:,3], plaw_datab[:,0])
    corre, _ = pearsonr(plaw_datae[:,3], plaw_datae[:,0])

    plt.text(80, -6.5, f'Decaying Exponential r = {np.round(corrb,2)} \nBeta r = {np.round(corre,2)}', bbox=dict(facecolor='none', edgecolor='black'))

    [bar.set_alpha(0.2) for bar in bars]
    [cap.set_alpha(0.2) for cap in caps]
    plt.legend()
    plt.savefig(f'timea_intrinss_tau_vs_alpha_intrinss_betaind{beta_ind}_betagwidthind{beta_gwidth_ind}_expgwidthind{exp_gwidth_ind}.pdf')
    plt.close(8)

    #now plot dec exp versus beta data
    plt.figure(9)

    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=10)

    fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (5,7))

    axs.flat[0].set_title(r'Exponential versus $\beta$ PBF Data')
    markers, caps, bars = axs.flat[0].errorbar(x = plaw_datae[:,3], y = plaw_datab[:,3], xerr = [plaw_datae[:,4], plaw_datae[:,5]], yerr = [plaw_datab[:,4], plaw_datab[:,5]], fmt = 'o', ms = 5, color = 'dimgrey', capsize = 2, label = r'$\tau_0$')

    axs.flat[0].set_xlabel(r'Exponential $\tau_0$ ($\mu$s)')
    axs.flat[0].set_ylabel(r'$\beta$ $\tau_0$ ($\mu$s)')

    [bar.set_alpha(0.2) for bar in bars]
    [cap.set_alpha(0.2) for cap in caps]

    axs.flat[0].legend()

    markers, caps, bars = axs.flat[1].errorbar(x = plaw_datae[:,0], y = plaw_datab[:,0], xerr = [plaw_datae[:,1], plaw_datae[:,2]], yerr = [plaw_datab[:,1], plaw_datab[:,2]], fmt = 's', ms = 5, color = 'dimgrey', capsize = 2, label = r'$\alpha$')

    axs.flat[1].set_xlabel(r'Exponential PBF $\alpha$')
    axs.flat[1].set_ylabel(r'$\beta$ PBF $\alpha$')

    [bar.set_alpha(0.2) for bar in bars]
    [cap.set_alpha(0.2) for cap in caps]

    axs.flat[1].legend()

    corry, _ = pearsonr(plaw_datab[:,3], plaw_datae[:,3])
    corrs, _ = pearsonr(plaw_datab[:,0], plaw_datae[:,0])

    axs.flat[0].text(76, 76, f'r = {np.round(corry,2)}', bbox=dict(facecolor='none', edgecolor='black'))
    axs.flat[1].text(-5, -6.5, f'r = {np.round(corrs,2)}', bbox=dict(facecolor='none', edgecolor='black'))

    fig.tight_layout()
    plt.savefig(f'timea_intrinss_beta_versus_dece_intrinss_betaind{beta_ind}_betagwidthind{beta_gwidth_ind}_expgwidthind{exp_gwidth_ind}.pdf')
    plt.close(9)


power_laws_and_plots(1, 56, 56)
