import sys
sys.path.append('/Users/abrageiger/Documents/research/projects/pbf_fitting')
import numpy as np
import pickle
from fit_functions import *
import matplotlib.ticker as tick
from profile_class import Profile_Fitting
from astropy.time import Time
from scipy.stats import pearsonr
import os

# arguments when running the script:
# 1 - intrinsic shape (str) either 'modeled', 'gaussian', or 'sband_avg'
intrinsic_shape = str(sys.argv[1])
# 2 - beta value (float or int) - from 3.0 - 4.0
beta = float(sys.argv[2])
# 3 - zeta value (float or int) - from 0.0 - 5.0
zeta = float(sys.argv[3])
# 4 - 'thin' or 'thick' medium pbf
screen = str(sys.argv[4])

if intrinsic_shape != 'modeled':
    # 5 - intrinsic width (float or int) for numerical pbf fitting
    iwidth_bzeta = float(sys.argv[5])
    # 6 - intrinsic width (float or int) for decaying exponential pbf fitting
    iwidth_exp = float(sys.argv[6])

# make sure inputted intrinsic shape is valid
if intrinsic_shape != 'gaussian' and intrinsic_shape != 'sband_avg' and intrinsic_shape != 'modeled':
    raise Exception('Please choose a valid intrinsic shape: either gaussian, sband_avg, or modeled.')

if (zeta != 0 and zeta != 0.01) and beta != 3.667:
    raise Exception('Invalid PBF Selected: Zeta greater that zero is only available for Beta = 3.667.')

freqs_arr = np.linspace(1170.0,1770.0,8)


def slopes_at_varying_freqs(freqs, taus):

    slopes = np.zeros(4)

    for i in range(0,8,2):

        freq_copy = np.copy(freqs)

        low_freq_index = find_nearest(freq_copy, freqs_arr[i])[1][0][0]
        freq_copy[low_freq_index] = 0.0
        high_freq_index = find_nearest(freq_copy, freqs_arr[i])[1][0][0]

        low_freq = freqs[low_freq_index]
        high_freq = freqs[high_freq_index]

        low_freq_tau = taus[low_freq_index]
        high_freq_tau = taus[high_freq_index]

        slope  = -(math.log10(high_freq_tau)-math.log10(low_freq_tau))/(math.log10(high_freq)-math.log10(low_freq))
        slopes[i//2] = slope

    return(slopes)

def slopes_at_varying_freqs_plot(slopes, type_test):

    freqs = freqs_arr[::2]

    if type_test == 'zeta':
        label = r'$\zeta$ = '+ str(zetas[bzeta_ind]) + ' PBF '
    elif type_test == 'beta':
        label = r'$\beta$ = '+ str(betas[bzeta_ind]) + ' PBF '
    elif type_test == 'exp':
        label = 'Exponential PBF '

    fig, axs = plt.subplots()
    markers, caps, bars = axs.errorbar(x = freqs, y = np.mean(slopes, axis = 0), yerr = np.std(slopes, axis = 0)/np.sqrt(np.size(slopes[:,0])-1.0), color = 'k', fmt = 'o', ms = 5, capsize = 4, label = label)

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    plt.legend()

    plt.xlabel('Frequency [MHz]')
    plt.ylabel(r'$X_{\tau}$')

    title = plot_title('slopes_at_varying_freqs', type_test)

    plt.savefig(title, bbox_inches='tight')
    plt.close('all')

def tau_vs_freq_pwrlaw(mjd_start_ind, mjd_end_ind, intrinsic_shape, pbf_type, bzeta_ind = -1, iwidth_ind = -1, set_slope = 0):

    '''Calculates best fit powerlaw intercept and slope for given mjd range.
    Uses data dict which is defined below.

    Inputs:
    mjd_start_ind: [int] Starting mjd index
    mjd_end_ind: [int] Ending mjd index (mjds go from 0-56)
    intrinsic_shape: [str] intrinsic shape - 'modeled', 'gaussian', or 'sband_avg'
    pbf_type: [str] pulse broadening function type - 'beta', 'zeta', 'exp'
    bzeta_ind: [int] index in betas above for the beta value to use for the pbfs
    iwidth_ind: [int] index in intrinsic widths - intrinsic_fwhms above - for the i fwhm to use for fitting
    '''

    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    plt.figure(1)
    figb, axsb = plt.subplots(nrows=7, ncols=4, sharex=True, sharey=True, figsize = (8.27,11.69))

    slopes_vs_freqs = np.zeros((mjd_end_ind-mjd_start_ind,4))
    frequency_collection = []
    tau_value_collection = []
    tau_lowerr_value_collection = []
    tau_higherr_value_collection = []

    #to collect the likelihood distributions for each mjd's fit
    likelihood_slope = np.zeros((mjd_end_ind-mjd_start_ind,num_slope))
    likelihood_yint = np.zeros((mjd_end_ind-mjd_start_ind,num_yint))

    #to collect power law slopes, intercepts, and errors
    plaw_data = np.zeros((mjd_end_ind-mjd_start_ind,6))

    mjd_start = data_dict[mjd_strings[mjd_start_ind]]['mjd']
    mjd_end = data_dict[mjd_strings[mjd_end_ind-1]]['mjd']

    inder = 0
    for i in range(mjd_start_ind, mjd_end_ind):

        '''First calculate and collect tau values with errors'''
        print(f'MJD {i}')
        mjd = data_dict[mjd_strings[i]]['mjd']
        data = data_dict[mjd_strings[i]]['data']
        freqs = data_dict[mjd_strings[i]]['freqs']
        dur = data_dict[mjd_strings[i]]['dur']

        prof = Profile_Fitting(mjd, data, freqs, dur, screen, intrinsic_shape, betas, zetas, fitting_profiles, tau_values, intrinsic_fwhms)

        freq_list = np.zeros(prof.num_sub)

        tau_listb = np.zeros(prof.num_sub)
        tau_low_listb = np.zeros(prof.num_sub)
        tau_high_listb = np.zeros(prof.num_sub)
        fse_listb = np.zeros(prof.num_sub)

        for ii in range(prof.num_sub):

            datab = prof.fit(ii, pbf_type, bzeta_ind = bzeta_ind, iwidth_ind = iwidth_ind)

            print(f'Frequency = {prof.freq_round}')

            tau_listb[ii] = datab['tau_fin']
            tau_low_listb[ii] = datab['tau_low']
            tau_high_listb[ii] = datab['tau_up']
            fse_listb[ii] = datab['fse_effect']

            freq_list[ii] = prof.freq_suba

        slopes = slopes_at_varying_freqs(freq_list, tau_listb)
        slopes_vs_freqs[inder] = slopes

        tau_low_listb = np.sqrt(np.array(tau_low_listb)**2+np.array(fse_listb)**2)
        tau_high_listb = np.sqrt(np.array(tau_high_listb)**2+np.array(fse_listb)**2)

        tau_value_collection.append(tau_listb)
        tau_lowerr_value_collection.append(tau_low_listb)
        tau_higherr_value_collection.append(tau_high_listb)
        frequency_collection.append(freq_list)

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

        likelihood_slope[inder] = probabilitiesx
        likelihood_yint[inder] = probabilitiesy

        slope = -likelihoodx[0]
        yint = likelihoody[0]

        plaw_data[inder][0] = slope
        plaw_data[inder][1] = likelihoodx[1]
        plaw_data[inder][2] = likelihoodx[2]
        plaw_data[inder][3] = yint
        plaw_data[inder][4] = likelihoody[1]
        plaw_data[inder][5] = likelihoody[2]

        axsb.flat[inder].loglog()
        y = ((np.subtract(logfreqs,math.log10(ref_freq)))*likelihoodx[0]) + math.log10(likelihoody[0])
        axsb.flat[inder].errorbar(x = freq_list/1000.0, y = tau_listb, yerr = [total_low_erra, total_high_erra], fmt = '.', color = '0.50', elinewidth = 0.78, ms = 4.5)
        textstr = '\n'.join((
        r'$\mathrm{MJD}=%.0f$' % (int(mjd), ),
        r'$\tau_0=%.1f$' % (yint, ),
        r'$X_{\tau}=%.2f$' % (slope, )))
        axsb.flat[inder].text(0.65, 0.95, textstr, fontsize=5, verticalalignment='top', transform=axsb.flat[inder].transAxes)
        axsb.flat[inder].plot(freq_list/1000.0, 10.0**y, color = 'dimgrey', linewidth = .8)

        #tick parameters
        axsb.flat[inder].tick_params(axis='x', labelsize='x-small')
        axsb.flat[inder].tick_params(axis='y', labelsize='x-small')
        axsb.flat[inder].set_yticks([])
        axsb.flat[inder].set_xticks([])
        for ticking in axsb.flat[inder].yaxis.get_minor_ticks():
            ticking.label1.set_visible(False)
        axsb.flat[inder].set_yticks([25,50,100,300])
        axsb.flat[inder].xaxis.set_major_formatter(tick.ScalarFormatter())
        axsb.flat[inder].yaxis.set_major_formatter(tick.ScalarFormatter())
        axsb.flat[inder].xaxis.set_minor_formatter(tick.ScalarFormatter())
        axsb.flat[inder].yaxis.set_minor_formatter(tick.ScalarFormatter())
        axsb.flat[inder].get_xaxis().get_major_formatter().labelOnlyBase = True
        axsb.flat[inder].get_yaxis().get_major_formatter().labelOnlyBase = True
        axsb.flat[inder].get_xaxis().get_minor_formatter().labelOnlyBase = True
        axsb.flat[inder].get_yaxis().get_minor_formatter().labelOnlyBase = True

        plt.rc('font', family = 'serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')
        inder += 1

    ax = figb.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel(r'$\nu$ (GHz)')
    ax.set_ylabel(r'$\tau$ ($\mu$s)')
    if pbf_type == 'beta':
        ax.set_title(r'J1903+0327 Scattering vs. Epoch; $\beta$ = '+ f'{betas[bzeta_ind]}' + r'; $\zeta$ = 0.0')
    elif pbf_type == 'zeta':
        ax.set_title(r'J1903+0327 Scattering vs. Epoch; $\zeta$ = '+ f'{zetas[bzeta_ind]}' + r'; $\beta$ = 11/3')
    elif pbf_type == 'exp':
        ax.set_title('J1903+0327 Scattering vs. Epoch; Exponential PBF')
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    if intrinsic_shape != 'modeled':
        if pbf_type == 'beta':
            title = f'power_laws|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}|MEDIUM={screen}|IWIDTH={intrinsic_fwhms[iwidth_ind]}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'zeta':
            title = f'power_laws|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}|MEDIUM={screen}|IWIDTH={intrinsic_fwhms[iwidth_ind]}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'exp':
            title = f'power_laws|{intrinsic_shape.upper()}|{pbf_type.upper()}|MEDIUM={screen}|IWIDTH={intrinsic_fwhms[iwidth_ind]}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'

    elif intrinsic_shape == 'modeled':
        if pbf_type == 'beta':
            title = f'power_laws|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}|MEDIUM={screen}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'zeta':
            title = f'power_laws|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}|MEDIUM={screen}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'exp':
            title = f'power_laws|{intrinsic_shape.upper()}|{pbf_type.upper()}|MEDIUM={screen}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'

    figb.savefig(title)
    figb.show()
    plt.close('all')

    return(plaw_data, likelihood_slope, likelihood_yint, slopes_vs_freqs, tau_value_collection, tau_lowerr_value_collection, tau_higherr_value_collection, frequency_collection)

def plot_title(plot_title, pbf_type):

    '''Plot title is string to indentify plot, pbf_type is string of pbf type'''

    if intrinsic_shape != 'modeled':
        if pbf_type == 'beta':
            title = f'{plot_title}|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}|MEDIUM={screen}|IWIDTH_DIS={intrinsic_fwhms[iwidth_ind_bzeta]}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'zeta':
            title = f'{plot_title}|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}|MEDIUM={screen}|IWIDTH_DIS={intrinsic_fwhms[iwidth_ind_bzeta]}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'exp':
            title = f'{plot_title}|{intrinsic_shape.upper()}|{pbf_type.upper()}|MEDIUM={screen}|IWIDTH_EXP={intrinsic_fwhms[iwidth_ind_exp]}|TIME_AVG_EVERY={time_avg_factor}.pdf'

    elif intrinsic_shape == 'modeled':
        if pbf_type == 'beta':
            title = f'{plot_title}|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}|MEDIUM={screen}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'zeta':
            title = f'{plot_title}|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}|MEDIUM={screen}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'exp':
            title = f'{plot_title}|{intrinsic_shape.upper()}|{pbf_type.upper()}|MEDIUM={screen}|TIME_AVG_EVERY={time_avg_factor}.pdf'

    return(title)


def time_scales(plaw_data, pbf_type):

    plt.figure(1)

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize = (12,5), sharex = True)
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='xx-large')
    plt.rc('ytick', labelsize='xx-large')

    if pbf_type == 'zeta':
        label = r'$\zeta$ = '+ str(zetas[bzeta_ind]) + ' PBF'
    elif pbf_type == 'beta':
        label = r'$\beta$ = '+ str(betas[bzeta_ind]) + ' PBF'
    elif pbf_type == 'exp':
        label = 'Exponential PBF'

    markers, caps, bars = axs.flat[0].errorbar(x = mjds, y = plaw_data[:,0], yerr = [plaw_data[:,1], plaw_data[:,2]], fmt = 'o', ms = 5, color = 'g', capsize = 2, label = label)

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    axs.flat[0].legend(loc = 'upper right', fontsize = 14, bbox_to_anchor = (1.4,1))
    axis2 = axs.flat[0].twiny()
    XLIM = axs.flat[0].get_xlim()
    XLIM = list(map(lambda x: Time(x,format='mjd',scale='utc').decimalyear,XLIM))
    axis2.set_xlim(XLIM)
    axis2.set_xlabel('Years', fontsize = 12)
    axis2.tick_params(axis='x', labelsize = 'large')

    axs.flat[1].set_ylabel(r'$\tau_0$ ($\mu$s)', fontsize = 14)
    axs.flat[1].set_xlabel('MJD', fontsize = 14)
    markers, caps, bars = axs.flat[1].errorbar(x = mjds, y = plaw_data[:,3], yerr = [plaw_data[:,4], plaw_data[:,5]], fmt = 'o', ms = 5, color = 'g', capsize = 2, label = label)

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    fig.tight_layout()

    title = plot_title('time_scales', pbf_type)

    plt.savefig(title, bbox_inches='tight')
    plt.close('all')

def autocorrelation(plaw_data, pbf_type):
    #autocorrelation
    plt.figure(1)

    #must make sure data is sorted by mjd for autocorrelation
    arr1inds = mjds.argsort()

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize = (12,6), sharex = True)
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=10)

    if pbf_type == 'zeta':
        label = r'$\zeta$ = '+ str(zetas[bzeta_ind]) + ' PBF '
    elif pbf_type == 'beta':
        label = r'$\beta$ = '+ str(betas[bzeta_ind]) + ' PBF '
    elif pbf_type == 'exp':
        label = 'Exponential PBF '

    acor1 = axs.flat[0].acorr(plaw_data[:,0][arr1inds], label = label+r'$X_{\tau}$', maxlags = 55)
    acor2 = axs.flat[1].acorr(plaw_data[:,3][arr1inds], label = label+r'$\tau_0$', maxlags = 55)

    fig.legend()

    fig.tight_layout()

    title = plot_title('autocorrelation', pbf_type)

    plt.savefig(title, bbox_inches='tight')
    plt.close('all')

def slope_and_int_hist(plaw_data, likelihood_slope, likelihood_yint, pbf_type):

    plt.figure(1)

    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=13)

    if pbf_type == 'zeta':
        label = r'$\zeta$ = '+ str(zetas[bzeta_ind]) + ' PBF '
    elif pbf_type == 'beta':
        label = r'$\beta$ = '+ str(betas[bzeta_ind]) + ' PBF '
    elif pbf_type == 'exp':
        label = 'Exponential PBF '

    #now plot the histograms and summed likeihoods (essentially the histograms w error)
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 14), sharex = 'col', sharey = 'row')

    axs.flat[0].hist(plaw_data[:,0], color = 'dimgrey', alpha = 0.8, bins = 10, label = label)
    axs.flat[0].set_ylabel('Counts', fontsize=14)

    axs.flat[1].hist(plaw_data[:,3], color = 'dimgrey', alpha = 0.8, bins = 10, label = label)

    fig.tight_layout()

    norm_likelihood_slope = np.zeros(np.shape(likelihood_slope))
    for i in range(np.size(likelihood_slope[:,0])):
        norm_likelihood_slope[i] = likelihood_slope[i]/ trapz(likelihood_slope[i])

    norm_likelihood_yint = np.zeros(np.shape(likelihood_yint))
    for i in range(np.size(likelihood_yint[:,0])):
        norm_likelihood_yint[i] = likelihood_yint[i]/ trapz(likelihood_yint[i])

    axs.flat[2].plot(-1.0*slope_test, np.sum(norm_likelihood_slope, axis = 0)/trapz(np.sum(norm_likelihood_slope, axis = 0)), color = 'dimgrey', alpha = 0.5, label = label)
    axs.flat[2].set_ylabel(r'Normalized Integrated Likelihood', fontsize=14)

    axs.flat[3].plot(yint_test, np.sum(norm_likelihood_yint, axis = 0)/trapz(np.sum(norm_likelihood_yint, axis = 0)), color = 'dimgrey', alpha = 0.5, label = label)

    axs.flat[0].set_yticks(np.linspace(0,12,5))

    axs.flat[2].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    axs.flat[0].legend(loc = 'upper left', fontsize = 13)
    axs.flat[2].legend(loc = 'upper left', fontsize = 13)

    axs.flat[2].set_xlabel(r'$X_{\tau}$', fontsize=16)
    axs.flat[3].set_xlabel(r'$\tau_0$ ($\mu$s)', fontsize=16)

    fig.tight_layout()

    title = plot_title('slope_and_int_hist', pbf_type)

    plt.savefig(title, bbox_inches='tight')

    plt.close('all')


def tau_vs_x_tau_corr(plaw_data, pbf_type):

    #now plot tau_0 versus x_tau
    plt.figure(1)
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=10)

    fig, ax = plt.subplots()

    if pbf_type == 'zeta':
        label = r'$\zeta$ = '+ str(zetas[bzeta_ind]) + ' PBF '
    elif pbf_type == 'beta':
        label = r'$\beta$ = '+ str(betas[bzeta_ind]) + ' PBF '
    elif pbf_type == 'exp':
        label = 'Exponential PBF '

    markers, caps, bars = ax.errorbar(x = plaw_data[:,3], y = plaw_data[:,0], xerr = [plaw_data[:,4], plaw_data[:,5]], yerr = [plaw_data[:,1], plaw_data[:,2]], fmt = 'o', ms = 5, color = 'g', capsize = 2, label = label)

    [bar.set_alpha(0.2) for bar in bars]
    [cap.set_alpha(0.2) for cap in caps]

    plt.xlabel(r'$\tau_0$ ($\mu$s)', fontsize = 14)
    plt.ylabel(r'$X_{\tau}$', fontsize = 14)

    corr, _ = pearsonr(plaw_data[:,3], plaw_data[:,0])

    plt.text(0.05, 0.05, f'r = {np.round(corr,2)}', bbox=dict(facecolor='none', edgecolor='black'), transform=ax.transAxes)

    plt.legend(loc = 'upper right')

    title = plot_title('tau_vs_x_tau_corr', pbf_type)

    plt.savefig(title, bbox_inches='tight')

    plt.close('all')


def plot_all_powerlaws(intrinsic_shape, pbf_type, bzeta_ind, iwidth_ind_bzeta = -1, iwidth_ind_exp = -1):

    '''Calculates best fit slopes and yintercepts and plots relevant information'''

    #saves the powerlaw data
    if intrinsic_shape != 'modeled':
        if pbf_type == 'beta':
            title = f'powerlaw_data|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}|MEDIUM={screen}|IWIDTH_DIS={intrinsic_fwhms[iwidth_ind_bzeta]}|TIME_AVG_EVERY{time_avg_factor}'
        elif pbf_type == 'zeta':
            title = f'powerlaw_data|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}|MEDIUM={screen}|IWIDTH_DIS={intrinsic_fwhms[iwidth_ind_bzeta]}|TIME_AVG_EVERY{time_avg_factor}'
        elif pbf_type == 'exp':
            title = f'powerlaw_data|{intrinsic_shape.upper()}|{pbf_type.upper()}|MEDIUM={screen}|IWIDTH_EXP={intrinsic_fwhms[iwidth_ind_exp]}|TIME_AVG_EVERY{time_avg_factor}'

    elif intrinsic_shape == 'modeled':
        if pbf_type == 'beta':
            title = f'powerlaw_data|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}|MEDIUM={screen}|TIME_AVG_EVERY{time_avg_factor}'
        elif pbf_type == 'zeta':
            title = f'powerlaw_data|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}|MEDIUM={screen}|TIME_AVG_EVERY{time_avg_factor}'
        elif pbf_type == 'exp':
            title = f'powerlaw_data|{intrinsic_shape.upper()}|{pbf_type.upper()}|MEDIUM={screen}|TIME_AVG_EVERY{time_avg_factor}'

    if (intrinsic_shape == 'modeled' and sys.argv[5] == 'rerun') or (intrinsic_shape != 'modeled' and sys.argv[7] == 'rerun'):

        if pbf_type == 'beta' or pbf_type == 'zeta':
            iwidth_ind = iwidth_ind_bzeta

        elif pbf_type == 'exp':
            iwidth_ind = iwidth_ind_exp

        plaw_data_f = np.zeros((np.size(mjds),6))
        likelihood_slope_f = np.zeros((np.size(mjds),num_slope))
        likelihood_yint_f = np.zeros((np.size(mjds),num_yint))
        slopes_arr_f = np.zeros((np.size(mjds),4))
        tau_values_collect_f = []
        tau_values_lowerr_collect_f = []
        tau_values_higherr_collect_f = []
        frequencies_collect_f = []

        plaw_data, likelihood_slope, likelihood_yint, slopes, taus, taus_low, taus_high, frequencies \
        = tau_vs_freq_pwrlaw(0, np.size(mjds)//2, intrinsic_shape, pbf_type, bzeta_ind = bzeta_ind, iwidth_ind = iwidth_ind)
        plaw_data_f[:np.size(mjds)//2] = plaw_data
        likelihood_slope_f[:np.size(mjds)//2] = likelihood_slope
        likelihood_yint_f[:np.size(mjds)//2] = likelihood_yint
        slopes_arr_f[:np.size(mjds)//2] = slopes
        for x in taus:
            tau_values_collect_f.append(x)
        for x in frequencies:
            frequencies_collect_f.append(x)
        for x in taus_low:
            tau_values_lowerr_collect_f.append(x)
        for x in taus_high:
            tau_values_higherr_collect_f.append(x)

        plaw_data, likelihood_slope, likelihood_yint, slopes, taus, taus_low, taus_high, frequencies \
        = tau_vs_freq_pwrlaw(np.size(mjds)//2, np.size(mjds), intrinsic_shape, pbf_type, bzeta_ind = bzeta_ind, iwidth_ind = iwidth_ind)
        plaw_data_f[np.size(mjds)//2:np.size(mjds)] = plaw_data
        likelihood_slope_f[np.size(mjds)//2:np.size(mjds)] = likelihood_slope
        likelihood_yint_f[np.size(mjds)//2:np.size(mjds)] = likelihood_yint
        slopes_arr_f[np.size(mjds)//2:np.size(mjds)] = slopes
        for x in taus:
            tau_values_collect_f.append(x)
        for x in frequencies:
            frequencies_collect_f.append(x)
        for x in taus_low:
            tau_values_lowerr_collect_f.append(x)
        for x in taus_high:
            tau_values_higherr_collect_f.append(x)

        np.savez(title, plaw_data = plaw_data_f, likelihood_slope = likelihood_slope_f, \
        likelihood_yint = likelihood_yint_f, \
        slopes_arr = slopes_arr_f, tau_values_collect = np.array(tau_values_collect_f, dtype = object), \
        frequencies_collect = np.array(frequencies_collect_f, dtype = object), \
        tau_values_lowerr_collect = np.array(tau_values_lowerr_collect_f, dtype = object), \
        tau_values_higherr_collect = \
        np.array(tau_values_higherr_collect_f, dtype = object))

    else:

        plaw_data_f = np.load(title+'.npz')['plaw_data']
        likelihood_slope_f = np.load(title+'.npz')['likelihood_slope']
        likelihood_yint_f = np.load(title+'.npz')['likelihood_yint']
        slopes_arr_f = np.load(title+'.npz')['slopes_arr']

    time_scales(plaw_data_f, pbf_type)
    autocorrelation(plaw_data_f, pbf_type)
    slope_and_int_hist(plaw_data_f, likelihood_slope_f, likelihood_yint_f, pbf_type)
    tau_vs_x_tau_corr(plaw_data_f, pbf_type)
    slopes_at_varying_freqs_plot(slopes_arr_f, pbf_type)

if __name__ == '__main__':

    #load and memory map the profile fitting grid
    if intrinsic_shape != 'modeled' and screen == 'thick':

        fitting_profile_data = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/convolved_profiles_intrinsic={intrinsic_shape}|PHASEBINS={phase_bins}.npz')
        convolved_profiles = {}
        tau_values = {}

        convolved_profiles_beta = np.memmap('/Users/abrageiger/Documents/research/projects/pbf_fitting/beta_convolved_profiles|SCRIPT=fin_fit', dtype='float64', mode='w+', shape=np.shape(fitting_profile_data['beta_profiles']))
        convolved_profiles_beta[:] = fitting_profile_data['beta_profiles'][:]
        convolved_profiles_beta.flush()
        convolved_profiles['beta'] = convolved_profiles_beta
        tau_values['beta'] = fitting_profile_data['beta_taus_mus']
        betas = fitting_profile_data['betas']

        convolved_profiles_zeta = np.memmap('/Users/abrageiger/Documents/research/projects/pbf_fitting/zeta_convolved_profiles|SCRIPT=fin_fit', dtype='float64', mode='w+', shape=np.shape(fitting_profile_data['zeta_profiles']))
        convolved_profiles_zeta[:] = fitting_profile_data['zeta_profiles'][:]
        convolved_profiles_zeta.flush()
        convolved_profiles['zeta'] = convolved_profiles_zeta
        tau_values['zeta'] = fitting_profile_data['zeta_taus_mus']
        zetas = fitting_profile_data['zetas']

        convolved_profiles_exp = np.memmap('/Users/abrageiger/Documents/research/projects/pbf_fitting/exp_convolved_profiles|SCRIPT=fin_fit', dtype='float64', mode='w+', shape=np.shape(fitting_profile_data['exp_profiles']))
        convolved_profiles_exp[:] = fitting_profile_data['exp_profiles'][:]
        convolved_profiles_exp.flush()
        convolved_profiles['exp'] = convolved_profiles_exp
        tau_values['exp'] = fitting_profile_data['exp_taus_mus']

        fitting_profiles = convolved_profiles

        intrinsic_fwhms = fitting_profile_data['intrinsic_fwhm_mus']

        del(fitting_profile_data)

    elif intrinsic_shape != 'modeled' and screen == 'thin':

        raise Exception("Thin screen model is only available for modeled intrinsic shape.")

    elif screen == 'thick':

        ua_pbfs = {}
        tau_values = {}

        ua_pbfs['beta'] = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']
        ua_pbfs['zeta'] = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']
        ua_pbfs['exp'] = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']

        tau_values['beta'] = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']
        tau_values['zeta'] = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']
        tau_values['exp'] = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']

        betas = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['betas']
        zetas = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['zetas']

        fitting_profiles = ua_pbfs
        intrinsic_fwhms = -1

    else:

        # thin screen intrinsic modeled case
        unith_pbfs = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS={phase_bins}.npz')['pbfs_unitheight']
        for i in unith_pbfs:
            for ii in i:
                for iii in ii:
                    iii = iii/trapz(iii)
        tau_values_start = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS={phase_bins}.npz')['tau_mus']

        betas = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS={phase_bins}.npz')['betas']
        zetas = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS={phase_bins}.npz')['zetas']

        beta_range_ind = np.where((zetas == 0.01))[0][0]
        zeta_range_ind = np.where((betas == 3.667))[0][0]

        ua_pbfs = {}
        tau_values = {}
        ua_pbfs['beta'] = unith_pbfs[:, beta_range_ind, :, :]
        ua_pbfs['zeta'] = unith_pbfs[zeta_range_ind, :, :, :]
        ua_pbfs['exp'] = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']

        tau_values['beta'] = tau_values_start[:, beta_range_ind, :]
        tau_values['zeta'] = tau_values_start[zeta_range_ind, :, :]
        tau_values['exp'] = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']

        print(tau_values['zeta'])
        fitting_profiles = ua_pbfs
        intrinsic_fwhms = -1


    if (zeta != 0 and screen == 'thick') or (zeta != 0.01 and screen == 'thin'):
        type_test = 'zeta'
        bzeta_ind = find_nearest(zetas, zeta)[1][0][0]

    elif beta != 0 and (zeta == 0 or zeta == 0.01):
        type_test = 'beta'
        bzeta_ind = find_nearest(betas,beta)[1][0][0]

    elif screen == 'exp':
        type_test = 'exp'
        bzeta_ind = -1

    if intrinsic_shape != 'modeled':

        iwidth_ind_bzeta = find_nearest(intrinsic_fwhms, iwidth_bzeta)[1][0][0]
        iwidth_ind_exp = find_nearest(intrinsic_fwhms, iwidth_exp)[1][0][0]

    else:

        iwidth_ind_bzeta = -1
        iwidth_ind_exp = -1

    time_avg_factor = init_data_phase_bins/phase_bins

    #set the tested slope and yint vals:
    num_slope = 800
    num_yint = 800

    ref_freq = 1.5

    slope_test = np.linspace(-14.0, -0.5, num = num_slope)
    yint_test = np.linspace(30.0, 200.0, num = num_yint)

    #load in the data
    with open('/Users/abrageiger/Documents/research/projects/pbf_fitting/j1903_data.pkl', 'rb') as fp:
        data_dict = pickle.load(fp)

    mjd_strings = list(data_dict.keys())
    mjds = np.zeros(np.size(mjd_strings))
    for i in range(np.size(mjd_strings)):
        mjds[i] = data_dict[mjd_strings[i]]['mjd']

    #run the data collection and plotting
    plot_all_powerlaws(intrinsic_shape, type_test, int(bzeta_ind), iwidth_ind_bzeta, iwidth_ind_exp)

    if intrinsic_shape != 'modeled' and screen == 'thick':
        os.remove('/Users/abrageiger/Documents/research/projects/pbf_fitting/beta_convolved_profiles|SCRIPT=fin_fit')
        os.remove('/Users/abrageiger/Documents/research/projects/pbf_fitting/zeta_convolved_profiles|SCRIPT=fin_fit')
        os.remove('/Users/abrageiger/Documents/research/projects/pbf_fitting/exp_convolved_profiles|SCRIPT=fin_fit')
