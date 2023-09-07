import sys
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

if intrinsic_shape != 'modeled':
    # 4 - intrinsic width (float or int) for numerical pbf fitting
    iwidth_bzeta = float(sys.argv[4])
    # 5 - intrinsic width (float or int) for decaying exponential pbf fitting
    iwidth_exp = float(sys.argv[5])

# make sure inputted intrinsic shape is valid
if intrinsic_shape != 'gaussian' and intrinsic_shape != 'sband_avg' and intrinsic_shape != 'modeled':
    raise Exception('Please choose a valid intrinsic shape: either gaussian, sband_avg, or modeled.')

if zeta != 0 and beta != 3.667:
    raise Exception('Invalid PBF Selected: Zeta greater that zero is only available for Beta = 3.667.')

freqs_arr = np.linspace(1170.0,1770.0,8)

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

    #to collect power law slopes, intercepts, and errors
    plaw_data = np.zeros((mjd_end_ind-mjd_start_ind,3))

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

        prof = Profile_Fitting(mjd, data, freqs, dur, intrinsic_shape, betas, zetas, fitting_profiles, tau_values, intrinsic_fwhms)

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
        #calculate the chi-squared surface for the varying slopes and yints
        chisqs = np.zeros(len(yint_test))
        for ii, n in enumerate(yint_test):
            error_above_vs_below = np.zeros(np.size(logtaus))
            for iii in np.arange(np.size(logtaus)):
                if logtaus[iii] > (set_slope * (np.subtract(logfreqs[iii],math.log10(ref_freq))) + math.log10(n)):
                    error_above_vs_below[iii] = total_low_erra[iii]
                elif logtaus[iii] < (set_slope * (np.subtract(logfreqs[iii],math.log10(ref_freq))) + math.log10(n)):
                    error_above_vs_below[iii] = total_high_erra[iii]
            yphi = (logtaus - (set_slope * (np.subtract(logfreqs,math.log10(ref_freq))) + math.log10(n))) /  error_above_vs_below # subtract so ref freq GHz y-int
            yphisq = yphi ** 2
            yphisq2sum = sum(yphisq)
            chisqs[ii] = yphisq2sum

        chisqs = chisqs - np.amin(chisqs)
        chisqs = np.exp((-0.5)*chisqs)

        likelihoody = likelihood_evaluator(yint_test, chisqs)

        yint = likelihoody[0]

        plaw_data[inder][0] = yint
        plaw_data[inder][1] = likelihoody[1]
        plaw_data[inder][2] = likelihoody[2]

        #FIXME
        axsb.flat[inder].loglog()
        y = ((np.subtract(logfreqs,math.log10(ref_freq)))*set_slope) + math.log10(likelihoody[0])
        axsb.flat[inder].errorbar(x = freq_list/1000.0, y = tau_listb, yerr = [total_low_erra, total_high_erra], fmt = '.', color = '0.50', elinewidth = 0.78, ms = 4.5)
        textstr = '\n'.join((
        r'$\mathrm{MJD}=%.0f$' % (int(mjd), ),
        r'$\tau_0=%.1f$' % (yint, )))
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
        ax.set_title(r'J1903+0327 Scattering vs. Epoch; $\beta$ = '+ f'{betas[bzeta_ind]}' + r'; $\zeta$ = 0.0; $X_{\tau}$ = ' + f'{set_slope}')
    elif pbf_type == 'zeta':
        ax.set_title(r'J1903+0327 Scattering vs. Epoch; $\zeta$ = '+ f'{zetas[bzeta_ind]}' + r'; $\beta$ = 11/3; $X_{\tau}$ = ' + f'{set_slope}')
    elif pbf_type == 'exp':
        ax.set_title(r'J1903+0327 Scattering vs. Epoch; Exponential PBF')
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    if intrinsic_shape != 'modeled':
        if pbf_type == 'beta':
            title = f'power_laws|SET_SLOPE={set_slope}|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}|IWIDTH={intrinsic_fwhms[iwidth_ind]}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'zeta':
            title = f'power_laws|SET_SLOPE={set_slope}|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}|IWIDTH={intrinsic_fwhms[iwidth_ind]}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'exp':
            title = f'power_laws|SET_SLOPE={set_slope}|{intrinsic_shape.upper()}|{pbf_type.upper()}|IWIDTH={intrinsic_fwhms[iwidth_ind]}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'

    elif intrinsic_shape == 'modeled':
        if pbf_type == 'beta':
            title = f'power_laws|SET_SLOPE={set_slope}|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'zeta':
            title = f'power_laws|SET_SLOPE={set_slope}|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'exp':
            title = f'power_laws|SET_SLOPE={set_slope}|{intrinsic_shape.upper()}|{pbf_type.upper()}|MJD={int(mjd_start)}-{int(mjd_end)}|TIME_AVG_EVERY={time_avg_factor}.pdf'

    figb.savefig(title)
    figb.show()
    plt.close('all')

    return(plaw_data)

def plot_title(plot_title, pbf_type):

    '''Plot title is string to indentify plot, pbf_type is string of pbf type'''

    if intrinsic_shape != 'modeled':
        if pbf_type == 'beta':
            title = f'{plot_title}|SET_SLOPE_E={set_slope_e}|SET_SLOPE_B={set_slope_b}|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}&EXP|IWIDTH_DIS={intrinsic_fwhms[iwidth_ind_bzeta]}|IWIDTH_EXP={intrinsic_fwhms[iwidth_ind_exp]}|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'zeta':
            title = f'{plot_title}|SET_SLOPE_E={set_slope_e}|SET_SLOPE_B={set_slope_b}|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}&EXP|IWIDTH_DIS={intrinsic_fwhms[iwidth_ind_bzeta]}||IWIDTH_EXP={intrinsic_fwhms[iwidth_ind_exp]}|TIME_AVG_EVERY={time_avg_factor}.pdf'

    elif intrinsic_shape == 'modeled':
        if pbf_type == 'beta':
            title = f'{plot_title}|SET_SLOPE_E={set_slope_e}|SET_SLOPE_B={set_slope_b}|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}&EXP|TIME_AVG_EVERY={time_avg_factor}.pdf'
        elif pbf_type == 'zeta':
            title = f'{plot_title}|SET_SLOPE_E={set_slope_e}|SET_SLOPE_B={set_slope_b}|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}&EXP|TIME_AVG_EVERY={time_avg_factor}.pdf'

    return(title)


def time_scales(plaw_datab, plaw_datae, intrinsic_shape, pbf_type, iwidth_ind_bzeta = -1, iwidth_ind_exp = -1):

    plt.figure(1)

    fig, axs = plt.subplots(figsize = (12,5), sharex = True)
    plt.rc('font', family = 'serif')
    plt.rc('xtick', labelsize='xx-large')
    plt.rc('ytick', labelsize='xx-large')

    markers, caps, bars = axs.errorbar(x = mjds, y = plaw_datae[:,0], yerr = [plaw_datae[:,1], plaw_datae[:,2]], fmt = 'o', ms = 5, color = 'g', capsize = 2, label = 'Exponential PBF')

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    if pbf_type == 'zeta':
        label = r'$\zeta$ = '+ str(zetas[bzeta_ind]) + ' PBF'
    else:
        label = r'$\beta$ = '+ str(betas[bzeta_ind]) + ' PBF'
    markers, caps, bars = axs.errorbar(x = mjds, y = plaw_datab[:,0], yerr = [plaw_datab[:,1], plaw_datab[:,2]], fmt = 's', color = 'dimgrey', capsize = 2, label = label)
    axs.set_ylabel(r'$X_{\tau}$', fontsize = 14)

    [bar.set_alpha(0.3) for bar in bars]
    [cap.set_alpha(0.3) for cap in caps]

    axs.legend(loc = 'upper right', fontsize = 14, bbox_to_anchor = (1.4,1))
    axis2 = axs.twiny()
    XLIM = axs.get_xlim()
    XLIM = list(map(lambda x: Time(x,format='mjd',scale='utc').decimalyear,XLIM))
    axis2.set_xlim(XLIM)
    axis2.set_xlabel('Years', fontsize = 12)
    axis2.tick_params(axis='x', labelsize = 'large')

    axs.set_ylabel(r'$\tau_0$ ($\mu$s)', fontsize = 14)
    axs.set_xlabel('MJD', fontsize = 14)

    fig.tight_layout()

    title = plot_title('time_scales', pbf_type)

    plt.savefig(title, bbox_inches='tight')
    plt.close('all')


def plot_all_powerlaws(intrinsic_shape, pbf_type, bzeta_ind, iwidth_ind_bzeta = -1, iwidth_ind_exp = -1):

    '''Calculates best fit slopes and yintercepts and plots relevant information'''

    #saves the powerlaw data
    if intrinsic_shape != 'modeled':
        if pbf_type == 'beta':
            title = f'powerlaw_data|SET_SLOPE_E={set_slope_e}|SET_SLOPE_B={set_slope_b}|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}&EXP|IWIDTH_DIS={intrinsic_fwhms[iwidth_ind_bzeta]}|IWIDTH_EXP={intrinsic_fwhms[iwidth_ind_exp]}|TIME_AVG_EVERY{time_avg_factor}'
        elif pbf_type == 'zeta':
            title = f'powerlaw_data|SET_SLOPE_E={set_slope_e}|SET_SLOPE_B={set_slope_b}|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}&EXP|IWIDTH_DIS={intrinsic_fwhms[iwidth_ind_bzeta]}||IWIDTH_EXP={intrinsic_fwhms[iwidth_ind_exp]}|TIME_AVG_EVERY{time_avg_factor}'

    elif intrinsic_shape == 'modeled':
        if pbf_type == 'beta':
            title = f'powerlaw_data|SET_SLOPE_E={set_slope_e}|SET_SLOPE_B={set_slope_b}|{intrinsic_shape.upper()}|{pbf_type.upper()}={betas[bzeta_ind]}&EXP|TIME_AVG_EVERY{time_avg_factor}'
        elif pbf_type == 'zeta':
            title = f'powerlaw_data|SET_SLOPE_E={set_slope_e}|SET_SLOPE_B={set_slope_b}|{intrinsic_shape.upper()}|{pbf_type.upper()}={zetas[bzeta_ind]}&EXP|TIME_AVG_EVERY{time_avg_factor}'

    plaw_datab = np.zeros((np.size(mjds),3))
    plaw_datae = np.zeros((np.size(mjds),3))

    plaw_data = tau_vs_freq_pwrlaw(0, np.size(mjds)//2, intrinsic_shape, pbf_type, bzeta_ind = bzeta_ind, iwidth_ind = iwidth_ind_bzeta, set_slope = set_slope_b)
    plaw_datab[:np.size(mjds)//2] = plaw_data

    plaw_data = tau_vs_freq_pwrlaw(0, np.size(mjds)//2, intrinsic_shape, 'exp', iwidth_ind = iwidth_ind_exp, set_slope = set_slope_e)
    plaw_datae[:np.size(mjds)//2] = plaw_data

    plaw_data = tau_vs_freq_pwrlaw(np.size(mjds)//2, np.size(mjds), intrinsic_shape, pbf_type, bzeta_ind = bzeta_ind, iwidth_ind = iwidth_ind_bzeta, set_slope = set_slope_b)
    plaw_datab[np.size(mjds)//2:np.size(mjds)] = plaw_data

    plaw_data = tau_vs_freq_pwrlaw(np.size(mjds)//2, np.size(mjds), intrinsic_shape, 'exp', iwidth_ind = iwidth_ind_exp, set_slope = set_slope_e)
    plaw_datae[np.size(mjds)//2:np.size(mjds)] = plaw_data

    np.savez(title, plaw_datab = plaw_datab, plaw_datae = plaw_datae)

    time_scales(plaw_datab, plaw_datae, intrinsic_shape, pbf_type, iwidth_ind_bzeta, iwidth_ind_exp)

#===============================================================================
# Main portion of code
# ==============================================================================


if __name__ == '__main__':


    #load and memory map the profile fitting grid
    if intrinsic_shape != 'modeled':

        fitting_profile_data = np.load(f'convolved_profiles_intrinsic={intrinsic_shape}|PHASEBINS={phase_bins}.npz')
        convolved_profiles = {}
        tau_values = {}

        convolved_profiles_beta = np.memmap('beta_convolved_profiles|SCRIPT=fin_fit_set_x_tau', dtype='float64', mode='w+', shape=np.shape(fitting_profile_data['beta_profiles']))
        convolved_profiles_beta[:] = fitting_profile_data['beta_profiles'][:]
        convolved_profiles_beta.flush()
        convolved_profiles['beta'] = convolved_profiles_beta
        tau_values['beta'] = fitting_profile_data['beta_taus_mus']
        betas = fitting_profile_data['betas']

        convolved_profiles_zeta = np.memmap('zeta_convolved_profiles|SCRIPT=fin_fit_set_x_tau', dtype='float64', mode='w+', shape=np.shape(fitting_profile_data['zeta_profiles']))
        convolved_profiles_zeta[:] = fitting_profile_data['zeta_profiles'][:]
        convolved_profiles_zeta.flush()
        convolved_profiles['zeta'] = convolved_profiles_zeta
        tau_values['zeta'] = fitting_profile_data['zeta_taus_mus']
        zetas = fitting_profile_data['zetas']

        convolved_profiles_exp = np.memmap('exp_convolved_profiles|SCRIPT=fin_fit_set_x_tau', dtype='float64', mode='w+', shape=np.shape(fitting_profile_data['exp_profiles']))
        convolved_profiles_exp[:] = fitting_profile_data['exp_profiles'][:]
        convolved_profiles_exp.flush()
        convolved_profiles['exp'] = convolved_profiles_exp
        tau_values['exp'] = fitting_profile_data['exp_taus_mus']

        fitting_profiles = convolved_profiles

        intrinsic_fwhms = fitting_profile_data['intrinsic_fwhm_mus']

        del(fitting_profile_data)

    else:

        ua_pbfs = {}
        tau_values = {}

        ua_pbfs['beta'] = np.load(f'beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']
        ua_pbfs['zeta'] = np.load(f'zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']
        ua_pbfs['exp'] = np.load(f'exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']

        tau_values['beta'] = np.load(f'beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']
        tau_values['zeta'] = np.load(f'zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']
        tau_values['exp'] = np.load(f'exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']

        betas = np.load(f'beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['betas']
        zetas = np.load(f'zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['zetas']

        fitting_profiles = ua_pbfs
        intrinsic_fwhms = -1

    time_avg_factor = init_data_phase_bins/phase_bins

    #collect beta, zeta, and relevant related information
    if zeta != 0:
        type_test = 'zeta'
        bzeta_ind = find_nearest(zetas, zeta)[1][0][0]
        title = f'powerlaw_data|{intrinsic_shape.upper()}|{type_test.upper()}={zetas[bzeta_ind]}&EXP|TIME_AVG_EVERY{time_avg_factor}.npz'

    else:
        type_test = 'beta'
        bzeta_ind = find_nearest(betas,beta)[1][0][0]
        title = f'powerlaw_data|{intrinsic_shape.upper()}|{type_test.upper()}={betas[bzeta_ind]}&EXP|TIME_AVG_EVERY{time_avg_factor}.npz'

    if intrinsic_shape != 'modeled':

        iwidth_ind_bzeta = find_nearest(intrinsic_fwhms, iwidth_bzeta)[1][0][0]
        iwidth_ind_exp = find_nearest(intrinsic_fwhms, iwidth_exp)[1][0][0]

    else:

        iwidth_ind_bzeta = -1
        iwidth_ind_exp = -1

    data_previous = np.load(title)
    set_slope_e = np.round(-np.average(data_previous['plaw_datae'][:,0]),3)
    print(r'$X_{\tau}$ for Exponential = ' + f'{set_slope_e}')
    set_slope_b = np.round(-np.average(data_previous['plaw_datab'][:,0]),3)
    print(r'$X_{\tau}$ for Extended Media PBF = ' + f'{set_slope_b}')
    #set_slope_e = -4.4
    #set_slope_b = -4.4

    #set the tested slope and yint vals:
    num_yint = 700

    ref_freq = 1.5

    yint_test = np.linspace(30.0, 300.0, num = num_yint)

    #load in the data
    with open('j1903_data.pkl', 'rb') as fp:
        data_dict = pickle.load(fp)

    mjd_strings = list(data_dict.keys())
    mjds = np.zeros(np.size(mjd_strings))
    for i in range(np.size(mjd_strings)):
        mjds[i] = data_dict[mjd_strings[i]]['mjd']

    #run the data collection and plotting
    plot_all_powerlaws(intrinsic_shape, type_test, int(bzeta_ind), iwidth_ind_bzeta, iwidth_ind_exp)

    if intrinsic_shape != 'modeled':
        os.remove('beta_convolved_profiles|SCRIPT=fin_fit_set_x_tau')
        os.remove('zeta_convolved_profiles|SCRIPT=fin_fit_set_x_tau')
        os.remove('exp_convolved_profiles|SCRIPT=fin_fit_set_x_tau')
