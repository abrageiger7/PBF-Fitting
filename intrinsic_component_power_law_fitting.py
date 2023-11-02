import pickle
import numpy as np
import matplotlib.pyplot as plt
from pypulse.singlepulse import SinglePulse
from scipy.integrate import trapz
from scipy import optimize
from scipy.interpolate import CubicSpline
import math
from pypulse.singlepulse import SinglePulse
from fit_functions import single_gauss, triple_gauss, convolve, stretch_or_squeeze, chi2_distance, calculate_tau
import sys

#system imports
beta = sys.argv[1]
zeta = sys.argv[2]
rerun = str(sys.argv[3])


if __name__ == '__main__':

    # shape normalized so that center gaussian component has a height of 1
    a2 = 1.0

    # load in the j1903 subband averages - these are averaged over all epochs
    with open('j1903_average_profiles_lband.pkl', 'rb') as fp:
        lband_avgs = pickle.load(fp)

    # frequencies corresponding to the j1903 subband averages
    freqs = np.array([1160,1260,1360,1460,1560,1660,1760,2200])

    # set plotting preferences
    plt.rc('font', family = 'serif')

    # the pulse broadening function corresponding to the system imports
    pbf = np.load(f'zeta_{zeta}_beta_{beta}_pbf.npy')
    cordes_phase_bins = np.size(pbf)

    # the sband average shape over all mjd and frequency
    sband = np.load('j1903_high_freq_temp_unsmoothed.npy')
    sband = sband / trapz(sband)
    # save the frequencies of each frequency subaverage
    starting_freq = 2200.0 #about the center of the sband for j1903 data
    lband_avgs['2200'] = sband
    freq_keys = []
    for i in lband_avgs.keys():
        freq_keys.append(i)
    phase_bins = np.size(sband)

    # rescale the pulse broadening function to have the same size as the data
    subs_time_avg = np.zeros(phase_bins)
    for ii in range(np.size(subs_time_avg)):
            subs_time_avg[ii] = np.average(pbf[((cordes_phase_bins//phase_bins)*ii):((cordes_phase_bins//phase_bins)*(ii+1))])
    # unitarea
    subs_time_avg = subs_time_avg / trapz(subs_time_avg)
    # calculate pbf tau at this pbf width
    tau_subs_time_avg = calculate_tau(subs_time_avg)[0]

    # calculate the rms of the noise for each frequency subaverage
    opr_size = np.size(lband_avgs[freq_keys[0]])//5
    rms_values = np.zeros(np.size(freqs))
    for ii in range(np.size(freqs)):
        rms_collect = 0
        for i in range(opr_size):
            rms_collect += lband_avgs[freq_keys[ii]][i]**2
        rms = math.sqrt(rms_collect/opr_size)
        rms_values[ii] = rms

    # the tvec arrays of profile size for fitting and plotting
    t = np.linspace(0,1,np.size(sband))
    timer = np.arange(np.size(sband))

    if rerun == 'rerun':

        #the sband component parameters calculated with mcmc
        parameters = np.load(f'mcmc_params|FREQ={int(starting_freq)}|BETA={beta}|ZETA={zeta}.npz')
        # gaussian component first in phase
        comp1 = parameters['parameters'][:3]
        # gaussian component second in phase
        comp2 = [a2, parameters['parameters'][3], parameters['parameters'][4]]
        # gaussian component third in phase
        comp3 = parameters['parameters'][5:8]
        #measured sband tau with this pulse broadening function
        tau = parameters['parameters'][8]
        gauss_params = [comp1,comp2,comp3]
        # now the parameters organized into the sband_comp_params array
        sband_comp_params = np.zeros(9)
        ind = 0
        for i in gauss_params:
            for ii in i:
                sband_comp_params[ind] = ii
                ind+=1

        # print the sband parameters. The powerlaws start here at sband
        print('Sband Component Parameters:')
        print(f'    Component 1 \n     Amplitude = {np.round(sband_comp_params[0],4)}    Phase = {np.round(sband_comp_params[1],4)}    Width = {np.round(sband_comp_params[2],4)}')
        print(f'    Component 2 \n     Amplitude = {np.round(sband_comp_params[3],4)}    Phase = {np.round(sband_comp_params[4],4)}    Width = {np.round(sband_comp_params[5],4)}')
        print(f'    Component 3 \n     Amplitude = {np.round(sband_comp_params[6],4)}    Phase = {np.round(sband_comp_params[7],4)}    Width = {np.round(sband_comp_params[8],4)}')

        # plot the mcmc sband fit
        plt.figure(1)
        plt.title('Fitted Pulse Shape, Frequency = Sband')
        plt.xlabel('Pulse Phase')
        plt.plot(np.linspace(0,1,np.size(sband)), sband, color = 'grey')
        convolved_prof = convolve(triple_gauss(comp1, comp2, comp3, timer)[0], stretch_or_squeeze(subs_time_avg, tau/tau_subs_time_avg))
        plt.plot(np.linspace(0,1,np.size(sband)), convolved_prof/trapz(convolved_prof), color = 'k')
        plt.show()
        plt.close('all')

        q = np.abs(sband_comp_params[:3]) #first comp

        g = np.abs(sband_comp_params[3:6]) # second comp

        r = np.abs(sband_comp_params[6:]) # third comp

        # the test amplitude powerlaw grid
        test_gamp3_pwrlaw = np.linspace(0.1,-0.7,9)
        print('3rd Component Amplitude Powerlaw Grid:')
        print(test_gamp3_pwrlaw)

        # the test phase powerlaw grid
        test_gcent3_pwrlaw = np.linspace(0.4,-0.4,9)
        print('3rd Component Phase Powerlaw Grid:')
        print(test_gcent3_pwrlaw)

        # the test width powerlaw grid
        test_gwidth3_pwrlaw = np.linspace(-0.6,-1.4,9)
        print('3rd Component Width Powerlaw Grid:')
        print(test_gwidth3_pwrlaw)

        # collect the reduced chi-squared values for each combination of the test powerlaws
        chisqs_comp3 = np.zeros((np.size(test_gamp3_pwrlaw), np.size(test_gcent3_pwrlaw), np.size(test_gwidth3_pwrlaw)))

        # the test values of tau for the fitting grid. the best value of tau is
        # kept and used to calculate the chi squared value for this combination
        # of powerlaws
        test_tau = np.linspace(0.0001,4.0,36)

        for i in range(np.size(test_gamp3_pwrlaw)):

            for iii in range(np.size(test_gcent3_pwrlaw)):

                for iv in range(np.size(test_gwidth3_pwrlaw)):

                    # sum the chi-squared over all frequencies
                    chi_sq_sum = 0

                    for ind in range(np.size(freqs)):

                        # first component amplitude at this frequency and powerlaw combination
                        gamp3 = r[0] * np.power((float(freqs[ind])/float(starting_freq)),test_gamp3_pwrlaw[i])

                        # first component phase at this frequency and powerlaw combination
                        gcent3 = (r[1] - g[1]) * np.power((float(freqs[ind])/float(starting_freq)),test_gcent3_pwrlaw[iii]) + g[1]

                        # first component width at this frequency and powerlaw combination
                        gwidth3 = r[2] * np.power((float(freqs[ind])/float(starting_freq)),test_gwidth3_pwrlaw[iv])

                        # collect the chi squared value for each grid value of tau
                        pbf_chisq = np.zeros(np.size(test_tau))

                        for ix in range(np.size(test_tau)):

                            # the
                            intrinsic_shape = triple_gauss([gamp3, gcent3, gwidth3], g, q, timer)[0]
                            pulse_broadening = stretch_or_squeeze(subs_time_avg, test_tau[ix])
                            profile = convolve(intrinsic_shape, pulse_broadening)

                            sp = SinglePulse(lband_avgs[freq_keys[ind]])
                            fitting = sp.fitPulse(profile)

                            sps = SinglePulse(profile*fitting[2])
                            fitted_template = sps.shiftit(fitting[1])

                            chi_sqs = chi2_distance(fitted_template, lband_avgs[freq_keys[ind]], rms_values[ind], 3)

                            pbf_chisq[ix] = chi_sqs

                        here = np.where((pbf_chisq == np.min(pbf_chisq)))[0][0]

                        chi_sq_sum += pbf_chisq[here]

                        print('...')

                    chisqs_comp3[i,iii,iv] = chi_sq_sum / np.size(freqs)

        print(chisqs_comp3)
        #RESULTS
        here = np.where((chisqs_comp3 == np.min(chisqs_comp3)))
        # starting_height = test_starting_height[here[1][0]]
        # print('Starting 3rd Component Height (2200 MHz) = ' + str(starting_height))
        gamp3_pwrlaw = test_gamp3_pwrlaw[here[0][0]]
        print('3rd Component Amplitude Powerlaw = ' + str(gamp3_pwrlaw))
        # starting_center = test_starting_center[here[3][0]]
        # print('Starting 3rd Component Center (2200 MHz) = ' + str(starting_center))
        gcent3_pwrlaw = test_gcent3_pwrlaw[here[1][0]]
        print('3rd Component Center Powerlaw = ' + str(gcent3_pwrlaw))
        gwidth3_pwrlaw = test_gwidth3_pwrlaw[here[2][0]]
        print('3rd Component Width Powerlaw = ' + str(gwidth3_pwrlaw))

        #PLOTTING

        phase_bins = 2048
        t = np.linspace(0,phase_bins,np.size(sband))
        time = np.linspace(0,1,phase_bins)

        freq_ticks = []
        for i in freqs:
            freq_ticks.append(int(np.round(i/1000,1)))

        test_gamp1_pwrlaw = np.linspace(0.5,1.1,14)
        print(test_gamp1_pwrlaw)

        chisqs_gamp1 = np.zeros((np.size(test_gamp1_pwrlaw)))

        for iv in range(np.size(test_gamp1_pwrlaw)):

            chi_sq_sum = 0

            for ind in range(np.size(freqs)):

                gamp3 = r[0] * np.power((float(freqs[ind])/starting_freq),gamp3_pwrlaw)

                gcent3 = (r[1] - g[1]) * np.power((float(freqs[ind])/starting_freq),gcent3_pwrlaw) + g[1]

                gwidth3 = r[2] * np.power((float(freqs[ind])/starting_freq),gwidth3_pwrlaw)

                gamp1 = q[0] * np.power((float(freqs[ind])/starting_freq),test_gamp1_pwrlaw[iv])

                pbf_chisq = np.zeros(np.size(test_tau))

                for ix in range(np.size(test_tau)):

                    profile = convolve(triple_gauss([gamp3, gcent3, gwidth3], g, [gamp1,q[1],q[2]], timer)[0], stretch_or_squeeze(subs_time_avg, test_tau[ix]))

                    sp = SinglePulse(lband_avgs[freq_keys[ind]])
                    fitting = sp.fitPulse(profile)

                    sps = SinglePulse(profile*fitting[2])
                    fitted_template = sps.shiftit(fitting[1])

                    chi_sqs = chi2_distance(fitted_template, lband_avgs[freq_keys[ind]], rms_values[ind], 1)

                    pbf_chisq[ix] = chi_sqs

                here = np.where((pbf_chisq == np.min(pbf_chisq)))[0][0]

                chi_sq_sum += pbf_chisq[here]

                print('...')

            chisqs_gamp1[iv] = chi_sq_sum / np.size(freqs)

        plt.figure(1)
        plt.plot(test_gamp1_pwrlaw, chisqs_gamp1)
        plt.xlabel('Test Powerlaws')
        plt.ylabel(r'$\chi^{2}$')
        plt.show()
        plt.close('all')

        here = np.where((chisqs_gamp1 == np.min(chisqs_gamp1)))[0][0]

        gamp1_pwrlaw = test_gamp1_pwrlaw[here]
        print('1st Component Amplitude Powerlaw = ' + str(gamp1_pwrlaw))

        # plt.figure(1)
        # plt.plot(freqs, collect_taus, color = 'k')
        # plt.show()
        # plt.close('all')

        fitted_gaussian_components = np.zeros((np.size(freqs),9))

        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (6,3), sharex = True, sharey = True)

        intrinsic_shapes = np.zeros((np.size(freqs), phase_bins))

        for ind in range(np.size(freqs)):

            gamp3 = r[0] * np.power((freqs[ind]/starting_freq),gamp3_pwrlaw)

            gcent3 = (r[1] - g[1]) * np.power((freqs[ind]/starting_freq),gcent3_pwrlaw) + g[1]

            gwidth3 = r[2] * np.power((freqs[ind]/starting_freq),gwidth3_pwrlaw)

            gamp1 = q[0] * np.power((freqs[ind]/starting_freq),gamp1_pwrlaw)

            fitted_gaussian_components[ind][0] = gamp1
            fitted_gaussian_components[ind][1] = q[1]
            fitted_gaussian_components[ind][2] = q[2]

            fitted_gaussian_components[ind][3] = g[0]
            fitted_gaussian_components[ind][4] = g[1]
            fitted_gaussian_components[ind][5] = g[2]

            fitted_gaussian_components[ind][6] = gamp3
            fitted_gaussian_components[ind][7] = gcent3
            fitted_gaussian_components[ind][8] = gwidth3

            profile = triple_gauss([gamp3, gcent3, gwidth3], g, [gamp1,q[1],q[2]], timer)[0]

            intrinsic_shapes[ind] = profile

            axs.flat[0].plot(np.linspace(0,1,np.size(subs_time_avg)), profile + (0.005*(ind)) + 0.002, color = 'grey')

        np.savez(f'j1903_modeled_params|BETA={beta}|ZETA={zeta}', fitted_gaussian_components = fitted_gaussian_components, gamp1_pwrlaw = gamp1_pwrlaw, gamp3_pwrlaw = gamp3_pwrlaw, gwidth3_pwrlaw = gwidth3_pwrlaw, gcent3_pwrlaw = gcent3_pwrlaw)

        #fig.text(0.517, 0.925, 'Modeled J1903+0327 Intrinsic', ha='center', va='center', fontsize = 14)
        axs.flat[0].set_yticks(np.linspace(0.002,(np.size(freqs)-1)*0.005+0.002,np.size(freqs)), freq_ticks)
        fig.text(0.517, 0.003, r'Pulse Phase', ha='center', va='center', fontsize = 10)
        axs.flat[0].set_ylabel('Frequency [GHz]', fontsize = 10)

        for ind in range(np.size(freqs)):

            axs.flat[1].plot(np.linspace(0,1,2048), (intrinsic_shapes[ind] -  intrinsic_shapes[3]) * 10 + (0.005*(np.size(freqs)-ind)) + 0.003, color = 'grey')

        plt.savefig(f'j1903_modeled|BETA={beta}|ZETA={zeta}.pdf', bbox_inches = 'tight')

        plt.close('all')

    fig,axs = plt.subplots(ncols = 3, nrows = 3, figsize = (14,14), sharex = True, sharey = True)

    fitted_gaussian_components = np.load(f'j1903_modeled_params|BETA={beta}|ZETA={zeta}.npz')['fitted_gaussian_components']

    print(np.shape(fitted_gaussian_components))

    test_pbfwidth = np.linspace(0.0001,4.0,200)

    total_chi_sq = 0

    chi_sq_sum = 0

    tau_values_collect = np.zeros(np.size(freqs))

    for ind in range(np.size(freqs)):

        pbf_chisq = np.zeros(np.size(test_pbfwidth))

        for iii in range(np.size(test_pbfwidth)):

            profile = convolve(triple_gauss(fitted_gaussian_components[ind][:3], fitted_gaussian_components[ind][3:6], fitted_gaussian_components[ind][6:], timer)[0], stretch_or_squeeze(subs_time_avg, test_pbfwidth[iii]))

            sp = SinglePulse(lband_avgs[freq_keys[ind]])
            fitting = sp.fitPulse(profile)

            sps = SinglePulse(profile*fitting[2])
            fitted_template = sps.shiftit(fitting[1])

            chi_sqs = chi2_distance(fitted_template, lband_avgs[freq_keys[ind]], rms_values[ind], 4)

            pbf_chisq[iii] = chi_sqs

        here = np.where((pbf_chisq == np.min(pbf_chisq)))[0][0]

        chi_sq_sum += pbf_chisq[here]

        pbfwidth = test_pbfwidth[here]
        tau = calculate_tau(stretch_or_squeeze(subs_time_avg, pbfwidth))[0]
        tau_values_collect[ind] = tau

        profile = convolve(triple_gauss(fitted_gaussian_components[ind][:3], fitted_gaussian_components[ind][3:6], fitted_gaussian_components[ind][6:], timer)[0], stretch_or_squeeze(subs_time_avg, pbfwidth))

        sp = SinglePulse(lband_avgs[freq_keys[ind]])
        fitting = sp.fitPulse(profile)

        sps = SinglePulse(profile*fitting[2])
        fitted_template = sps.shiftit(fitting[1])

        index = ind

        textstr = r'$\tau$'+f' ={int(np.round(tau))}' +r' $\mu$s'
        axs.flat[index].text(0.65, 0.95, textstr, fontsize=10, verticalalignment='top', transform=axs.flat[index].transAxes)

        axs.flat[index].plot(t, lband_avgs[freq_keys[ind]], color = 'darkgrey', lw = 2.0)

        axs.flat[index].plot(t, fitted_template, color = 'k')

        axs.flat[index].set_yticks([])

        axs.flat[index].set_title(f'{freqs[ind]} [MHz]')

    chi_sq_sum = chi_sq_sum / np.size(freqs)
    print(f'CHI-SQUARED = {chi_sq_sum}')

    np.savez(f'j1903_intrinsic_pwrlaw_chisqs|BETA={beta}|ZETA={zeta}', chisqs_comp3 = chisqs_comp3, test_gamp3_pwrlaw = test_gamp3_pwrlaw, test_gcent3_pwrlaw = test_gcent3_pwrlaw, test_gwidth3_pwrlaw = test_gwidth3_pwrlaw, chisqs_gamp1 = chisqs_gamp1, test_gamp1_pwrlaw = test_gamp1_pwrlaw, chi_sq_sum = chi_sq_sum)

    fig.delaxes(axs.flat[8])
    fig.text(0.517, 0.083, r'Pulse Phase', ha='center', va='center', fontsize = 12)

    plt.savefig(f'j1903_modeled_fitted|BETA={beta}|ZETA={zeta}.pdf', bbox_inches = 'tight')
    plt.close('all')

    plt.figure(1)
    plt.plot(freqs, tau_values_collect, color = 'k')
    plt.plot(freqs, tau_values_collect, '.', color = 'k', label = '$\tau$ Collected')
    plt.plot(freqs, tau_values_collect[np.size(freqs)//2]*np.power(freqs/freqs[np.size(freqs)//2],-4.4), ls = '--', label = '-4.4', color = 'grey')
    plt.xlabel(r'$\nu$ [MHz]')
    plt.ylabel(r'$\tau$')
    plt.xscale('log')
    plt.yscale('log')
    title = r'$\tau$ vs $\nu$; $\beta$=' + f'{beta}' + '; $\zeta$='+ f'{zeta}'
    plt.title(title)
    plt.savefig(f'j1903_modeled_tau|BETA={beta}|ZETA={zeta}.pdf', bbox_inches = 'tight')
    plt.close('all')
