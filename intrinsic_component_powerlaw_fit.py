from fit_functions import *
from pypulse.singlepulse import SinglePulse
import pickle
from correct_dm_from_scattering import Correct_DM_From_Scattering_J1903
from pathlib import Path


class Intrinsic_Component_Powerlaw_Fit_Per_Epoch:

    def __init__(self, beta, zeta, mjd, sband_param_path, guess_correct_dm_pwrlaws, sband_freq, thin_or_thick_medium, sband_data_profile):

        self.beta = beta
        self.zeta = zeta
        self.screen = thin_or_thick_medium
        if thin_or_thick_medium == 'thick':
            self.pbf = np.load(f'zeta_{zeta}_beta_{beta}_pbf.npy')
            self.pbf_tau = calculate_tau(self.pbf)[0]
        elif thin_or_thick_medium == 'thin':
            self.beta = float(beta)
            self.zeta = float(zeta)
            betas = np.load(Path(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS=256.npz'))['betas']
            zetas = np.load(Path(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS=256.npz'))['zetas']
            self.pbf_options = np.load(Path(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS=256.npz'))['pbfs_unitheight'][np.where((betas==self.beta))[0][0]][np.where((zetas==self.zeta))[0][0]]
            self.tau_options = np.load(Path(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS=256.npz'))['tau_mus'][np.where((betas==self.beta))[0][0]][np.where((zetas==self.zeta))[0][0]]
        elif thin_or_thick_medium != 'exp':
            valid_thick = '\'thick\' or \'thin\' or \'exp\''
            raise Exception(f'Choose a valid medium thickness: {valid_thick}')

        sband_params = np.load(sband_param_path)['parameters']

        self.comp1 = np.abs(sband_params[:3]) #first comp

        self.comp2 = np.abs([1.0, sband_params[3], sband_params[4]]) # second comp

        self.comp3 = np.abs(sband_params[5:8]) # third comp

        self.sband_freq = sband_freq

        self.sband_profile = sband_data_profile

        self.sband_tau = sband_params[8]

        # load in data
        with open('j1903_data.pkl', 'rb') as fp:
            data_dict = pickle.load(fp)

        mjd_strings = list(data_dict.keys())
        mjds = np.zeros(np.size(mjd_strings))
        for i in range(np.size(mjd_strings)):
            mjds[i] = data_dict[mjd_strings[i]]['mjd']

        # the mjd from the data closest to the input
        epoch_mjd = np.asarray(mjd_strings)[find_nearest(mjds, mjd)[1][0][0]]
        print(f"MJD = {epoch_mjd}")

        # data for this epoch
        epoch_data = data_dict[epoch_mjd]
        self.mjd = epoch_data['mjd']

        data = epoch_data['data']
        frequencies = np.asarray(epoch_data['freqs'])

        object_to_correct_dm = Correct_DM_From_Scattering_J1903(frequencies, data, sband_freq, sband_params, guess_correct_dm_pwrlaws)

        data = object_to_correct_dm.correct_data(beta, zeta, thin_or_thick_medium)

        # subaverage the data so that there are 8 resulting frequency channels
        self.data, self.frequencies = subaverage(data, frequencies, np.shape(data)[1]//8, np.shape(data)[0]//8)

        # calculate the rms of the noise for each frequency subaverage
        self.rms_values = np.zeros(np.size(self.frequencies))
        for ii in range(np.size(self.frequencies)):
            self.rms_values[ii] = calculate_rms(self.data[ii], np.size(self.data[ii])//5)

        self.yerr = np.zeros(np.size(self.frequencies))
        for i in range(np.size(self.frequencies)):
            self.yerr[i] = calculate_rms(self.data[i], np.size(self.data[i])//5)

        self.plot_tag = f"FREQ=lband|BETA={beta}" +\
        f"|ZETA={zeta}|SCREEN={thin_or_thick_medium.upper()}|MJD={int(np.round(mjd))}"

        if self.screen == 'exp':
            self.plot_tag = f"FREQ=lband|SCREEN={thin_or_thick_medium.upper()}|MJD={int(np.round(mjd))}"


    def fit_comp3(self, amp_test_arr, phase_test_arr, width_test_arr):
        '''Fits for the best fit powerlaws for variation of the component
        parameters over frequency.'''

        # collect the reduced chi-squared values for each combination of the test powerlaws
        chisqs = np.zeros((np.size(amp_test_arr), np.size(phase_test_arr), \
        np.size(width_test_arr)))

        test_tau = np.linspace(0.1,400.0,36)

        t = np.linspace(0,1,np.shape(self.data)[1])

        timer = np.arange(np.shape(self.data)[1])

        num_iterations = 0

        for i in range(np.size(amp_test_arr)):

            for iii in range(np.size(phase_test_arr)):

                for iv in range(np.size(width_test_arr)):

                    # sum the chi-squared over all frequencies
                    chi_sq_sum = 0

                    for ind in range(np.size(self.frequencies)):

                        # first component amplitude at this frequency and powerlaw combination
                        amp = self.comp3[0] * np.power((self.frequencies[ind]/self.sband_freq),amp_test_arr[i])

                        # first component phase at this frequency and powerlaw combination
                        phase = (self.comp3[1] - self.comp2[1]) * np.power((self.frequencies[ind]/self.sband_freq),phase_test_arr[iii]) + self.comp2[1]

                        # first component width at this frequency and powerlaw combination
                        width = self.comp3[2] * np.power((self.frequencies[ind]/self.sband_freq),width_test_arr[iv])

                        # collect the chi squared value for each grid value of tau
                        pbf_chisq = np.zeros(np.size(test_tau))

                        for ix in range(np.size(test_tau)):

                            if self.screen == 'thick':
                                #stretch or squeeze pbf, then time average
                                stretch_or_squeeze_factor = test_tau[ix]/self.pbf_tau
                                pulse_broadening = time_average(stretch_or_squeeze(self.pbf, \
                                stretch_or_squeeze_factor), np.shape(self.data)[1])

                            elif self.screen == 'thin':
                                closest_tau_ind = find_nearest(self.tau_options, test_tau[ix])[1][0][0]
                                pulse_broadening =  time_average(self.pbf_options[closest_tau_ind], np.shape(self.data)[1])

                            elif self.screen == 'exp':
                                t = np.linspace(0, j1903_period, np.shape(self.data)[1], endpoint=False)
                                pulse_broadening = (1.0/test_tau[ix])*np.power(math.e,-1.0*t/test_tau[ix])

                            intrinsic_shape = triple_gauss([amp, phase, width], self.comp2, self.comp1, timer)[0]

                            profile = convolve(intrinsic_shape, pulse_broadening)

                            sp = SinglePulse(self.data[ind])
                            fitting = sp.fitPulse(profile)

                            sps = SinglePulse(profile*fitting[2])
                            fitted_template = sps.shiftit(fitting[1])

                            # fitted_template = profile/trapz(profile)*trapz(self.data[ind])

                            chi_sqs = chi2_distance(fitted_template, self.data[ind], self.rms_values[ind], 3)

                            pbf_chisq[ix] = chi_sqs

                        here = np.where((pbf_chisq == np.min(pbf_chisq)))[0][0]

                        chi_sq_sum += pbf_chisq[here]

                        print(num_iterations)
                        num_iterations += 1

                    chisqs[i,iii,iv] = chi_sq_sum / np.size(self.frequencies)

        # sum chi-sq along the various axes in order to visulaize for each
        # parameter (parameter values are at global minimum, however)

        # example to justify (observe array shapes and sums):
        # - array = np.array([[[[0,1],[2,1],[4,2]],[[3,5],[4,7],[6,9]]]])
        # - array_sum = np.sum(array,axis=1)
        # - array_sum_2 = np.sum(np.sum(array_sum,axis = 0), axis=0)
        # - etc.

        here = np.where((chisqs == np.min(chisqs)))
        self.amp3_pwrlaw = amp_test_arr[here[0][0]]
        plt.figure(1)
        plt.title(r'$\chi^2$ vs Amplitude 3 Powerlaw')
        plt.xlabel('Amplitude Powerlaw')
        plt.ylabel(r'$\chi^2$')
        plt.plot(amp_test_arr, np.sum(np.sum(chisqs, axis=2)/np.shape(chisqs)[2], axis=1)/np.shape(chisqs)[1])
        plt.savefig(f'amp3_pwrlaw_fitting|{self.plot_tag}.pdf')
        plt.show()
        plt.close('all')
        print('3rd Component Amplitude Powerlaw = ' + str(self.amp3_pwrlaw))
        self.phase3_pwrlaw = phase_test_arr[here[1][0]]
        plt.figure(1)
        plt.title(r'$\chi^2$ vs Phase 3 Powerlaw')
        plt.xlabel('Phase Powerlaw')
        plt.ylabel(r'$\chi^2$')
        plt.plot(phase_test_arr, np.sum(np.sum(chisqs, axis=2)/np.shape(chisqs)[2], axis=0)/np.shape(chisqs)[0])
        plt.savefig(f'phi3_pwrlaw_fitting|{self.plot_tag}.pdf')
        plt.show()
        plt.close('all')
        print('3rd Component Center Powerlaw = ' + str(self.phase3_pwrlaw))
        self.width3_pwrlaw = width_test_arr[here[2][0]]
        plt.figure(1)
        plt.title(r'$\chi^2$ vs Width 3 Powerlaw')
        plt.xlabel('Width Powerlaw')
        plt.ylabel(r'$\chi^2$')
        plt.plot(width_test_arr, np.sum(np.sum(chisqs, axis=0)/np.shape(chisqs)[0], axis=0)/np.shape(chisqs)[1])
        plt.savefig(f'width3_pwrlaw_fitting|{self.plot_tag}.pdf')
        plt.show()
        plt.close('all')
        print('3rd Component Width Powerlaw = ' + str(self.width3_pwrlaw))

        return(self.amp3_pwrlaw, self.phase3_pwrlaw, self.width3_pwrlaw)

    def fit_amp1(self, param_test_array):
        '''Fits for the best fit powerlaw for variation of the first component
        amplitude over frequency.'''

        chisqs_gamp1 = np.zeros(np.size(param_test_array))

        timer = np.arange(np.shape(self.data)[1])

        test_tau = np.linspace(0.1,400.0,36)

        for iv in range(np.size(param_test_array)):

            chi_sq_sum = 0

            for ind in range(np.size(self.frequencies)):

                gamp3 = self.comp3[0] * np.power((self.frequencies[ind]/self.sband_freq),self.amp3_pwrlaw)

                gcent3 = (self.comp3[1] - self.comp2[1]) * np.power((self.frequencies[ind]/self.sband_freq),self.phase3_pwrlaw) + self.comp2[1]

                gwidth3 = self.comp3[2] * np.power((self.frequencies[ind]/self.sband_freq),self.width3_pwrlaw)

                gamp1 = self.comp1[0] * np.power((self.frequencies[ind]/self.sband_freq),param_test_array[iv])

                pbf_chisq = np.zeros(np.size(test_tau))

                for ix in range(np.size(test_tau)):

                    if self.screen == 'thick':
                        #stretch or squeeze pbf, then time average
                        stretch_or_squeeze_factor = test_tau[ix]/self.pbf_tau
                        pulse_broadening = time_average(stretch_or_squeeze(self.pbf, \
                        stretch_or_squeeze_factor), np.shape(self.data)[1])

                    elif self.screen == 'thin':
                        closest_tau_ind = find_nearest(self.tau_options, test_tau[ix])[1][0][0]
                        pulse_broadening =  time_average(self.pbf_options[closest_tau_ind], np.shape(self.data)[1])

                    elif self.screen == 'exp':
                        t = np.linspace(0, j1903_period, np.shape(self.data)[1], endpoint=False)
                        pulse_broadening = (1.0/test_tau[ix])*np.power(math.e,-1.0*t/test_tau[ix])

                    intrinsic_shape = triple_gauss([gamp3, gcent3, gwidth3], self.comp2, [gamp1,self.comp1[1],self.comp1[2]], timer)[0]

                    profile = convolve(intrinsic_shape, pulse_broadening)

                    sp = SinglePulse(self.data[ind])
                    fitting = sp.fitPulse(profile)

                    sps = SinglePulse(profile*fitting[2])
                    fitted_template = sps.shiftit(fitting[1])

                    # fitted_template = profile/trapz(profile)*trapz(self.data[ind])

                    chi_sqs = chi2_distance(fitted_template, self.data[ind], self.rms_values[ind], 1)

                    pbf_chisq[ix] = chi_sqs

                here = np.where((pbf_chisq == np.min(pbf_chisq)))[0][0]

                chi_sq_sum += pbf_chisq[here]

            chisqs_gamp1[iv] = chi_sq_sum / np.size(self.frequencies)

        here = np.where((chisqs_gamp1 == np.min(chisqs_gamp1)))[0][0]

        self.amp1_pwrlaw = param_test_array[here]
        plt.figure(1)
        plt.title(r'$\chi^2$ vs Amplitude 1 Powerlaw')
        plt.xlabel('Amplitude Powerlaw')
        plt.ylabel(r'$\chi^2$')
        plt.plot(param_test_array, chisqs_gamp1)
        plt.savefig(f'amp1_pwrlaw_fitting|{self.plot_tag}.pdf')
        plt.show()
        plt.close('all')
        print('1st Component Amplitude Powerlaw = ' + str(self.amp1_pwrlaw))
        return(self.amp1_pwrlaw)


    def plot_modeled(self):

        '''Plots and saves the fits for the best fit powerlaws for the intrinsic
        shape.'''

        timer = np.arange(np.shape(self.data)[1])

        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (6,3), sharex = True, sharey = True)

        intrinsic_shapes = np.zeros(np.shape(self.data))

        for ind in range(np.size(self.frequencies)):

            gamp3 = self.comp3[0] * np.power((self.frequencies[ind]/self.sband_freq),self.amp3_pwrlaw)

            gcent3 = (self.comp3[1] - self.comp2[1]) * np.power((self.frequencies[ind]/self.sband_freq),self.phase3_pwrlaw) + self.comp2[1]

            gwidth3 = self.comp3[2] * np.power((self.frequencies[ind]/self.sband_freq),self.width3_pwrlaw)

            gamp1 = self.comp1[0] * np.power((self.frequencies[ind]/self.sband_freq),self.amp1_pwrlaw)

            profile = triple_gauss([gamp3, gcent3, gwidth3], self.comp2, [gamp1,self.comp1[1],self.comp1[2]], timer)[0]

            intrinsic_shapes[ind] = profile

            axs.flat[0].plot(np.linspace(0,1,np.shape(self.data)[1]), profile + (0.005*(ind)) + 0.002, color = 'grey')

        np.savez(f'j1903_modeled_params|{self.plot_tag}', sband_params = [self.comp1, self.comp2, self.comp3], sband_freq = self.sband_freq, amp1_pwrlaw = self.amp1_pwrlaw, amp3_pwrlaw = self.amp3_pwrlaw, width3_pwrlaw = self.width3_pwrlaw, phase3_pwrlaw = self.phase3_pwrlaw)

        frequency_ints = []
        for i in self.frequencies:
            frequency_ints.append(int(np.round(i)))

        axs.flat[0].set_yticks(np.linspace(0.002,(np.size(self.frequencies)-1)*0.005+0.002,np.size(self.frequencies)), frequency_ints, minor = False)
        fig.text(0.517, 0.003, r'Pulse Phase', ha='center', va='center', fontsize = 10)
        axs.flat[0].set_ylabel('Frequency [MHz]', fontsize = 10)

        for ind in range(np.size(self.frequencies)):

            axs.flat[1].plot(np.linspace(0,1,np.shape(self.data)[1]), (intrinsic_shapes[ind] -  intrinsic_shapes[3]) * 10 + (0.005*(ind)) + 0.002, color = 'grey')

        plt.savefig(f'j1903_modeled|{self.plot_tag}.png', dpi = 300, bbox_inches = 'tight')
        plt.savefig(f'j1903_modeled|{self.plot_tag}.pdf', bbox_inches = 'tight')

        plt.close('all')


    def plot_modeled_fitted(self):

        timer = np.arange(np.shape(self.data)[1])

        t = np.linspace(0,1,np.shape(self.data)[1])

        fig, axs = plt.subplots(ncols = 3, nrows = 3, figsize = (14,14), sharex = True, sharey = True)

        test_tau = np.linspace(0.1,400.0,200)

        total_chi_sq = 0

        chi_sq_sum = 0

        tau_values_collect = np.zeros(np.size(self.frequencies))

        for ind in range(np.size(self.frequencies)):

            pbf_chisq = np.zeros(np.size(test_tau))

            for iii in range(np.size(test_tau)):

                gamp3 = self.comp3[0] * np.power((self.frequencies[ind]/self.sband_freq),self.amp3_pwrlaw)

                gcent3 = (self.comp3[1] - self.comp2[1]) * np.power((self.frequencies[ind]/self.sband_freq),self.phase3_pwrlaw) + self.comp2[1]

                gwidth3 = self.comp3[2] * np.power((self.frequencies[ind]/self.sband_freq),self.width3_pwrlaw)

                gamp1 = self.comp1[0] * np.power((self.frequencies[ind]/self.sband_freq),self.amp1_pwrlaw)

                if self.screen == 'thick':
                    #stretch or squeeze pbf, then time average
                    stretch_or_squeeze_factor = test_tau[iii]/self.pbf_tau
                    pulse_broadening = time_average(stretch_or_squeeze(self.pbf, \
                    stretch_or_squeeze_factor), np.shape(self.data)[1])

                elif self.screen == 'thin':
                    closest_tau_ind = find_nearest(self.tau_options, test_tau[iii])[1][0][0]
                    pulse_broadening =  time_average(self.pbf_options[closest_tau_ind], np.shape(self.data)[1])

                elif self.screen == 'exp':
                    t = np.linspace(0, j1903_period, np.shape(self.data)[1], endpoint=False)
                    pulse_broadening = (1.0/test_tau[iii])*np.power(math.e,-1.0*t/test_tau[iii])

                intrinsic = triple_gauss([gamp3, gcent3, gwidth3], self.comp2, [gamp1,self.comp1[1],self.comp1[2]], timer)[0]

                profile = convolve(intrinsic, pulse_broadening)

                sp = SinglePulse(self.data[ind])
                fitting = sp.fitPulse(profile)

                sps = SinglePulse(profile*fitting[2])
                fitted_template = sps.shiftit(fitting[1])

                # fitted_template = profile/trapz(profile)*trapz(self.data[ind])

                chi_sqs = chi2_distance(fitted_template, self.data[ind], self.rms_values[ind], 4)

                pbf_chisq[iii] = chi_sqs

            here = np.where((pbf_chisq == np.min(pbf_chisq)))[0][0]

            chi_sq_sum += pbf_chisq[here]

            tauer = test_tau[here]

            if self.screen == 'thick':
                #stretch or squeeze pbf, then time average
                stretch_or_squeeze_factor = tauer/self.pbf_tau
                final_profiler = time_average(stretch_or_squeeze(self.pbf, \
                stretch_or_squeeze_factor), np.shape(self.data)[1])

            elif self.screen == 'thin':
                closest_tau_ind = find_nearest(self.tau_options, tauer)[1][0][0]
                final_profiler =  time_average(self.pbf_options[closest_tau_ind], np.shape(self.data)[1])

            elif self.screen == 'exp':
                t = np.linspace(0, j1903_period, np.shape(self.data)[1], endpoint=False)
                final_profiler = (1.0/tauer)*np.power(math.e,-1.0*t/tauer)

            tau = calculate_tau(final_profiler)[0]
            tau_values_collect[ind] = tau

            intrinsic = triple_gauss([gamp3, gcent3, gwidth3], self.comp2, [gamp1,self.comp1[1],self.comp1[2]], timer)[0]

            profile = convolve(intrinsic, final_profiler)

            sp = SinglePulse(self.data[ind])
            fitting = sp.fitPulse(profile)

            sps = SinglePulse(profile*fitting[2])
            fitted_template = sps.shiftit(fitting[1])

            # fitted_template = profile/trapz(profile)*trapz(self.data[ind])

            index = ind

            textstr = r'$\tau$'+f' ={int(np.round(tau))}' +r' $\mu$s'
            axs.flat[index].text(0.65, 0.95, textstr, fontsize=14, verticalalignment='top', transform=axs.flat[index].transAxes)

            axs.flat[index].plot(t, self.data[ind], color = 'darkgrey', lw = 2.0)

            axs.flat[index].plot(t, fitted_template, color = 'k')

            axs.flat[index].set_yticks([], minor = False)

            axs.flat[index].set_title(f'{int(np.round(self.frequencies[ind],2))} [MHz]', fontsize = 15)

        # now add sband profile
        ind = np.size(self.frequencies)

        intrinsic = triple_gauss(self.comp1, self.comp2, self.comp3, timer)[0]

        if self.screen == 'thick':
            #stretch or squeeze pbf, then time average
            stretch_or_squeeze_factor = self.sband_tau/self.pbf_tau
            final_profiler = time_average(stretch_or_squeeze(self.pbf, \
            stretch_or_squeeze_factor), np.shape(self.data)[1])

        elif self.screen == 'thin':
            closest_tau_ind = find_nearest(self.tau_options, self.sband_tau)[1][0][0]
            final_profiler =  time_average(self.pbf_options[closest_tau_ind], np.shape(self.data)[1])

        elif self.screen == 'exp':
            t = np.linspace(0, j1903_period, np.shape(self.data)[1], endpoint=False)
            final_profiler = (1.0/self.sband_tau)*np.power(math.e,-1.0*t/self.sband_tau)

        profile = convolve(intrinsic, final_profiler)

        sp = SinglePulse(self.sband_profile)
        fitting = sp.fitPulse(profile)

        sps = SinglePulse(profile*fitting[2])
        fitted_template = sps.shiftit(fitting[1])

        # fitted_template = profile/trapz(profile)*trapz(self.sband_profile)

        index = ind

        textstr = r'$\tau$'+f' ={int(np.round(self.sband_tau))}' +r' $\mu$s'
        axs.flat[index].text(0.65, 0.95, textstr, fontsize=14, verticalalignment='top', transform=axs.flat[index].transAxes)

        axs.flat[index].plot(t, self.sband_profile, color = 'darkgrey', lw = 2.0)

        axs.flat[index].plot(t, fitted_template, color = 'k')

        axs.flat[index].set_yticks([], minor = False)

        axs.flat[index].tick_params(axis='x', which='major', labelsize=12)

        axs.flat[index].set_title(f'{int(np.round(self.sband_freq,2))} [MHz]', fontsize = 15)

        # add sband chi squared to chi squared sum and divide by this total
        # number of frequency channels
        # this profile was fitted with mcmc, not powerlaws, so 9 parameters and
        # take away these 9 degrees of freedom
        chi_sq_sum += chi2_distance(fitted_template, self.sband_profile, calculate_rms(self.sband_profile, np.size(self.sband_profile)//5), 9)

        chi_sq_sum = chi_sq_sum / (np.size(self.frequencies)+1)
        print(f'CHI-SQUARED = {chi_sq_sum}')

        fig.delaxes(axs.flat[8])
        fig.text(0.517, 0.083, r'Pulse Phase', ha='center', va='center', fontsize = 15)

        plt.savefig(f'j1903_modeled_fitted|{self.plot_tag}.png', dpi = 300, bbox_inches = 'tight')
        plt.savefig(f'j1903_modeled_fitted|{self.plot_tag}.pdf', bbox_inches = 'tight')
        plt.close('all')

        # also add sband frequency and tau by concantenating the sband data
        # to the end of the already calculated l-band data

        plt.figure(1)

        tau_frequencies = np.append(self.frequencies, self.sband_freq)
        tau_values_collect = np.append(tau_values_collect, self.sband_tau)

        plt.plot(tau_frequencies, tau_values_collect, color = 'k')
        plt.plot(tau_frequencies, tau_values_collect, '.', color = 'k', label = '$\tau$ Collected')
        # -4.4 tau versus frequency
        plt.plot(tau_frequencies, tau_values_collect[np.size(tau_frequencies)//2]*np.power(tau_frequencies/tau_frequencies[np.size(tau_frequencies)//2],-4.4), ls = '--', label = '-4.4', color = 'grey')
        plt.xlabel(r'$\nu$ [MHz]')
        plt.ylabel(r'$\tau$')
        plt.xscale('log')
        plt.yscale('log')
        title = r'$\tau$ vs $\nu$; $\beta$=' + f'{self.beta}' + '; $\zeta$='+ f'{self.zeta}'
        plt.title(title)
        plt.savefig(f'j1903_modeled_tau|{self.plot_tag}.pdf', bbox_inches = 'tight')
        plt.close('all')

class Intrinsic_Component_Powerlaw_Fit(Intrinsic_Component_Powerlaw_Fit_Per_Epoch):

    def __init__(self, beta, zeta, mjd_tag, sband_param_path, sband_freq, thin_or_thick_medium, data, frequencies, sband_data_profile):

        self.beta = beta
        self.zeta = zeta
        self.screen = thin_or_thick_medium
        if thin_or_thick_medium == 'thick':
            self.pbf = np.load(f'zeta_{zeta}_beta_{beta}_pbf.npy')
            self.pbf_tau = calculate_tau(self.pbf)[0]
        elif thin_or_thick_medium == 'thin':
            self.beta = float(beta)
            self.zeta = float(zeta)
            betas = np.load(Path(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS=2048.npz'))['betas']
            zetas = np.load(Path(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS=2048.npz'))['zetas']
            self.pbf_options = np.load(Path(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS=2048.npz'))['pbfs_unitheight'][np.where((betas==self.beta))[0][0]][np.where((zetas==self.zeta))[0][0]]
            self.tau_options = np.load(Path(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS=2048.npz'))['tau_mus'][np.where((betas==self.beta))[0][0]][np.where((zetas==self.zeta))[0][0]]
            print(self.tau_options)
            plt.figure(1)
            ind = 0
            for i in self.pbf_options:
                if ind%10==0:
                    plt.plot(i)
            plt.show()
            plt.close('all')
        elif thin_or_thick_medium != 'exp':
            valid_thick = '\'thick\' or \'thin\' or \'exp\''
            raise Exception(f'Choose a valid medium thickness: {valid_thick}')

        sband_params = np.load(sband_param_path)['parameters']

        print(sband_params)

        self.comp1 = np.abs(sband_params[:3]) #first comp

        self.comp2 = np.abs([1.0, sband_params[3], sband_params[4]]) # second comp

        self.comp3 = np.abs(sband_params[5:8]) # third comp

        self.sband_tau = sband_params[8]

        self.sband_freq = sband_freq

        self.sband_profile = sband_data_profile

        self.data = data
        self.frequencies = frequencies

        # calculate the rms of the noise for each frequency subaverage
        self.rms_values = np.zeros(np.size(self.frequencies))
        for ii in range(np.size(self.frequencies)):
            self.rms_values[ii] = calculate_rms(self.data[ii], np.size(self.data[ii])//5)

        self.yerr = np.zeros(np.size(self.frequencies))
        for i in range(np.size(self.frequencies)):
            self.yerr[i] = calculate_rms(self.data[i], np.size(self.data[i])//5)

        self.plot_tag = f"FREQ=lband|BETA={beta}" +\
        f"|ZETA={zeta}|SCREEN={thin_or_thick_medium.upper()}|MJD={mjd_tag.upper()}"

        if self.screen == 'exp':
            self.plot_tag = f"FREQ=lband|SCREEN={thin_or_thick_medium.upper()}|MJD={mjd_tag.upper()}"
