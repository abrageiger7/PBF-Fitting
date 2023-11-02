from fit_functions import *
from pypulse.singlepulse import SinglePulse
from scipy import optimize


class Correct_DM_From_Scattering_J1903:

    k_DM = 4.14e3

    def __init__(self, frequencies, data, ref_freq, parameters_at_ref_freq, pwr_laws):

        self.frequencies = frequencies
        self.data = data
        self.ref_freq = ref_freq
        self.parameters_at_ref_freq = parameters_at_ref_freq
        self.pwrlaws = pwr_laws
        self.a1_pwrlaw = pwr_laws[0]
        self.a3_pwrlaw = pwr_laws[1]
        self.phi3_pwrlaw = pwr_laws[2]
        self.w3_pwrlaw = pwr_laws[3]

    def best_intrinsic_params(self, frequency):

        a2 = 1.0

        starting_freq = self.ref_freq
        parameters = self.parameters_at_ref_freq
        comp1 = parameters[:3]
        comp2 = [a2, parameters[3], parameters[4]]
        comp3 = parameters[5:8]

        diff_freq_comp1 = comp1
        diff_freq_comp1[0] = comp1[0]*np.power((frequency/starting_freq),self.a1_pwrlaw)
        diff_freq_comp2 = comp2
        diff_freq_comp3 = comp3
        diff_freq_comp3[0] = comp3[0]*np.power((frequency/starting_freq),self.a3_pwrlaw)
        diff_freq_comp3[1] = comp3[1]*np.power((frequency/starting_freq),self.phi3_pwrlaw)
        diff_freq_comp3[2] = comp3[2]*np.power((frequency/starting_freq),self.w3_pwrlaw)

        return(diff_freq_comp1, diff_freq_comp2, diff_freq_comp3)

    def calculate_time_delay_j1903_modeled(self, tau, frequency, beta, zeta, thin_or_thick_medium):

        '''Return the time delay corresponding to the inputed tau in microseconds.'''

        phase_bins = 2048

        if thin_or_thick_medium == 'thick':
            pbf = np.load(f'zeta_{zeta}_beta_{beta}_pbf.npy')
        elif thin_or_thick_medium == 'thin':
            pbf = np.load(f'zeta_{zeta}_beta_{beta}_thin_screen_pbf.npy')
        else:
            valid_thick = '\'thick\' or \'thin\''
            raise Exception(f'Choose a valid medium thickness: {valid_thick}')

        cordes_phase_bins = np.size(pbf)
        subs_time_avg = pbf
        subs_time_avg = np.zeros(phase_bins)

        for ii in range(np.size(subs_time_avg)):
            subs_time_avg[ii] = np.average(pbf[((cordes_phase_bins//phase_bins)*ii):((cordes_phase_bins//phase_bins)*(ii+1))])
        subs_time_avg = subs_time_avg / trapz(subs_time_avg)

        tau_subs_time_avg = calculate_tau(subs_time_avg)[0]

        comp1, comp2, comp3 = self.best_intrinsic_params(frequency)

        intrinsico = triple_gauss(comp1, comp2, comp3, np.arange(phase_bins))[0]

        spi = SinglePulse(intrinsico)

        pbf = stretch_or_squeeze(subs_time_avg, tau/tau_subs_time_avg)

        conv_temp = convolve(intrinsico, pbf).real/trapz(convolve(intrinsico, pbf).real)

        #Calculates mode of data profile to shift template to
        x = np.max(conv_temp)
        xind = np.where(conv_temp == x)[0][0]

        intrinsico = intrinsico / np.max(intrinsico) #fitPulse likes template height of one
        z = np.max(intrinsico)
        zind = np.where(intrinsico == z)[0][0]
        ind_diff = zind-xind

        conv_temp = np.roll(conv_temp, ind_diff)
        sp = SinglePulse(conv_temp)

        fitting = sp.fitPulse(intrinsico)

        toa_delay = ((fitting[1]-ind_diff) * 0.0021499 * 1e6 / phase_bins) #in phase bins

        return(toa_delay)

    def time_delay_delta_DM(delta_dm, v):
        '''v: frequencies'''

        return((Correct_DM_From_Scattering_J1903.k_DM * delta_dm)/(v**2) * 1e6)

    def calc_delta_DM(self, delta_dm_guess, t_inf_guess, full_output=False, plot=True):
        '''Given array of toa delays corresponding to given array of frequencies,
        fit dispersive sweep and return corresponding delta DM. Initialize the
        nonlinear fit with delta_dm_guess and t_inf_guess. Return fitted delta DM.
        If full_output is True, return infinite frequency time of arrival as well.

        float delta_dm_guess: delta DM guess to initialize fit
        float t_inf_guess: infinite frequency arrival time to initialize the fit
        numpy array frequencies: 1D array of frequencies with same size as toa_delays
        numpy array toa_delays: 1D array of toa delays with same size as frequencies.
        '''
        test = [delta_dm_guess, t_inf_guess]

        fit = lambda tests, v: Correct_DM_From_Scattering_J1903.time_delay_delta_DM(tests[0], v) + tests[1] #microseconds
        errfunc = lambda tests, v, y: y - fit(tests, v) #microseconds
        out = optimize.leastsq(errfunc, test[:], args = (self.frequencies,self.toa_delays), full_output=1)

        delta_dm_fitted = out[0][0]
        t_inf_fitted = out[0][1]

        if plot:

            print(f'Delta DM = {delta_dm_fitted} for infinite frequency arrival time {t_inf_fitted}')
            plt.figure(5)
            plt.plot(self.frequencies, (Correct_DM_From_Scattering_J1903.k_DM * delta_dm_fitted)/(self.frequencies**2) * 1e6 + t_inf_fitted, color = 'grey')
            plt.plot(self.frequencies, self.toa_delays, '.', color = 'k')
            plt.xlabel(r'$\nu$ [MHz]')
            plt.ylabel(r'TOA Delay [$\mu$s]')
            plt.title('Fitted Dispersion')
            plt.text(self.frequencies[-5], self.toa_delays[2], r'$\delta$DM = '+f'{np.round(delta_dm_fitted,3)} ' + r'pc/cm$^3$')
            plt.savefig('simulated_scattering_delta_dm_fit.pdf')
            plt.show()
            plt.close('all')

        if full_output:

            return([delta_dm_fitted, t_inf_fitted])

        return(delta_dm_fitted)

    def correct_delta_DM(self, delta_DM):
        '''Given 2D data array corresponding to frequencies, correct by -delta DM
        and return corrected data array. Plots corrected data. The time of arrival delay
        for infinite frequency is zero.

        float delta_DM: DM to correct for in pc/cm^3
        numpy array frequencies: 1D array of frequencies corresponding to data
        numpy array toa_delays: 2D array of data corresponding to frequencies.
        '''

        plt.figure(1)
        plt.imshow(self.data, aspect = 30)
        plt.xlabel('Phase Bins')
        plt.ylabel('Frequency Bins')
        plt.title('Before DM Correction')
        plt.show()
        plt.close('all')

        t_d = Correct_DM_From_Scattering_J1903.time_delay_delta_DM(delta_DM, self.frequencies) # converted to microseconds
        shift = np.round((t_d*(np.shape(self.data)[1]/(0.0021499*1e6)))).astype(int) # converted to bins

        data_corrected = np.zeros(np.shape(self.data))
        for i in range(np.shape(self.data)[0]):
            data_corrected[i] = np.roll(self.data[i],shift[i])

        plt.figure(1)
        plt.imshow(data_corrected, aspect = 30)
        plt.title('After DM Correction')
        plt.show()
        plt.close('all')

        return(data_corrected)

    def correct_data(self, beta, zeta, thin_or_thick_medium):

        self.toa_delays = np.zeros(np.size(self.frequencies))

        # assume Kolmogorov $\tau \propto \nu^{-4.4}$
        x_tau = -4.4
        dm_guess = 0.01
        t_inf_guess = 0.0


        for i in range(np.size(self.frequencies)):

            tau = self.parameters_at_ref_freq[8] * np.power(self.frequencies[i]/self.ref_freq, x_tau)

            self.toa_delays[i] = self.calculate_time_delay_j1903_modeled(tau, self.frequencies[i], beta, zeta, thin_or_thick_medium)

        delta_dm = self.calc_delta_DM(dm_guess, t_inf_guess)

        return(self.correct_delta_DM(delta_dm))
