from fit_functions import *
from pypulse.singlepulse import SinglePulse
import emcee
import pickle
import corner

class MCMC_Profile_Fit_Per_Epoch:

    # center component is set to an amplitude of 1 realtive to the other
    # component amplitudes
    a2 = 1.0

    # the MCMC test parameters in question in this order for the duration of the
    # code
    labels = [r'$A_1$', r'$\phi_1$', r'$W_1$', r'$\phi_2$', r'$W_2$', r'$A_3$' \
    , r'$\phi_3$', r'$W_3$', r'$\tau$']

    def __init__(self, beta, zeta, mjd, thin_or_thick_medium):

        self.beta = beta
        self.zeta = zeta
        if thin_or_thick_medium == 'thick':
            pbf = np.load(f'zeta_{zeta}_beta_{beta}_pbf.npy')
        elif thin_or_thick_medium == 'thin':
            pbf = np.load(f'zeta_{zeta}_beta_{beta}_thin_screen_pbf.npy')
        else:
            valid_thick = '\'thick\' or \'thin\''
            raise Exception(f'Choose a valid medium thickness: {valid_thick}')

        # load in data
        with open('j1903_sband_data.pkl', 'rb') as fp:
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
        profile = profile_fscrunch(epoch_data['data'])
        self.profile = time_average(profile, np.size(profile)//8)
        self.frequency = np.average(np.asarray(epoch_data['freqs']))

        self.pbf = time_average(pbf, np.size(self.profile))
        self.pbf_tau = calculate_tau(pbf)[0]

        # check profile
        plt.figure(1)
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0,1,np.size(profile)), profile, color = 'k')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Arbitrary Intensity')
        ax.set_title(f'MJD {int(np.round(mjd))} S-band Average')
        fig.show()
        plt.close('all')

        self.yerr = calculate_rms(self.profile)

        self.plot_tag = f"FREQ={np.round(self.frequency)}|BETA={beta}" +\
        f"|ZETA={zeta}|SCREEN={thin_or_thick_medium.upper()}|MJD={int(np.round(mjd))}"

    def ln_likelihood(self, theta, x, y, yerr):
        '''Returns ln(likelihood) for the parameters, theta, which in this case
        are for the threes parameters for each of three intrinsic gaussians and
        tau. The ln(likelihood) is simply the ln() of a gaussian distribution.

        numpy array theta: array of intrinsic component parameters
        numpy array x: time vector
        numpy array y: data vector
        numpy array yerr: error on y data vector'''

        a1, phi1, w1, phi2, w2, a3, phi3, w3, tau = theta
        comp1 = [a1, phi1, w1]
        comp2 = [MCMC_Profile_Fit.a2, phi2, w2]
        comp3 = [a3, phi3, w3]

        profile = convolve(triple_gauss(np.abs(comp1), np.abs(comp2), \
        np.abs(comp3), x)[0], stretch_or_squeeze(self.pbf, \
        np.abs(tau/self.pbf_tau)))
        sp = SinglePulse(self.profile)
        fitting = sp.fitPulse(profile)
        sps = SinglePulse(profile*fitting[2])
        # don't roll the pbf because also fitting for phase!
        model = sps.data

        resids = self.profile - model

        N = len(y)
        lnL = -(N/2)*np.log(2*np.pi) - np.log(self.yerr).sum() - \
        0.5*np.power(resids/self.yerr,2).sum()
        return lnL

    def ln_prior(self, theta):
        '''defines the acceptable probabalistic ranges in which the parameters
        can be expected to fall 100% of the time returns either 0.0 of -np.inf
        depending if the sample parameters are reasonable or not.

        numpy array theta: array of intrinsic component parameters'''

        a1, phi1, w1, phi2, w2, a3, phi3, w3, tau = theta
        #amplitudes relative to a2 which is 1, phases from 0-1, widths relative
        #to percentage of phase
        if 0.005 < a1 < 0.3 and 0.05 < a3 < 0.9 and 0.1 < phi1 < 0.8 and \
        0.1 < phi2 < 0.8 and 0.1 < phi3 < 0.8 and 0.001 < w1 < 0.2 and \
        0.001 < w2 < 0.2 and 0.001 < w3 < 0.2 and 1.0 < tau < 80.0:
            return 0.0
        return -np.inf


    def ln_probability(self, theta, x, y, yerr):
        '''defines probability based upon the established accepted parameter
        ranges and the ln_likelihood function. returns the probability for the
        sample parameters.

        numpy array x: time vector
        numpy array y: data vector
        numpy array yerr: error on y data vector'''

        lp = self.ln_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta, x, y, yerr)


    def plot_fit(self, three_gaussian_parameters, tau):

        '''Plots the fitted model with intrinsic parameters
        'three_gaussian_parameters' and pbf width 'tau'

        1D numpy array three_gaussian_parameters: 3 gaussian parameters
        float tau: tau value to scale self.pbf'''

        phase = np.linspace(0,1,np.size(self.profile))

        comp1 = three_gaussian_parameters[:3]
        comp2 = [MCMC_Profile_Fit.a2, three_gaussian_parameters[3], \
        three_gaussian_parameters[4]]
        comp3 = three_gaussian_parameters[5:8]

        profile_fitted = convolve(triple_gauss(comp1, comp2, comp3, \
        np.arange(np.size(self.profile)))[0], stretch_or_squeeze(self.pbf, \
        tau/self.pbf_tau))

        sp = SinglePulse(self.profile)
        fitting = sp.fitPulse(profile_fitted)

        sps = SinglePulse(profile_fitted*fitting[2])
        fitted_template = sps.shiftit(fitting[1])

        plt.figure(1)
        plt.plot(phase, self.profile/trapz(self.profile), color = 'darkgrey', \
        lw = 2.0)
        plt.plot(phase, fitted_template/trapz(self.profile), color = 'k')
        plt.ylabel('Normalised Flux')
        plt.xlabel('Pulse Phase')
        plt.savefig(f'mcmc_fitted|{self.plot_tag}.pdf')
        plt.show()
        plt.close('all')

    def intrinsic_initial_guess(self, tau_guess):

        '''Returns an array of the initial guess parameters for a 3 gaussian
        component fit to 'profile'. No pulse broadening is included.

        1D numpy array profile: data profile to fit'''

        # guess values of amplitude, phase, and width, respectively, for the 3
        # components (center component height set to one for mcmc fitting)
        sband = np.load('j1903_high_freq_temp_unsmoothed.npy')
        sband = sband / trapz(sband)
        initial_guess = np.zeros(8)
        sp = SinglePulse(sband)
        componented = sp.component_fitting(nmax = 3, full = True)
        sband_comp_params = componented[1]
        #normalize gaussian amplitudes relative to second component amplitude
        sband_comp_params[::3] = sband_comp_params[::3] / sband_comp_params[3]
        for i in range(np.size(sband_comp_params)):
            if i != 3:
                ind = i
                if i > 3:
                    ind = i-1
                initial_guess[ind] = sband_comp_params[i]

        # order components in phase
        fixed_order_for_components_starting_values = np.zeros(9)
        fixed_order_for_components_starting_values[:3] = initial_guess[5:8]
        fixed_order_for_components_starting_values[3:5] = initial_guess[3:5]
        fixed_order_for_components_starting_values[5:8] = initial_guess[:3]
        fixed_order_for_components_starting_values[8] = tau_guess

        return(fixed_order_for_components_starting_values)

    def profile_component_fit(self, numruns, tau_guess, include_plots = True):

        '''Fits the best fit three intrinsic components convolved with a pulse
        broadening function.
        '''

        starting_values = self.intrinsic_initial_guess(tau_guess)
        starting_values = np.abs(starting_values)

        starting_values = np.round(starting_values, 3)

        print(f"Initial Guess Parameters: \n A1 = {starting_values[0]}; " + \
        f"Phi1 = {starting_values[1]}; W1 = {starting_values[2]} \n A2 = 1.0;"+\
        f" Phi2 = {starting_values[3]}; W2 = {starting_values[4]} \n " + \
        f"A3 = {starting_values[5]}; Phi3 = {starting_values[6]}; " + \
        f"W3 = {starting_values[7]} \n Tau = {starting_values[8]}")

        #initializes the MCMC
        pos = np.zeros((72,9))

        #setting starting amp range
        pos[:,::5] = starting_values[::5] + 1e-4 * np.random.randn(72, 2)

        #setting starting phi range
        pos[:,1:4:2] = starting_values[1:4:2] + 1e-4 * np.random.randn(72, 2)
        pos[:,6] = starting_values[6] + 1e-4 * np.random.randn(72)

        #setting starting width range
        pos[:,2] = starting_values[2] + 1e-5 * np.random.randn(72)
        pos[:,4] = starting_values[4] + 1e-5 * np.random.randn(72)
        pos[:,7] = starting_values[7] + 1e-5 * np.random.randn(72)

        pos[:,8] = starting_values[8] + 1e-4 * np.random.randn(72)

        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(nwalkers, ndim, \
        self.ln_probability, \
        args=(np.arange(np.size(self.profile)), self.profile, self.yerr))
        #runs the MCMC
        sampler.run_mcmc(pos, numruns, progress=True);
        if include_plots == True:
            plt.figure(1)
            figure, axes = plt.subplots(np.size(starting_values), figsize = \
            (10,7), sharex=True)
            samples_init = sampler.get_chain()
            for i in range(ndim):
                data = samples_init[:,:,i]
                ax = axes[i]
                ax.set_xlim(0, len(samples_init))
                ax.set_xlabel("Iterations")
                ax.set_ylabel(MCMC_Profile_Fit.labels[i])
                ax.plot(data, 'k', alpha=0.2)
            plt.savefig(f'mcmc_sampling|{self.plot_tag}.pdf')
            plt.show()
            plt.close('all')
        #discards some samples and thins
        auto_corr = sampler.get_autocorr_time()
        print(auto_corr)
        samples = sampler.get_chain(discard=2000, thin=50, flat=True)
        [a1, phi1, w1, phi2, w2, a3, phi3, w3, tau] = np.percentile(samples, \
        50, axis = 0)
        parameters = [a1, phi1, w1, phi2, w2, a3, phi3, w3, tau]

        comp1 = [a1, phi1, w1]
        comp2 = [MCMC_Profile_Fit.a2, phi2, w2]
        comp3 = [a3, phi3, w3]

        [a1_low, phi1_low, w1_low, phi2_low, w2_low, a3_low, phi3_low, w3_low, \
        tau_low] = (np.percentile(samples, 50, axis = 0) - \
        np.percentile(samples, 16, axis = 0))

        [a1_high, phi1_high, w1_high, phi2_high, w2_high, a3_high, phi3_high, \
        w3_high, tau_high] = (np.percentile(samples, 84, axis = 0) - \
        np.percentile(samples, 50, axis = 0))

        parameters_low = [a1_low, phi1_low, w1_low, phi2_low, w2_low, a3_low, \
        phi3_low, w3_low, tau_low]

        parameters_high = [a1_high, phi1_high, w1_high, phi2_high, w2_high, \
        a3_high, phi3_high, w3_high, tau_high]

        if (include_plots == True):

            self.plot_fit(parameters[:8], parameters[8])

            plt.figure(1)
            fig = corner.corner(samples, bins=50, color='dimgrey', smooth=0.6, \
            plot_datapoints=False, plot_density=True, plot_contours=True, \
            fill_contour=False, show_titles=True, labels = \
            MCMC_Profile_Fit.labels)
            plt.savefig(f'mcmc_corner|{self.plot_tag}.pdf')
            fig.show()
            plt.close('all')


        print(f"Fitted Parameters: \n A1 = {parameters[0]}; " + \
        f"Phi1 = {parameters[1]}; W1 = {parameters[2]} \n A2 = 1.0;"+\
        f" Phi2 = {parameters[3]}; W2 = {parameters[4]} \n " + \
        f"A3 = {parameters[5]}; Phi3 = {parameters[6]}; " + \
        f"W3 = {parameters[7]} \n Tau = {parameters[8]}")

        np.savez(f'mcmc_params|{self.plot_tag}', \
        parameters = parameters, params_low_err = parameters_low, \
        params_high_err = parameters_high)

        return(parameters, parameters_low, parameters_high)

class MCMC_Profile_Fit(MCMC_Profile_Fit_Per_Epoch):

    def __init__(self, beta, zeta, thin_or_thick_medium, data_profile, \
    profile_freq, mjd_tag):

        self.beta = beta
        self.zeta = zeta
        if thin_or_thick_medium == 'thick':
            pbf = np.load(f'zeta_{zeta}_beta_{beta}_pbf.npy')
        elif thin_or_thick_medium == 'thin':
            pbf = np.load(f'zeta_{zeta}_beta_{beta}_thin_screen_pbf.npy')
        else:
            valid_thick = '\'thick\' or \'thin\''
            raise Exception(f'Choose a valid medium thickness: {valid_thick}')

        self.mjd = mjd_tag
        self.profile = data_profile
        self.frequency = profile_freq

        self.pbf = time_average(pbf, np.size(self.profile))
        self.pbf_tau = calculate_tau(pbf)[0]

        # check profile
        plt.figure(1)
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0,1,np.size(data_profile)), data_profile, \
        color = 'k')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Arbitrary Intensity')
        ax.set_title(f'MJD {mjd_tag} S-band Average')
        fig.show()
        plt.close('all')

        self.yerr = calculate_rms(self.profile)

        self.plot_tag = f"FREQ={np.round(self.frequency)}|BETA={beta}" +\
        f"|ZETA={zeta}|SCREEN={thin_or_thick_medium.upper()}|MJD={mjd_tag.upper()}"
