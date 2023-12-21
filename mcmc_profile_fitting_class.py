from fit_functions import *
from pypulse.singlepulse import SinglePulse
import emcee
import pickle
import corner
from pathlib import Path

class MCMC_Profile_Fit_Per_Epoch:

    # the MCMC test parameters in question in this order for the duration of the
    # code
    labels = [r'$A_1$', r'$\phi_1$', r'$W_1$', r'$A_2$', r'$\phi_2$', r'$W_2$', r'$A_3$' \
    , r'$\phi_3$', r'$W_3$', r'$\tau$']

    def __init__(self, beta, zeta, mjd, thin_or_thick_medium):

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

        # check profile
        plt.figure(1)
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0,1,np.size(profile)), profile, color = 'k')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Arbitrary Intensity')
        ax.set_title(f'MJD {int(np.round(mjd))} S-band Average')
        fig.show()
        plt.close('all')

        self.yerr = calculate_rms(self.profile, np.size(self.profile)//5)

        self.plot_tag = f"FREQ={np.round(self.frequency)}|BETA={beta}" +\
        f"|ZETA={zeta}|SCREEN={thin_or_thick_medium.upper()}|MJD={int(np.round(mjd))}"

        if self.screen == 'exp':
            self.plot_tag = f"FREQ={np.round(self.frequency)}|SCREEN={thin_or_thick_medium.upper()}|MJD={int(np.round(mjd))}"

    def ln_likelihood(self, theta, x, y, yerr):
        '''Returns ln(likelihood) for the parameters, theta, which in this case
        are for the threes parameters for each of three intrinsic gaussians and
        tau. The ln(likelihood) is simply the ln() of a gaussian distribution.

        numpy array theta: array of intrinsic component parameters
        numpy array x: time vector
        numpy array y: data vector
        numpy array yerr: error on y data vector'''

        a1, phi1, w1, a2, phi2, w2, a3, phi3, w3, tau = theta
        comp1 = [a1, phi1, w1]
        comp2 = [a2, phi2, w2]
        comp3 = [a3, phi3, w3]

        if self.screen == 'thick':
            #stretch or squeeze pbf, then time average
            pbf_test = time_average(stretch_or_squeeze(self.pbf, \
            np.abs(tau/self.pbf_tau)), np.size(self.profile))

        elif self.screen == 'thin':
            closest_tau_ind = find_nearest(self.tau_options, tau)[1][0][0]
            pbf_test = time_average(self.pbf_options[closest_tau_ind], np.size(self.profile))

        elif self.screen == 'exp':
            t = np.linspace(0, j1903_period, np.size(self.profile), endpoint=False)
            pbf_test = (1.0/tau)*np.power(math.e,-1.0*t/tau)

        profile = convolve_same_height_arr1(triple_gauss(np.abs(comp1), np.abs(comp2), \
        np.abs(comp3), x, unit_area=False), pbf_test)

        # don't roll the pbf because also fitting for phase!
        model = profile

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

        a1, phi1, w1, a2, phi2, w2, a3, phi3, w3, tau = theta
        if ((0.005*a2 < a1 < 0.3*a2) and (0.00001 < a2 < 0.015) and \
        (0.05*a2 < a3 < 0.9*a2) and (0.1 < phi1 < 0.8) and \
        (0.1 < phi2 < 0.8) and (0.1 < phi3 < 0.8) and (0.001 < w1 < 0.2) and \
        (0.001 < w2 < 0.2) and (0.001 < w3 < 0.2) and (0.0001 < tau < 80.0)):
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

    def plot_a_fit_sp(self, three_gaussian_parameters, tau):

        '''Plot fitted parameters normalized to second component height.'''

        phase = np.linspace(0,1,np.size(self.profile))

        comp1 = three_gaussian_parameters[:3]
        comp2 = [1.0, three_gaussian_parameters[3], three_gaussian_parameters[4]]
        comp3 = three_gaussian_parameters[5:8]

        if self.screen == 'thick':
            #stretch or squeeze pbf, then time average
            pbf_test = time_average(stretch_or_squeeze(self.pbf, \
            np.abs(tau/self.pbf_tau)), np.size(self.profile))

        elif self.screen == 'thin':
            closest_tau_ind = find_nearest(self.tau_options, tau)[1][0][0]
            pbf_test = time_average(self.pbf_options[closest_tau_ind], np.size(self.profile))

        elif self.screen == 'exp':
            t = np.linspace(0, j1903_period, np.size(self.profile), endpoint=False)
            pbf_test = (1.0/tau)*np.power(math.e,-1.0*t/tau)

        intrinsic = triple_gauss(comp1, comp2, comp3, np.arange(np.size(self.profile)))[0]

        fitted_template = convolve(intrinsic, pbf_test)

        fitted_template = fitted_template/trapz(fitted_template)

        print(f'Fit Chi-Squared = {chi2_distance(fitted_template, self.profile, self.yerr, 10)}')

        plt.figure(1)
        plt.plot(phase, self.profile, color = 'darkgrey', \
        lw = 2.0)
        plt.plot(phase, fitted_template, color = 'k')
        plt.ylabel('Normalised Flux')
        plt.xlabel('Pulse Phase')
        plt.show()
        plt.close('all')

    def plot_fit(self, three_gaussian_parameters, tau):

        '''Plots the fitted model with intrinsic parameters
        'three_gaussian_parameters' and pbf width 'tau'

        1D numpy array three_gaussian_parameters: 3 gaussian parameters
        float tau: tau value to scale self.pbf'''

        phase = np.linspace(0,1,np.size(self.profile))

        comp1 = three_gaussian_parameters[:3]
        comp2 = three_gaussian_parameters[3:6]
        comp3 = three_gaussian_parameters[6:9]

        if self.screen == 'thick':
            #stretch or squeeze pbf, then time average
            pbf_test = time_average(stretch_or_squeeze(self.pbf, \
            np.abs(tau/self.pbf_tau)), np.size(self.profile))

        elif self.screen == 'thin':
            closest_tau_ind = find_nearest(self.tau_options, tau)[1][0][0]
            pbf_test = time_average(self.pbf_options[closest_tau_ind], np.size(self.profile))

        elif self.screen == 'exp':
            t = np.linspace(0, j1903_period, np.size(self.profile), endpoint=False)
            pbf_test = (1.0/tau)*np.power(math.e,-1.0*t/tau)

        fitted_template = convolve_same_height_arr1(triple_gauss(comp1, comp2, \
        comp3, np.arange(np.size(self.profile)), unit_area=False), pbf_test)

        print(f'Fit Chi-Squared = {chi2_distance(fitted_template, self.profile, self.yerr, 10)}')

        plt.figure(1)
        plt.plot(phase, self.profile, color = 'darkgrey', \
        lw = 2.0)
        plt.plot(phase, fitted_template, color = 'k')
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
        sband = self.profile
        sband = sband / trapz(sband)
        sp = SinglePulse(sband)
        componented = sp.component_fitting(nmax = 3, full = True)
        initial_guess = np.zeros(9)
        initial_guess[:] = componented[1]

        # order components in phase
        fixed_order_for_components_starting_values = np.zeros(10)
        fixed_order_for_components_starting_values[:3] = initial_guess[6:9]
        fixed_order_for_components_starting_values[3:6] = initial_guess[3:6]
        fixed_order_for_components_starting_values[6:9] = initial_guess[:3]
        fixed_order_for_components_starting_values[9] = tau_guess

        return(fixed_order_for_components_starting_values)

    def profile_component_fit(self, numruns, tau_guess, include_plots = True):

        '''Fits the best fit three intrinsic components convolved with a pulse
        broadening function.
        '''

        starting_values = self.intrinsic_initial_guess(tau_guess)
        starting_values = np.abs(starting_values)

        starting_values_copy = np.round(starting_values, 5)

        print(f"Initial Guess Parameters: \n A1 = {starting_values_copy[0]}; " + \
        f"Phi1 = {starting_values_copy[1]}; W1 = {starting_values_copy[2]} \n A2 = {starting_values_copy[3]};"+ \
        f" Phi2 = {starting_values_copy[4]}; W2 = {starting_values_copy[5]} \n " + \
        f"A3 = {starting_values_copy[6]}; Phi3 = {starting_values_copy[7]}; " + \
        f"W3 = {starting_values_copy[8]} \n Tau = {starting_values_copy[9]}")

        #initializes the MCMC
        pos = np.zeros((72,10))

        #setting starting amp range
        pos[:,:9:3] = starting_values[:9:3] + 1e-4 * np.random.randn(72, 3)

        #setting starting phi range
        pos[:,1:9:3] = starting_values[1:9:3] + 1e-4 * np.random.randn(72, 3)

        #setting starting width range
        pos[:,2:9:3] = starting_values[2:9:3] + 1e-4 * np.random.randn(72, 3)

        #setting starting tau range
        pos[:,9] = starting_values[9] + 1e-4 * np.random.randn(72)

        if self.screen == 'thin':
            #setting starting tau range
            pos[:,9] = np.abs(starting_values[9] + 2.0*np.random.randn(72)) + 1e-4

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
                ax.set_ylabel(MCMC_Profile_Fit.labels[i])
                ax.plot(data, 'k', alpha=0.2)
            axes[ndim-1].set_xlabel("Iterations")
            plt.savefig(f'mcmc_sampling|{self.plot_tag}.pdf')
            plt.show()
            plt.close('all')
        #discards some samples and thins
        auto_corr = sampler.get_autocorr_time()
        print(auto_corr)
        samples = sampler.get_chain(discard=2000, thin=50, flat=True)
        [a1, phi1, w1, a2, phi2, w2, a3, phi3, w3, tau] = np.percentile(samples, \
        50, axis = 0)
        parameters = [a1/a2, phi1, w1, a2/a2, phi2, w2, a3/a2, phi3, w3, tau]

        #16th percentile for all params, then normalizing the gaussian component
        #amplitudes
        bott_params = np.percentile(samples, 16, axis = 0)
        bott_params[:9:3] = bott_params[:9:3]/bott_params[3]

        [a1_low, phi1_low, w1_low, a2_low, phi2_low, w2_low, a3_low, phi3_low, w3_low, \
        tau_low] = (parameters - bott_params)

        #84th percentile for all params, then normalizing the gaussian component
        #amplitudes
        top_params = np.percentile(samples, 84, axis = 0)
        top_params[:9:3] = top_params[:9:3]/top_params[3]

        [a1_high, phi1_high, w1_high, a2_high, phi2_high, w2_high, a3_high, phi3_high, \
        w3_high, tau_high] = (top_params - parameters)

        parameters_low = [a1_low, phi1_low, w1_low, phi2_low, w2_low, a3_low, \
        phi3_low, w3_low, tau_low]

        parameters_high = [a1_high, phi1_high, w1_high, phi2_high, w2_high, \
        a3_high, phi3_high, w3_high, tau_high]

        parameters = [a1/a2, phi1, w1, phi2, w2, a3/a2, phi3, w3, tau]

        print(f"Fitted Parameters: \n A1 = {parameters[0]}; " + \
        f"Phi1 = {parameters[1]}; W1 = {parameters[2]} \n"+\
        f"Phi2 = {parameters[3]}; W2 = {parameters[4]} \n " + \
        f"A3 = {parameters[5]}; Phi3 = {parameters[6]}; " + \
        f"W3 = {parameters[7]} \n Tau = {parameters[8]}")

        np.savez(f'mcmc_params|{self.plot_tag}', \
        parameters = parameters, params_low_err = parameters_low, \
        params_high_err = parameters_high)

        if (include_plots == True):

            self.plot_fit([a1, phi1, w1, a2, phi2, w2, a3, phi3, w3], tau)

            plt.figure(1)
            fig = corner.corner(samples, bins=50, color='dimgrey', smooth=0.6, \
            plot_datapoints=False, plot_density=True, plot_contours=True, \
            fill_contour=False, show_titles=True, labels = \
            MCMC_Profile_Fit.labels)
            plt.savefig(f'mcmc_corner|{self.plot_tag}.pdf')
            fig.show()
            plt.close('all')

        return(parameters, parameters_low, parameters_high)

class MCMC_Profile_Fit(MCMC_Profile_Fit_Per_Epoch):

    def __init__(self, beta, zeta, thin_or_thick_medium, data_profile, \
    profile_freq, mjd_tag):
        '''pbf must have at least as many phase bins as data profile'''

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

        self.mjd = mjd_tag
        self.profile = data_profile
        self.frequency = profile_freq

        # check profile
        plt.figure(1)
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0,1,np.size(data_profile)), data_profile, \
        color = 'k')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Arbitrary Intensity')
        ax.set_title(f'MJD: {mjd_tag} S-band Average')
        plt.show()
        plt.close('all')

        self.yerr = calculate_rms(self.profile, np.size(self.profile)//5)

        self.plot_tag = f"FREQ={np.round(self.frequency)}|BETA={beta}" +\
        f"|ZETA={zeta}|SCREEN={thin_or_thick_medium.upper()}|MJD={mjd_tag.upper()}"

        if self.screen == 'exp':
            self.plot_tag = f"FREQ={np.round(self.frequency)}|SCREEN={thin_or_thick_medium.upper()}|MJD={mjd_tag.upper()}"
