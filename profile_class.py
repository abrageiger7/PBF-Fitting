import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import numpy as np
import math
from pypulse.singlepulse import SinglePulse
import itertools

from fit_functions import *

plt.rc('font', family = 'serif')

class Profile_Fitting:


    def __init__(self, mjd, data, frequencies, dur, screen, intrinsic_shape, betas, zetas, fitting_profiles, tau_values, intrinsic_fwhms = -1, subaverage=True):
        '''
        mjd (float) - epoch of observation
        data (2D array) - pulse data for epoch
        frequencies (1D array) - frequencies corresponding to the data channels
        dur (float) - observation duration in seconds
        fitting profiles -
        '''

        # make sure inputted intrinsic shape is valid
        if intrinsic_shape != 'gaussian' and intrinsic_shape != 'sband_avg' and intrinsic_shape != 'modeled':
            raise Exception('Please choose a valid intrinsic shape: either gaussian, sband_avg, or modeled.')

        if intrinsic_shape != 'modeled' and np.size(intrinsic_fwhms) == 1:
            raise Exception('Please provide intrinsic widths.')

        #initialize the object attributes
        self.mjd = mjd
        self.mjd_round = int(np.around(mjd))
        self.data_orig = data
        self.freq_orig = frequencies
        self.dur = dur
        self.intrinsic_shape = intrinsic_shape
        self.betas = betas
        self.zetas = zetas
        self.screen = screen


        if intrinsic_shape == 'gaussian' or intrinsic_shape == 'sband_avg':

            self.convolved_profiles = fitting_profiles
            self.tau_values = tau_values
            self.intrinsic_fwhms = intrinsic_fwhms

        elif intrinsic_shape == 'modeled':

            self.pbfs = fitting_profiles
            self.tau_values = tau_values

            print(self.tau_values)


        #subaverages the data for every four frequency channels
        if subaverage==True:
            s = subaverages4(mjd, data, frequencies, phase_bins)
            self.num_sub = len(s[1])
            self.subaveraged_info = s
        else:
            self.num_sub = np.size(frequencies)
            self.subaveraged_info = [data, frequencies]


    def fit_sing(self, profile, num_par):
        '''Fits a data profile to a template
        Helper function for all fitting functions below

        Pre-conditions:
        profile (numpy array): the template
        num_par (int): the number of fitted parameters

        Returns the fit chi-squared value (float)'''

        #decide where to cut off noise depending on the frequency (matches with
        #data as well)

        num_masked = phase_bins - (self.stop_index-self.start_index)

        profile = profile / np.max(profile) #fitPulse requires template height of one
        z = np.max(profile)
        zind = np.where(profile == z)[0][0]
        ind_diff = self.xind-zind
        #this lines the profiles up approximately so that Single Pulse finds the
        #true minimum, not just a local min
        profile = np.roll(profile, ind_diff)

        sp = SinglePulse(self.data_suba, opw = np.arange(0, opr_size))
        fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template
        #matching, scale factor, TOA error, scale factor error, signal to noise
        #ratio, cross-correlation coefficient
        #based on the fitPulse fitting, scale and shift the profile to best fit
        #the inputted data
        #fitPulse figures out the best amplitude itself
        spt = SinglePulse(profile*fitting[2])
        fitted_template = spt.shiftit(fitting[1])

        chi_sq_measure = chi2_distance((self.data_suba*self.mask), (fitted_template*self.mask), self.rms_noise, num_par+num_masked)

        return(chi_sq_measure)


    def chi_plot(self, chi_sq_arr, pbf_type, bzeta = -1, iwidth = -1, pbfwidth = -1):

        '''Plots the inputted chi_sq_arr against the given parameters.

        If iwidth is -1, indicates that the chi-squared surface exists over
        all possible gaussian widths. Same for pbfwidth.

        pbf_type is one of these strings: 'beta', 'zeta', 'exp'

        iwidth and pbfwidth are actual values of intrinsic width and pbf stretch
        factor. bzeta is the actual value of beta or zeta if not exp pbf.'''

        #check of logical function inputs

        if pbf_type != 'beta' and pbf_type != 'zeta' and pbf_type != 'exp':
            raise Exception('Invalid pbf_type.')

        if pbf_type == 'beta' or pbf_type == 'zeta':
            if bzeta == -1:
                raise Exception('Please indicate the b/zeta value used.')

        plt.figure(1)

        #1D Chi-Squared

        if self.intrinsic_shape == 'modeled' or (iwidth != -1 and pbfwidth == -1):

            iwidth_round = int(np.around(iwidth))

            plt.title('Fit Chi-sqs')
            plt.xlabel(r'Tau [$\mu$s]')
            plt.ylabel('Reduced Chi-Sq')

            if pbf_type == 'beta':
                bzeta_ind = np.where((self.betas == bzeta))[0][0];
                plt.plot(self.tau_values[pbf_type][bzeta_ind], chi_sq_arr, drawstyle='steps-pre')

            elif pbf_type == 'zeta':
                bzeta_ind = np.where((self.zetas == bzeta))[0][0];
                plt.plot(self.tau_values[pbf_type][bzeta_ind], chi_sq_arr, drawstyle='steps-pre')

            elif pbf_type == 'exp':
                plt.plot(self.tau_values[pbf_type], chi_sq_arr, drawstyle='steps-pre')

            if (pbf_type == 'beta' or pbf_type == 'zeta'):
                if self.intrinsic_shape == 'modeled':
                    title = f"PBF_fit_chisq|{pbf_type.upper()}={bzeta}|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}.pdf"
                else:
                    title = f"PBF_fit_chisq_setg|{pbf_type.upper()}={bzeta}|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}|IWIDTH={iwidth_round}.pdf"

            elif pbf_type == 'exp':
                if self.intrinisic_shape == 'modeled':
                    title = f"PBF_fit_chisq|{pbf_type.upper()}|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}.pdf"
                else:
                    title = f"PBF_fit_chisq_setg|{pbf_type.upper()}|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}|IWIDTH={iwidth_round}.pdf"

            plt.savefig(title, bbox_inches='tight')
            print(title)
            plt.close('all')

        elif iwidth == -1 and pbfwidth == -1: #neither width set, so 2D chi^2 surface

            plt.title("Reduced Chi-sqs")
            plt.xlabel(r"Gaussian FWHM [$\mu$s]")
            plt.ylabel(r"Tau [$\mu$s]")

            #adjust the imshow tick marks
            gauss_ticks = np.zeros(10)
            for ii in range(10):
                gauss_ticks[ii] = str(self.intrinsic_fwhms[ii*(np.size(self.intrinsic_fwhms)//10)])[:3]

            tau_ticks = np.zeros(10)
            if pbf_type == 'beta' or pbf_type == 'zeta':
                for ii in range(10):
                    tau_ticks[ii] = str(self.tau_values['pbf_type'][bzeta_ind][ii*(np.size(self.tau_values['pbf_type'][bzeta_ind])//10)])[:3]

            elif pbf_type == 'exp':
                for ii in range(10):
                    tau_ticks[ii] = str(self.tau_values['pbf_type'][ii*((np.size(self.tau_values['pbf_type'][bzeta_ind]))//10)])[:3]

            plt.xticks(ticks = np.linspace(0,num_iwidth,num=10), labels = gauss_ticks)
            plt.yticks(ticks = np.linspace(0,num_pbfwidth,num=10), labels = tau_ticks)

            plt.imshow(chi_sq_arr, cmap=plt.cm.viridis_r, origin = 'lower', aspect = 0.25)
            plt.colorbar()

            if pbf_type == 'beta' or 'zeta':
                    title = f"PBF_fit_chisq|{pbf_type.upper()}={bzeta}|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}.pdf"

            elif pbf_type == 'exp':
                    title = f"PBF_fit_chisq|{pbf_type.upper()}|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}.pdf"

            plt.savefig(title, bbox_inches='tight')
            print(title)
            plt.close('all')


    def fit_plot(self, pbf_type, bzeta_ind, pbfwidth_ind, low_chi, iwidth_ind = -1, low_pbf = -1, high_pbf = -1):

        '''Plots and saves the fit of the profile subaveraged data to the
        template indicated by the argument indexes and pbf type.

        bzeta_ind, pbfwidth_ind, iwidth_ind are ints; exp is boolean

        If low_pbf and or high_pbf != -1, plots additional fits in different color.
        These should be used for demonstrating error on tau and how this makes the
        fit different.'''

        #test that arguement combination is logical

        if pbf_type != 'beta' and pbf_type != 'zeta' and pbf_type != 'exp':
            raise Exception('Invalid pbf_type.')

        if pbf_type == 'beta' or pbf_type == 'zeta':
            if bzeta_ind == -1:
                raise Exception('Please indicate the b/zeta index.')
            elif type(bzeta_ind) != int:
                raise Exception('bzeta_ind must be an integer.')

        #depending on pbf type, get profiles and taus
        if self.intrinsic_shape == 'gaussian' or self.intrinsic_shape == 'sband_avg':

            plt.figure(1)
            fig1 = plt.figure(1)
            #Plot Data-model
            frame1=fig1.add_axes((.1,.3,.8,.6))
            #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]

            if pbf_type == 'beta' or pbf_type == 'zeta':

                if pbf_type == 'beta':
                    plt.title(r'Fitted Profile; $\beta$ = ' + f'{self.betas[bzeta_ind]}; ' + '$\zeta$ = 0.0')

                i = self.convolved_profiles[pbf_type][bzeta_ind][pbfwidth_ind][iwidth_ind]
                tau_val = self.tau_values[pbf_type][bzeta_ind][pbfwidth_ind]

                if low_pbf != -1:
                    low_pbf_i = self.convolved_profiles[pbf_type][bzeta_ind][low_pbf][iwidth_ind]
                    tau_val_low = self.tau_values[pbf_type][bzeta_ind][low_pbf]
                if high_pbf != -1:
                    high_pbf_i = self.convolved_profiles[pbf_type][bzeta_ind][high_pbf][iwidth_ind]
                    tau_val_high = self.tau_values[pbf_type][bzeta_ind][high_pbf]

            elif pbf_type == 'exp':

                #plt.title(r'Fitted Profile; Exponential')
                plt.title('Fitted Profile')

                i = self.convolved_profiles[pbf_type][pbfwidth_ind][iwidth_ind]
                tau_val = self.tau_values[pbf_type][pbfwidth_ind]

                if low_pbf != -1:
                    low_pbf_i = self.convolved_profiles[pbf_type][low_pbf][iwidth_ind]
                    tau_val_low = self.tau_values[pbf_type][low_pbf]
                if high_pbf != -1:
                    high_pbf_i = self.convolved_profiles[pbf_type][high_pbf][iwidth_ind]
                    tau_val_high = self.tau_values[pbf_type][high_pbf]

            elif pbf_type == 'zeta':

                #plt.title(r'Fitted Profile; $\zeta$ = ' + f'{zetaselect[bzeta_ind]}; ' + '$\beta$ = 11/3')
                plt.title('Fitted Profile')

        elif self.intrinsic_shape == 'modeled':

            plt.figure(1)
            fig1 = plt.figure(1)
            #Plot Data-model
            frame1=fig1.add_axes((.1,.3,.8,.6))

            if pbf_type == 'beta' or pbf_type == 'zeta':

                pbf = self.pbfs[pbf_type][bzeta_ind][pbfwidth_ind]
                i = convolve(self.intrinsic_model, pbf)
                tau_val = self.tau_values[pbf_type][bzeta_ind][pbfwidth_ind]

                if low_pbf != -1:
                    pbf = self.pbfs[pbf_type][bzeta_ind][low_pbf]
                    low_pbf_i = convolve(self.intrinsic_model, pbf)
                    tau_val_low = self.tau_values[pbf_type][bzeta_ind][low_pbf]
                if high_pbf != -1:
                    pbf = self.pbfs[pbf_type][bzeta_ind][high_pbf]
                    high_pbf_i = convolve(self.intrinsic_model, pbf)
                    tau_val_high = self.tau_values[pbf_type][bzeta_ind][high_pbf]

            elif pbf_type == 'exp':

                pbf = self.pbfs[pbf_type][pbfwidth_ind]
                i = convolve(self.intrinsic_model, pbf)
                tau_val = self.tau_values[pbf_type][pbfwidth_ind]

                if low_pbf != -1:
                    pbf = self.pbfs[pbf_type][low_pbf]
                    low_pbf_i = convolve(self.intrinsic_model, pbf)
                    tau_val_low = self.tau_values[pbf_type][low_pbf]
                if high_pbf != -1:
                    pbf = self.pbfs[pbf_type][high_pbf]
                    high_pbf_i = convolve(self.intrinsic_model, pbf)
                    tau_val_high = self.tau_values[pbf_type][high_pbf]

        profile = i / np.max(i) #fitPulse requires template height of one
        z = np.max(profile)
        zind = np.where(profile == z)[0][0]
        ind_diff = self.xind-zind
        #this lines the profiles up approximately so that Single Pulse finds the
        #true minimum, not just a local min
        profile = np.roll(profile, ind_diff)
        sp = SinglePulse(self.data_suba, opw = np.arange(0, opr_size))
        fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template
        #matching, scale factor, TOA error, scale factor error, signal to noise
        #ratio, cross-correlation coefficient
        #based on the fitPulse fitting, scale and shift the profile to best fit
        #the inputted data
        #fitPulse figures out the best amplitude itself
        spt = SinglePulse(profile*fitting[2])
        fitted_template = spt.shiftit(fitting[1])

        fitted_template = fitted_template*self.mask

        #plot the lower error profile alongside the best fit for comparison
        if low_pbf != -1:

            profilel = low_pbf_i / np.max(low_pbf_i) #fitPulse requires template height of one
            z = np.max(profilel)
            zind = np.where(profilel == z)[0][0]
            ind_diff = self.xind-zind
            #this lines the profiles up approximately so that Single Pulse finds the
            #true minimum, not just a local min
            profilel = np.roll(profilel, ind_diff)
            sp = SinglePulse(self.data_suba, opw = np.arange(0, opr_size))
            fitting = sp.fitPulse(profilel) #TOA cross-correlation, TOA template
            #matching, scale factor, TOA error, scale factor error, signal to noise
            #ratio, cross-correlation coefficient
            #based on the fitPulse fitting, scale and shift the profile to best fit
            #the inputted data
            #fitPulse figures out the best amplitude itself
            spt = SinglePulse(profilel*fitting[2])
            fitted_templatel = spt.shiftit(fitting[1])

            fitted_templatel = fitted_templatel*self.mask


        #plot the upper error profile alongside the best fit for comparison
        if high_pbf != -1:

            profileh = high_pbf_i / np.max(high_pbf_i) #fitPulse requires template height of one
            z = np.max(profileh)
            zind = np.where(profileh == z)[0][0]
            ind_diff = self.xind-zind
            #this lines the profiles up approximately so that Single Pulse finds the
            #true minimum, not just a local min
            profileh = np.roll(profileh, ind_diff)
            sp = SinglePulse(self.data_suba, opw = np.arange(0, opr_size))
            fitting = sp.fitPulse(profileh) #TOA cross-correlation, TOA template
            #matching, scale factor, TOA error, scale factor error, signal to noise
            #ratio, cross-correlation coefficient
            #based on the fitPulse fitting, scale and shift the profile to best fit
            #the inputted data
            #fitPulse figures out the best amplitude itself
            spt = SinglePulse(profileh*fitting[2])
            fitted_templateh = spt.shiftit(fitting[1])

            fitted_templateh = fitted_templateh*self.mask


        plt.ylabel('Pulse Intensity')
        plt.plot(time, self.data_suba*self.mask, '.', ms = '2.4', label = f'{self.freq_round} [MHz] Data', color = 'gray')
        plt.plot(time, fitted_template, label = fr'Best fit; $\tau$ = {int(np.around(tau_val,0))} $\mu$s', color = 'red')
        if high_pbf != -1:
            plt.plot(time, fitted_templateh, alpha = 0.33, label = fr'Upper Error; $\tau$ = {int(np.around(tau_val_high))} $\mu$s', color = 'orange', lw = 3)
        if low_pbf != -1:
            plt.plot(time, fitted_templatel, alpha = 0.33, label = fr'Lower Error; $\tau$ = {int(np.around(tau_val_low))} $\mu$s', color = 'orange', lw = 3)
        plt.plot([], [], ' ', label=fr"min $\chi^2$ = {np.around(low_chi,2)}")
        plt.legend(prop={'size': 7})
        frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
        plt.plot()

        #Residual plot
        difference = np.subtract(self.data_suba*self.mask, fitted_template)
        frame2=fig1.add_axes((.1,.1,.8,.2))
        plt.plot(time, difference, '.', ms = '2.4', color = 'gray')
        plt.xlabel(r'Time [ms]')
        plt.ylabel('Residuals')
        plt.plot()

        tau_round = int(np.around(tau_val))

        if self.intrinsic_shape == 'gaussian' or self.intrinsic_shape == 'sband-avg':
            iwidth_round = int(np.around(self.intrinsic_fwhms[iwidth_ind]))


            if pbf_type == 'beta':
                title = f'PBF_fit_plot|BETA={self.betas[bzeta_ind]}|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}||TAU={tau_round}|IW={iwidth_round}.pdf'
            elif pbf_type == 'exp':
                title = f'PBF_fit_plot|EXP|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}|TAU={tau_round}|IW={iwidth_round}.pdf'
            elif pbf_type == 'zeta':
                title = f'PBF_fit_plot|ZETA={zetaselect[bzeta_ind]}|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}|TAU={tau_round}|IW={iwidth_round}.pdf'

        else:

            if pbf_type == 'beta':
                title = f'PBF_fit_plot|BETA={self.betas[bzeta_ind]}|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}||TAU={tau_round}.pdf'
            elif pbf_type == 'exp':
                title = f'PBF_fit_plot|EXP|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}|TAU={tau_round}.pdf'
            elif pbf_type == 'zeta':
                title = f'PBF_fit_plot|ZETA={self.zetas[bzeta_ind]}|{self.intrinsic_shape.upper()}|MEDIUM={self.screen}|MJD={self.mjd_round}|FREQ={self.freq_round}|TAU={tau_round}.pdf'

        plt.savefig(title, bbox_inches='tight')
        print(title)
        plt.close('all')


    def comp_fse(self, tau):
        '''Calculates the error due to the finite scintile effect. Reference
        Michael's email for details. Tau is the fitted time delay in microseconds.'''

        T = self.dur

        v = self.freq_suba / 1000.0 #GHz
        vd = (c_1)/(2.0*(math.pi)*tau) #MHz
        #td = ((math.sqrt(D*(vd*1000.0)))/v)*(1338.62433862) #seconds
        td = ((math.sqrt(D*(vd)))/v)*(vel_cons/V) #seconds
        nscint = (1.0 + nt*(T/td))*(1.0 + nv*(B/vd))
        error = tau/(math.sqrt(nscint)) #microseconds
        #print(error) -> seems to be a very small number of microseconds (order of 1)
        return(error)


    def init_freq_subint(self, freq_subint_index, pbf_type, bzeta_ind, intrinsic_shape = 'none'):
        '''Initializes variables dependent upon the index of frequency subintegration.
        For use by fitting functions.'''

        #isolate the data profile at the frequency desired for this fit
        self.data_suba = self.subaveraged_info[0][freq_subint_index]
        self.freq_suba = self.subaveraged_info[1][freq_subint_index]
        self.freq_round = int(np.around(self.freq_suba))


        #Calculates mode of data profile to shift template to
        x = np.max(self.data_suba)
        self.xind = np.where(self.data_suba == x)[0][0]

        #Set the offpulse regions to zero for fitting because essentially
        #oscillating there.
        #This region size varies depending on frequency

        mask = np.zeros(phase_bins)

        # if self.freq_suba >= 1600:
        #     self.start_index = int((700/2048)*phase_bins)
        #     self.stop_index = int((1548/2048)*phase_bins)
        # elif self.freq_suba >= 1400 and self.freq_suba < 1600:
        #     self.start_index = int((700/2048)*phase_bins)
        #     self.stop_index = int((1648/2048)*phase_bins)
        # elif self.freq_suba >= 1200 and self.freq_suba < 1400:
        #     self.start_index = int((650/2048)*phase_bins)
        #     self.stop_index = int((1798/2048)*phase_bins)
        # elif self.freq_suba >= 1000 and self.freq_suba < 1200:
        #     self.start_index = int((600/2048)*phase_bins)
        #     self.stop_index = int((1948/2048)*phase_bins)
        self.start_index = 0
        self.stop_index = phase_bins
        mask[self.start_index:self.stop_index] = 1.0

        self.bin_num_care = self.stop_index-self.start_index

        self.mask = mask


        #Calculates the root mean square noise of the off pulse.
        #Used later to calculate normalized chi-squared.

        self.rms_noise = calculate_rms(self.data_suba, opr_size)


        if self.intrinsic_shape == 'modeled':

            # thick beta = 3.667 zeta = 0.0 as of 10/18/23
            # (all powerlaws with error +- 0.1, except amp1 which is +-0.05)
            # self.sband_freq = 2132.0
            # self.comp1_amp_sband = 0.054
            # self.comp1_amp_pwrlaw = 1.08
            # self.comp3_amp_sband = 0.332
            # self.comp3_amp_pwrlaw = 0.0
            # self.comp3_width_sband = 0.020
            # self.comp3_width_pwrlaw = -0.9
            # self.comp3_mean_sband = 0.540
            # self.comp3_mean_pwrlaw = 0.0

            if pbf_type == 'exp':
                params = np.load('j1903_modeled_params|FREQ=lband|SCREEN=EXP|MJD=MJD_AVERAGE.npz')
            elif bzeta_ind == -1:
                params = np.load('j1903_modeled_params|FREQ=lband|' + intrinsic_shape + '.npz')
            elif pbf_type == 'beta' and self.screen == 'thin':
                params = np.load(f'j1903_modeled_params|FREQ=lband|BETA={self.betas[bzeta_ind]}|ZETA=0.01|SCREEN={str(self.screen).upper()}|MJD=MJD_AVERAGE.npz')
            elif pbf_type == 'beta' and self.screen == 'thick':
                params = np.load(f'j1903_modeled_params|FREQ=lband|BETA={self.betas[bzeta_ind]}|ZETA=0.0|SCREEN={str(self.screen).upper()}|MJD=MJD_AVERAGE.npz')
            elif pbf_type == 'zeta':
                params = np.load(f'j1903_modeled_params|FREQ=lband|BETA=3.667|ZETA={self.zetas[bzeta_ind]}|SCREEN={str(self.screen).upper()}|MJD=MJD_AVERAGE.npz')

            comp1 = params['sband_params'][0]
            comp2 = params['sband_params'][1]
            comp3 = params['sband_params'][2]
            sband_freq = params['sband_freq']
            amp1_pwrlaw = params['amp1_pwrlaw']
            amp3_pwrlaw = params['amp3_pwrlaw']
            phase3_pwrlaw = params['phase3_pwrlaw']
            width3_pwrlaw = params['width3_pwrlaw']

            if self.freq_suba > 1800.0:
                amp1_pwrlaw = 0.0
                amp3_pwrlaw = 0.0
                phase3_pwrlaw = 0.0
                width3_pwrlaw = 0.0

            comp1[0] = comp1[0] * np.power(self.freq_suba/sband_freq, amp1_pwrlaw)
            comp3[0] = comp3[0] * np.power(self.freq_suba/sband_freq, amp3_pwrlaw)
            comp3[1] = (comp3[1]-comp2[1]) * np.power(self.freq_suba/sband_freq, phase3_pwrlaw) + comp2[1]
            comp3[2] = comp3[2] * np.power(self.freq_suba/sband_freq, width3_pwrlaw)

            self.intrinsic_model = triple_gauss(comp1,comp2,comp3,t)[0]


    def fit(self, freq_subint_index, pbf_type, bzeta_ind = -1, iwidth_ind = -1, pbfwidth_ind = -1, intrinsic_shape = 'none'):
        '''Calculates the best broadening function and corresponding parameters
        for the Profile object. (no chi-squared plots generated for now)

        pbf_type (str): either 'beta', 'exp', or 'zeta'
        bzeta_ind (int): if nonzero, set beta to this index of betaselect
        iwidth_ind (int): if nonzero, set intrins width to this index of intrinss_fwhm
        pbfwidth_ind (int) : if nonzero, set pbf width to this index of widths

        No error calculations for varying more than one parameter
        '''

        if pbf_type != 'beta' and pbf_type != 'zeta' and pbf_type != 'exp':
            raise Exception('Invalid pbf_type.')

        if pbf_type == 'beta' or pbf_type == 'zeta':
            if type(bzeta_ind) != int:
                raise Exception('bzeta_ind must be an integer.')


        self.init_freq_subint(freq_subint_index, pbf_type, bzeta_ind, intrinsic_shape)

        #number of each parameter in the parameter grid
        num_beta = np.size(self.betas)
        num_zeta = np.size(self.zetas)
        if pbf_type == 'beta' or pbf_type == 'zeta':
            num_taus = np.size(self.tau_values[pbf_type][0])
        else:
            num_taus = np.size(self.tau_values[pbf_type])

        beta_inds = np.arange(num_beta)
        zeta_inds = np.arange(num_zeta)
        tau_inds = np.arange(num_taus)

        #set more convenient names for the data to be fitted to
        data_care = self.data_suba
        freq_care = self.freq_suba

        print(f'Intrinsic shape is {self.intrinsic_shape}.')

        num_par = 2

        for i in [bzeta_ind, iwidth_ind, pbfwidth_ind]:

            if i == -1:
                num_par += 1


#===============================================================================
# modeled intrinsic shape
# ==============================================================================


        if self.intrinsic_shape == 'modeled':

            if pbf_type == 'beta':

                #case where beta and pbfwidth are free, intrinsic shape is set
                if pbfwidth_ind == -1 and bzeta_ind == -1:

                    print(f'Fitting for Beta and tau.')

                    chi_sqs_array = np.zeros((num_beta, num_taus))
                    for i in itertools.product(beta_inds, tau_inds):

                        pbf = self.pbfs[pbf_type][i[0]][i[1]]

                        template = convolve(self.intrinsic_model, pbf)

                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i[0]][i[1]] = chi_sq

                    chi_sqs_collect = np.zeros(num_beta)
                    taus_collect = np.zeros(num_beta)
                    taus_err_collect = np.zeros((2,num_beta))
                    pbf_width_ind_collect = np.zeros(num_beta)
                    ind = 0
                    for i in chi_sqs_array:

                        beta = self.betas[ind]

                        #least squares
                        low_chi = find_nearest(i, 0.0)[0]
                        chi_sqs_collect[ind] = low_chi

                        if i[0] < low_chi+(1/(self.bin_num_care-num_par)) or i[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                            #raise Exception('NOT CONVERGING ENOUGH')
                            print('WARNING: PBF WIDTH NOT CONVERGING')

                        lsqs_pbf_index = find_nearest(i, 0.0)[1][0][0]

                        pbf_width_ind_collect[ind] = int(lsqs_pbf_index)

                        taus_collect[ind] = self.tau_values[pbf_type][ind][lsqs_pbf_index]

                        ind+=1

                    low_chi = np.min(chi_sqs_collect)
                    chi_beta_ind = np.where(chi_sqs_collect == low_chi)[0][0]

                    beta_fin = self.betas[chi_beta_ind]
                    tau_fin = taus_collect[chi_beta_ind]

                    #plotting the fit parameters over beta
                    for i in range(2):

                        plt.figure(1)
                        plt.xlabel('Beta')

                        if i == 0:
                            plt.ylabel('Chi-Squared')
                            plt.plot(self.betas, chi_sqs_collect)
                            param = 'chisqs'

                        if i == 1:
                            plt.ylabel('Overall Best Tau (microseconds)')
                            plt.plot(self.betas, taus_collect)
                            param = 'tau'

                        title = f'ALLBETA|MODELLED|PBF_fit_overall_{param}|MJD={self.mjd_round}|FREQ={self.freq_round}|bestBETA={self.betas[chi_beta_ind]}.pdf'
                        plt.savefig(title, bbox_inches='tight')
                        plt.close('all')

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width

                    if chi_sqs_collect[0] > low_chi+(1/(self.bin_num_care-num_par)) and chi_sqs_collect[-1] > low_chi+(1/(self.bin_num_care-num_par)):
                        below = find_nearest(chi_sqs_collect[:chi_beta_ind], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                        above = find_nearest(chi_sqs_collect[chi_beta_ind+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + chi_beta_ind + 1
                        beta_low = beta_fin - self.betas[below]
                        beta_up = self.betas[above] - beta_fin
                    else:
                        beta_low = 0
                        beta_up = 0

                    #overall best fit plot
                    self.fit_plot(pbf_type, int(chi_beta_ind), int(pbf_width_ind_collect[chi_beta_ind]), low_chi = low_chi)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns[f'{pbf_type}'] = beta_fin
                    data_returns['beta_low'] = beta_low
                    data_returns['beta_up'] = beta_up

                    return(data_returns)

                #case where pbfwidth are free, intrinsic shape and beta is set
                elif pbfwidth_ind == -1 and bzeta_ind != -1:

                    print(f'Set beta = {self.betas[bzeta_ind]} and fitting for tau.')

                    beta = self.betas[bzeta_ind]

                    chi_sqs_array = np.zeros(num_taus)
                    for i in tau_inds:

                        pbf = self.pbfs[pbf_type][bzeta_ind][i]

                        template = convolve(self.intrinsic_model, pbf)

                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i] = chi_sq

                    low_chi = find_nearest(chi_sqs_array, 0.0)[0]

                    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]

                    tau_fin = self.tau_values[pbf_type][bzeta_ind][lsqs_pbf_index]

                    self.chi_plot(chi_sqs_array, pbf_type, bzeta = beta)

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width
                    tau_arr = self.tau_values[pbf_type][bzeta_ind]

                    if chi_sqs_array[0] < low_chi+(1/(self.bin_num_care-num_par)):
                        print('WARNING: PBF WIDTH NOT CONVERGING ENOUGH')
                        above = find_nearest(chi_sqs_array[lsqs_pbf_index+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + lsqs_pbf_index + 1
                        below = 0
                        tau_low = tau_fin - 0.0
                        tau_up = tau_arr[above] - tau_fin

                    elif chi_sqs_array[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                        print('WARNING: PBF WIDTH NOT CONVERGING ENOUGH')
                        below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                        above = -1
                        tau_low = tau_fin - tau_arr[below]
                        tau_up = tau_arr[above] - tau_fin

                    else:
                        below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                        above = find_nearest(chi_sqs_array[lsqs_pbf_index+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + lsqs_pbf_index + 1
                        tau_low = tau_fin - tau_arr[below]
                        tau_up = tau_arr[above] - tau_fin

                    self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, low_chi = low_chi, low_pbf = below, high_pbf = above)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns[f'{pbf_type}'] = beta
                    data_returns['tau_low'] = tau_low
                    data_returns['tau_up'] = tau_up

                    return(data_returns)

            elif pbf_type == 'exp':

                #case where fitting for pbf width and intrinsic shape is modelled
                if pbfwidth_ind == -1:

                    print(f'Decaying exponential pbf and fitting for tau.')

                    chi_sqs_array = np.zeros(num_taus)
                    for i in tau_inds:

                        pbf = self.pbfs[pbf_type][i]

                        template = convolve(self.intrinsic_model, pbf)

                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i] = chi_sq

                    low_chi = find_nearest(chi_sqs_array, 0.0)[0]

                    if chi_sqs_array[0] < low_chi+(1/(self.bin_num_care-num_par)) or chi_sqs_array[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                        print('WARNING: PBF WIDTH NOT CONVERGING ENOUGH')

                    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]

                    tau_fin = self.tau_values[pbf_type][lsqs_pbf_index]

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width
                    below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                    above = find_nearest(chi_sqs_array[lsqs_pbf_index+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + lsqs_pbf_index + 1

                    tau_low = tau_fin - self.tau_values[pbf_type][below]
                    tau_up = self.tau_values[pbf_type][above] - tau_fin

                    self.fit_plot(pbf_type, 0, lsqs_pbf_index, low_chi = low_chi, low_pbf = below, high_pbf = above)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns[f'{pbf_type}'] = 'exp'
                    data_returns['tau_low'] = tau_low
                    data_returns['tau_up'] = tau_up

                    return(data_returns)

            elif pbf_type == 'zeta':

                #case where beta and pbfwidth are free, intrinsic shape is set
                if pbfwidth_ind == -1 and bzeta_ind == -1:

                    print(f'Fitting for zeta and tau.')

                    chi_sqs_array = np.zeros((num_zeta, num_taus))
                    for i in itertools.product(zeta_inds, tau_inds):

                        pbf = self.pbfs[pbf_type][i[0]][i[1]]

                        template = convolve(self.intrinsic_model, pbf)

                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i[0]][i[1]] = chi_sq

                    chi_sqs_collect = np.zeros(num_zeta)
                    taus_collect = np.zeros(num_zeta)
                    taus_err_collect = np.zeros((2,num_zeta))
                    pbf_width_ind_collect = np.zeros(num_beta)
                    ind = 0
                    for i in chi_sqs_array:

                        zeta = self.zetas[ind]

                        #least squares
                        low_chi = find_nearest(i, 0.0)[0]
                        chi_sqs_collect[ind] = low_chi

                        lsqs_pbf_index = find_nearest(i, 0.0)[1][0][0]
                        pbf_width_ind_collect[ind] = int(lsqs_pbf_index)

                        if i[0] < low_chi+(1/(self.bin_num_care-num_par)) or i[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                            #raise Exception('NOT CONVERGING ENOUGH')
                            print('WARNING: PBF WIDTH NOT CONVERGING')

                        taus_collect[ind] = self.tau_values[pbf_type][ind][lsqs_pbf_index]

                        ind+=1

                    low_chi = np.min(chi_sqs_collect)
                    chi_zeta_ind = np.where(chi_sqs_collect == low_chi)[0][0]

                    zeta_fin = self.zetas[chi_zeta_ind]
                    tau_fin = taus_collect[chi_zeta_ind]

                    #plotting the fit parameters over beta
                    for i in range(2):

                        plt.figure(1)
                        plt.xlabel('Beta')

                        if i == 0:
                            plt.ylabel('Chi-Squared')
                            plt.plot(self.zetas, chi_sqs_collect)
                            param = 'chisqs'

                        if i == 1:
                            plt.ylabel('Overall Best Tau (microseconds)')
                            plt.plot(self.zetas, taus_collect)
                            param = 'tau'

                        title = f'ALLZETA|MODELLED|PBF_fit_overall_{param}|MJD={self.mjd_round}|FREQ={self.freq_round}|bestZETA={self.zetas[chi_zeta_ind]}.pdf'
                        plt.savefig(title, bbox_inches='tight')
                        plt.close('all')

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width

                    if chi_sqs_collect[0] > low_chi+(1/(self.bin_num_care-num_par)) and chi_sqs_collect[-1] > low_chi+(1/(self.bin_num_care-num_par)):
                        below = find_nearest(chi_sqs_collect[:chi_zeta_ind], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                        above = find_nearest(chi_sqs_collect[chi_zeta_ind+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + chi_zeta_ind + 1
                        zeta_low = zeta_fin - self.zetas[below]
                        zeta_up = self.zetas[above] - zeta_fin
                    else:
                        zeta_low = 0
                        zeta_up = 0

                    #overall best fit plot
                    self.fit_plot(pbf_type, int(chi_zeta_ind), int(pbf_width_ind_collect[chi_zeta_ind]), low_chi = low_chi)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns[f'{pbf_type}'] = zeta_fin
                    data_returns['zeta_low'] = zeta_low
                    data_returns['zeta_up'] = zeta_up

                    return(data_returns)

                #case where pbfwidth are free, intrinsic shape and beta is set
                elif pbfwidth_ind == -1 and bzeta_ind != -1:

                    print(f'Set zeta = {self.zetas[bzeta_ind]} and fitting for tau.')

                    zeta = self.zetas[bzeta_ind]

                    chi_sqs_array = np.zeros(num_taus)
                    for i in tau_inds:

                        pbf = self.pbfs[pbf_type][bzeta_ind][i]

                        template = convolve(self.intrinsic_model, pbf)

                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i] = chi_sq

                    low_chi = find_nearest(chi_sqs_array, 0.0)[0]

                    if chi_sqs_array[0] < low_chi+(1/(self.bin_num_care-num_par)) or chi_sqs_array[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                        print('WARNING: PBF WIDTH NOT CONVERGING ENOUGH')

                    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]

                    tau_fin = self.tau_values[pbf_type][bzeta_ind][lsqs_pbf_index]

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width

                    below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                    above = find_nearest(chi_sqs_array[lsqs_pbf_index+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + lsqs_pbf_index + 1

                    tau_arr = self.tau_values[pbf_type][bzeta_ind]
                    tau_low = tau_fin - tau_arr[below]
                    tau_up = tau_arr[above] - tau_fin

                    self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, low_chi = low_chi, low_pbf = below, high_pbf = above)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns[f'{pbf_type}'] = zeta
                    data_returns['tau_low'] = tau_low
                    data_returns['tau_up'] = tau_up

                    return(data_returns)


#===============================================================================
# gaussian or sband intrinsic shape
# ==============================================================================


        elif self.intrinsic_shape == 'gaussian' or self.intrinsic_shape == 'sband_avg':

            num_iwidth = np.size(self.intrinsic_fwhms)
            iwidth_inds = np.arange(num_iwidth)


            if pbf_type == 'beta':

                #case where beta, iwidth, and pbfwidth are free
                if iwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind == -1:

                    print('Fitting for beta, intrinsic width, and tau.')

                    chi_sqs_array = np.zeros((num_beta, num_taus, num_iwidth))
                    for i in itertools.product(beta_inds, tau_inds, iwidth_inds):

                        template = self.convolved_profiles[pbf_type][i[0]][i[1]][i[2]]
                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i[0]][i[1]][i[2]] = chi_sq

                    chi_sqs_collect = np.zeros(num_beta)
                    intrinsic_width_collect = np.zeros(num_beta)
                    taus_collect = np.zeros(num_beta)
                    taus_err_collect = np.zeros((2,num_beta))
                    ind = 0
                    for i in chi_sqs_array:

                        #least squares
                        low_chi = find_nearest(i, 0.0)[0]
                        chi_sqs_collect[ind] = low_chi

                        if i[0][0] < low_chi+1 and i[-1][-1] < low_chi+1:
                            print('Warning: NOT CONVERGING ENOUGH') #stops code if not
                            #enough parameters to reach reduced low_chi + 1 before end
                            #of parameter space

                        lsqs_pbf_index = find_nearest(i, 0.0)[1][0][0]

                        #lsqs intrinsic width
                        lsqs_intrins_index = find_nearest(i, 0.0)[1][1][0]
                        lsqs_intrins_val = self.intrinsic_fwhms[lsqs_intrins_index]
                        intrinsic_width_collect[ind] = lsqs_intrins_val

                        taus_collect[ind] = self.tau_values[pbf_type][ind][lsqs_pbf_index]

                        ind+=1

                    low_chi = np.min(chi_sqs_collect)
                    chi_beta_ind = np.where(chi_sqs_collect == low_chi)[0][0]

                    beta_fin = self.betas[chi_beta_ind]
                    intrins_width_fin = self.intrinsic_fwhms[chi_beta_ind]
                    tau_fin = taus_collect[chi_beta_ind]

                    intrins_width_ind = np.where((self.intrinsic_fwhms == intrinsic_width_collect[chi_beta_ind]))[0][0]

                    #plotting the fit parameters over beta
                    for i in range(3):

                        plt.figure(1)
                        plt.xlabel('Beta')

                        if i == 0:
                            plt.ylabel('Chi-Squared')
                            plt.plot(self.betas, chi_sqs_collect)
                            param = 'chisqs'

                        if i == 2:
                            plt.ylabel('Overall Best Intrinsic Width FWHM (milliseconds)')
                            plt.plot(self.betas, intrinsic_width_collect) #already converted to micro fwhm
                            param = 'iwidth'

                        if i == 3:
                            plt.ylabel('Overall Best Tau (microseconds)')
                            plt.plot(self.betas, taus_collect)
                            param = 'tau'

                        title = f'ALLBETA|{self.intrinsic_shape.upper()}|PBF_fit_overall_{param}|MJD={self.mjd_round}|FREQ={self.freq_round}|bestBETA={self.betas[chi_beta_ind]}.pdf'
                        plt.savefig(title, bbox_inches='tight')
                        plt.close('all')

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width

                    if chi_sqs_collect[0] > low_chi+(1/(self.bin_num_care-num_par)) and chi_sqs_collect[-1] > low_chi+(1/(self.bin_num_care-num_par)):
                        below = find_nearest(chi_sqs_collect[:chi_beta_ind], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                        above = find_nearest(chi_sqs_collect[chi_beta_ind+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + chi_beta_ind + 1
                        beta_low = beta_fin - self.betas[below]
                        beta_up = self.betas[above] - beta_fin
                    else:
                        beta_low = 0
                        beta_up = 0

                    #overall best fit plot
                    self.fit_plot(pbf_type, int(chi_beta_ind), pbf_width_ind, low_chi = low_chi, iwidth_ind = intrins_width_ind)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns['intrins_width'] = intrins_width_fin
                    data_returns[f'{pbf_type}'] = beta_fin
                    data_returns['beta_low'] = beta_low
                    data_returns['beta_up'] = beta_up

                    return(data_returns)

                #case where beta and pbfwidth are free, iwidth is set
                if iwidth_ind != -1 and pbfwidth_ind == -1 and bzeta_ind == -1:

                    iwidth = self.intrinsic_fwhms[iwidth_ind]

                    print(f'Set intrinsic width to {iwidth} microseconds. Fitting for beta and tau.')

                    chi_sqs_array = np.zeros((num_beta, num_taus))
                    for i in itertools.product(beta_inds, tau_inds):

                        template = self.convolved_profiles[pbf_type][i[0]][i[1]][iwidth_ind]
                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i[0]][i[1]] = chi_sq

                    chi_sqs_collect = np.zeros(num_beta)
                    taus_collect = np.zeros(num_beta)
                    taus_err_collect = np.zeros((2,num_beta))
                    ind = 0
                    for i in chi_sqs_array:

                        beta = self.betas[ind]

                        #least squares
                        low_chi = find_nearest(i, 0.0)[0]
                        chi_sqs_collect[ind] = low_chi

                        lsqs_pbf_index = find_nearest(i, 0.0)[1][0][0]

                        if i[0] < low_chi+(1/(self.bin_num_care-num_par)) or i[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                            print('WARNING: TAU NOT CONVERGING')

                        taus_collect[ind] = self.tau_values[pbf_type][ind][lsqs_pbf_index]

                        ind+=1

                    low_chi = np.min(chi_sqs_collect)
                    chi_beta_ind = np.where(chi_sqs_collect == low_chi)[0][0]

                    beta_fin = self.betas[chi_beta_ind]
                    tau_fin = taus_collect[chi_beta_ind]

                    #plotting the fit parameters over beta
                    for i in range(2):

                        plt.figure(1)
                        plt.xlabel('Beta')

                        if i == 0:
                            plt.ylabel('Chi-Squared')
                            plt.plot(self.betas, chi_sqs_collect)
                            param = 'chisqs'

                        if i == 1:
                            plt.ylabel('Overall Best Tau (microseconds)')
                            plt.plot(self.betas, taus_collect)
                            param = 'tau'

                        title = f'ALLBETA|{self.intrinsic_shape.upper()}|PBF_fit_overall_{param}|MJD={self.mjd_round}|FREQ={self.freq_round}|set_ifwhm_mus={int(np.round(iwidth))}|bestBETA={self.betas[chi_beta_ind]}.pdf'
                        plt.savefig(title, bbox_inches='tight')
                        plt.close('all')

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width

                    if chi_sqs_collect[0] > low_chi+(1/(self.bin_num_care-num_par)) and chi_sqs_collect[-1] > low_chi+(1/(self.bin_num_care-num_par)):
                        below = find_nearest(chi_sqs_collect[:chi_beta_ind], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                        above = find_nearest(chi_sqs_collect[chi_beta_ind+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + chi_beta_ind + 1
                        beta_low = beta_fin - self.betas[below]
                        beta_up = self.betas[above] - beta_fin
                    else:
                        beta_low = 0
                        beta_up = 0

                    #overall best fit plot
                    self.fit_plot(pbf_type, int(chi_beta_ind), pbf_width_ind, low_chi = low_chi, iwidth_ind = iwidth_ind)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns['intrins_width_set'] = iwidth
                    data_returns[f'{pbf_type}'] = beta_fin
                    data_returns['beta_low'] = beta_low
                    data_returns['beta_up'] = beta_up

                    return(data_returns)

                #case where beta is set, but iwidth and pbfwidth free
                elif iwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                    print(f'Set beta = {self.betas[bzeta_ind]} and fitting for intrinsic width and tau.')

                    beta = self.betas[bzeta_ind]

                    chi_sqs_array = np.zeros((num_taus, num_iwidth))
                    for i in itertools.product(tau_inds, iwidth_inds):

                        template = self.convolved_profiles[pbf_type][bzeta_ind][i[0]][i[1]]
                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i[0]][i[1]] = chi_sq

                    #least squares
                    low_chi = find_nearest(chi_sqs_array, 0.0)[0]

                    if chi_sqs_array[0][0] < low_chi+(1/(self.bin_num_care-num_par)) or chi_sqs_array[-1][-1] < low_chi+(1/(self.bin_num_care-num_par)):
                        raise Exception('NOT CONVERGING ENOUGH') #stops code if not
                        #enough parameters to reach reduced low_chi + 1 before end
                        #of parameter space

                    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]

                    #lsqs intrinsic width
                    lsqs_intrins_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
                    lsqs_intrins_val = self.intrinsic_fwhms[lsqs_intrins_index]

                    tau_fin = self.tau_values[pbf_type][bzeta_ind][lsqs_pbf_index]

                    self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, low_chi = low_chi, iwidth_ind = lsqs_intrins_index)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns['intrins_width'] = lsqs_intrins_val
                    data_returns[f'{pbf_type}'] = beta

                    return(data_returns)

                #case where beta and intrinsix width are set, but fitting for pbf width
                elif iwidth_ind != -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                    print(f'Set beta = {self.betas[bzeta_ind]} and intrinsic width = {self.intrinsic_fwhms[iwidth_ind]} microseconds and fitting for tau.')

                    beta = self.betas[bzeta_ind]
                    iwidth = self.intrinsic_fwhms[iwidth_ind]

                    chi_sqs_array = np.zeros(num_taus)
                    for i in tau_inds:

                        template = self.convolved_profiles[pbf_type][bzeta_ind][i][iwidth_ind]
                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i] = chi_sq

                    #self.chi_plot(chi_sqs_array, pbf_type, bzeta = beta, iwidth = iwidth)

                    low_chi = find_nearest(chi_sqs_array, 0.0)[0]
                    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]

                    if chi_sqs_array[0] < low_chi+(1/(self.bin_num_care-num_par)) or chi_sqs_array[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                        raise Exception('NOT CONVERGING ENOUGH')

                    tau_fin = self.tau_values[pbf_type][bzeta_ind][lsqs_pbf_index]

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width
                    below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                    above = find_nearest(chi_sqs_array[lsqs_pbf_index+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + lsqs_pbf_index + 1

                    tau_arr = self.tau_values[pbf_type][bzeta_ind]
                    tau_low = tau_fin - tau_arr[below]
                    tau_up = tau_arr[above] - tau_fin

                    self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, low_chi = low_chi, iwidth_ind = iwidth_ind, low_pbf = below, high_pbf = above)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns['intrins_width'] = iwidth
                    data_returns[f'{pbf_type}'] = beta
                    data_returns['tau_low'] = tau_low
                    data_returns['tau_up'] = tau_up

                    return(data_returns)


            elif pbf_type == 'zeta':

                #case where zeta, iwidth, and pbfwidth are free
                if iwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind == -1:

                    print('Fitting for zeta, intrinsic width, and tau.')

                    chi_sqs_array = np.zeros((num_zeta, num_taus, num_iwidth))
                    for i in itertools.product(zeta_inds, tau_inds, iwidth_inds):

                        template = self.convolved_profiles[pbf_type][i[0]][i[1]][i[2]]
                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i[0]][i[1]][i[2]] = chi_sq

                    chi_sqs_collect = np.zeros(num_zeta)
                    intrinsic_width_collect = np.zeros(num_zeta)
                    taus_collect = np.zeros(num_zeta)
                    taus_err_collect = np.zeros((2,num_zeta))
                    ind = 0
                    for i in chi_sqs_array:

                        #least squares
                        low_chi = find_nearest(i, 0.0)[0]
                        chi_sqs_collect[ind] = low_chi

                        if i[0][0] < low_chi+1 and i[-1][-1] < low_chi+1:
                            print('Warning: NOT CONVERGING ENOUGH') #stops code if not
                            #enough parameters to reach reduced low_chi + 1 before end
                            #of parameter space

                        lsqs_pbf_index = find_nearest(i, 0.0)[1][0][0]

                        #lsqs intrinsic width
                        lsqs_intrins_index = find_nearest(i, 0.0)[1][1][0]
                        lsqs_intrins_val = self.intrinsic_fwhms[lsqs_intrins_index]
                        intrinsic_width_collect[ind] = lsqs_intrins_val

                        taus_collect[ind] = self.tau_values[pbf_type][ind][lsqs_pbf_index]

                        ind+=1

                    low_chi = np.min(chi_sqs_collect)
                    chi_zeta_ind = np.where(chi_sqs_collect == low_chi)[0][0]

                    zeta_fin = self.zetas[chi_zeta_ind]
                    intrins_width_fin = self.intrinsic_fwhms[chi_zeta_ind]
                    tau_fin = taus_collect[chi_zeta_ind]

                    intrins_width_ind = np.where((self.intrinsic_fwhms == intrinsic_width_collect[chi_zeta_ind]))[0][0]

                    #plotting the fit parameters over zeta
                    for i in range(3):

                        plt.figure(1)
                        plt.xlabel('Zeta')

                        if i == 0:
                            plt.ylabel('Chi-Squared')
                            plt.plot(self.zetas, chi_sqs_collect)
                            param = 'chisqs'

                        if i == 2:
                            plt.ylabel('Overall Best Intrinsic Width FWHM (milliseconds)')
                            plt.plot(self.zetas, intrinsic_width_collect) #already converted to micro fwhm
                            param = 'iwidth'

                        if i == 3:
                            plt.ylabel('Overall Best Tau (microseconds)')
                            plt.plot(self.zetas, taus_collect)
                            param = 'tau'

                        title = f'ALLZETA|{self.intrinsic_shape.upper()}|PBF_fit_overall_{param}|MJD={self.mjd_round}|FREQ={self.freq_round}|bestZETA={self.zetas[chi_zeta_ind]}.pdf'
                        plt.savefig(title, bbox_inches='tight')
                        plt.close('all')

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width

                    if chi_sqs_collect[0] > low_chi+(1/(self.bin_num_care-num_par)) and chi_sqs_collect[-1] > low_chi+(1/(self.bin_num_care-num_par)):
                        below = find_nearest(chi_sqs_collect[:chi_zeta_ind], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                        above = find_nearest(chi_sqs_collect[chi_zeta_ind+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + chi_zeta_ind + 1
                        zeta_low = zeta_fin - self.zetas[below]
                        zeta_up = self.zetas[above] - zeta_fin
                    else:
                        zeta_low = 0
                        zeta_up = 0

                    #overall best fit plot
                    self.fit_plot(pbf_type, int(chi_zeta_ind), pbf_width_ind, low_chi = low_chi, iwidth_ind = intrins_width_ind)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns['intrins_width'] = intrins_width_fin
                    data_returns[f'{pbf_type}'] = zeta_fin
                    data_returns['zeta_low'] = zeta_low
                    data_returns['zeta_up'] = zeta_up

                    return(data_returns)

                #case where zeta and pbfwidth are free, iwidth is set
                if iwidth_ind != -1 and pbfwidth_ind == -1 and bzeta_ind == -1:

                    iwidth = self.intrinsic_fwhms[iwidth_ind]

                    print(f'Set intrinsic width to {iwidth} microseconds. Fitting for zeta and tau.')

                    chi_sqs_array = np.zeros((num_zeta, num_taus))
                    for i in itertools.product(zeta_inds, tau_inds):

                        template = self.convolved_profiles[pbf_type][i[0]][i[1]][iwidth_ind]
                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i[0]][i[1]] = chi_sq

                    chi_sqs_collect = np.zeros(num_zeta)
                    taus_collect = np.zeros(num_zeta)
                    taus_err_collect = np.zeros((2,num_zeta))
                    ind = 0
                    for i in chi_sqs_array:

                        zeta = self.zetas[ind]

                        #least squares
                        low_chi = find_nearest(i, 0.0)[0]
                        chi_sqs_collect[ind] = low_chi

                        if i[0] < low_chi+(1/(self.bin_num_care-num_par)) or i[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                            print('WARNING: TAU NOT CONVERGING')

                        lsqs_pbf_index = find_nearest(i, 0.0)[1][0][0]

                        taus_collect[ind] = self.tau_values[pbf_type][ind][lsqs_pbf_index]

                        ind+=1

                    low_chi = np.min(chi_sqs_collect)
                    chi_zeta_ind = np.where(chi_sqs_collect == low_chi)[0][0]

                    zeta_fin = self.zetas[chi_zeta_ind]
                    tau_fin = taus_collect[chi_zeta_ind]

                    #plotting the fit parameters over zeta
                    for i in range(2):

                        plt.figure(1)
                        plt.xlabel('Zeta')

                        if i == 0:
                            plt.ylabel('Chi-Squared')
                            plt.plot(self.zetas, chi_sqs_collect)
                            param = 'chisqs'

                        if i == 1:
                            plt.ylabel('Overall Best Tau (microseconds)')
                            plt.plot(self.zetas, taus_collect)
                            param = 'tau'

                        title = f'ALLZETA|{self.intrinsic_shape.upper()}|PBF_fit_overall_{param}|MJD={self.mjd_round}|FREQ={self.freq_round}|set_ifwhm_mus={int(np.round(iwidth))}|bestZETA={self.zetas[chi_zeta_ind]}.pdf'
                        plt.savefig(title, bbox_inches='tight')
                        plt.close('all')

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width

                    if chi_sqs_collect[0] > low_chi+(1/(self.bin_num_care-num_par)) and chi_sqs_collect[-1] > low_chi+(1/(self.bin_num_care-num_par)):
                        below = find_nearest(chi_sqs_collect[:chi_zeta_ind], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                        above = find_nearest(chi_sqs_collect[chi_zeta_ind+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + chi_zeta_ind + 1
                        zeta_low = zeta_fin - self.zetas[below]
                        zeta_up = self.zetas[above] - zeta_fin
                    else:
                        zeta_low = 0
                        zeta_up = 0

                    #overall best fit plot
                    self.fit_plot(pbf_type, int(chi_zeta_ind), pbf_width_ind, low_chi = low_chi, iwidth_ind = iwidth_ind)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns['intrins_width_set'] = iwidth
                    data_returns[f'{pbf_type}'] = zeta_fin
                    data_returns['zeta_low'] = zeta_low
                    data_returns['zeta_up'] = zeta_up

                    return(data_returns)

                #case where zeta is set, but iwidth and pbfwidth free
                elif iwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                    print(f'Set zeta = {self.zetas[bzeta_ind]} and fitting for intrinsic width and tau.')

                    zeta = self.zetas[bzeta_ind]

                    chi_sqs_array = np.zeros((num_taus, num_iwidth))
                    for i in itertools.product(tau_inds, iwidth_inds):

                        template = self.convolved_profiles[pbf_type][bzeta_ind][i[0]][i[1]]
                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i[0]][i[1]] = chi_sq

                    #least squares
                    low_chi = find_nearest(chi_sqs_array, 0.0)[0]

                    if chi_sqs_array[0][0] < low_chi+(1/(self.bin_num_care-num_par)) or chi_sqs_array[-1][-1] < low_chi+(1/(self.bin_num_care-num_par)):
                        raise Exception('NOT CONVERGING ENOUGH') #stops code if not
                        #enough parameters to reach reduced low_chi + 1 before end
                        #of parameter space

                    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]

                    #lsqs intrinsic width
                    lsqs_intrins_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
                    lsqs_intrins_val = self.intrinsic_fwhms[lsqs_intrins_index]

                    tau_fin = self.tau_values[pbf_type][bzeta_ind][lsqs_pbf_index]

                    self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, low_chi = low_chi, iwidth_ind = lsqs_intrins_index)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns['intrins_width'] = lsqs_intrins_val
                    data_returns[f'{pbf_type}'] = zeta

                    return(data_returns)

                #case where zeta and intrinsix width are set, but fitting for pbf width
                elif iwidth_ind != -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                    print(f'Set zeta = {zetaselect[bzeta_ind]} and intrinsic width = {self.intrinsic_fwhms[iwidth_ind]} microseconds and fitting for tau.')

                    zeta = self.zetas[bzeta_ind]
                    iwidth = self.intrinsic_fwhms[iwidth_ind]

                    chi_sqs_array = np.zeros(num_taus)
                    for i in tau_inds:

                        template = self.convolved_profiles[pbf_type][bzeta_ind][i][iwidth_ind]
                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i] = chi_sq

                    #self.chi_plot(chi_sqs_array, pbf_type, bzeta = zeta, iwidth = iwidth)

                    if chi_sqs_array[0] < low_chi+(1/(self.bin_num_care-num_par)) or chi_sqs_array[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                        raise Exception('NOT CONVERGING ENOUGH')

                    low_chi = find_nearest(chi_sqs_array, 0.0)[0]
                    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]

                    tau_fin = self.tau_values[pbf_type][bzeta_ind][lsqs_pbf_index]

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width
                    below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                    above = find_nearest(chi_sqs_array[lsqs_pbf_index+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + lsqs_pbf_index + 1

                    tau_arr = self.tau_values[pbf_type][bzeta_ind]
                    tau_low = tau_fin - tau_arr[below]
                    tau_up = tau_arr[above] - tau_fin

                    self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, low_chi = low_chi, iwidth_ind = iwidth_ind, low_pbf = below, high_pbf = above)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns['intrins_width'] = iwidth
                    data_returns[f'{pbf_type}'] = zeta
                    data_returns['tau_low'] = tau_low
                    data_returns['tau_up'] = tau_up

                    return(data_returns)


            elif pbf_type == 'exp':

                #case where iwidth and pbfwidth are free
                if iwidth_ind == -1 and pbfwidth_ind == -1:

                    print(f'Fitting for intrinsic width and tau.')

                    chi_sqs_array = np.zeros((num_taus, num_iwidth))
                    for i in itertools.product(tau_inds, iwidth_inds):

                        template = self.convolved_profiles[pbf_type][i[0]][i[1]]
                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i[0]][i[1]] = chi_sq

                    #least squares
                    low_chi = find_nearest(chi_sqs_array, 0.0)[0]

                    if chi_sqs_array[0][0] < low_chi+(1/(self.bin_num_care-num_par)) or chi_sqs_array[-1][-1] < low_chi+(1/(self.bin_num_care-num_par)):
                        raise Exception('NOT CONVERGING ENOUGH') #stops code if not
                        #enough parameters to reach reduced low_chi + 1 before end
                        #of parameter space

                    #lsqs pbf width
                    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]

                    #lsqs intrinsic width
                    lsqs_intrins_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
                    lsqs_intrins_val = self.intrinsic_fwhms[lsqs_intrins_index]

                    tau_fin = self.tau_values[pbf_type][lsqs_pbf_index]

                    self.fit_plot(pbf_type, 0, lsqs_pbf_index, low_chi = low_chi, iwidth_ind = lsqs_intrins_index)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns['intrins_width'] = lsqs_intrins_val
                    data_returns[f'{pbf_type}'] = 'exp'

                    return(data_returns)

                #case where intrinsic width is set but fitting for pbf width
                elif iwidth_ind != -1 and pbfwidth_ind == -1:

                    iwidth = self.intrinsic_fwhms[iwidth_ind]

                    print(f'Set intrinsic width = {iwidth} microseconds and fitting for tau.')

                    chi_sqs_array = np.zeros(num_taus)
                    for i in tau_inds:

                        template = self.convolved_profiles[pbf_type][i][iwidth_ind]
                        chi_sq = self.fit_sing(template, num_par)
                        chi_sqs_array[i] = chi_sq

                    low_chi = find_nearest(chi_sqs_array, 0.0)[0]

                    if chi_sqs_array[0] < low_chi+(1/(self.bin_num_care-num_par)) or chi_sqs_array[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                        raise Exception('NOT CONVERGING ENOUGH')

                    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]

                    tau_fin = self.tau_values[pbf_type][lsqs_pbf_index]

                    #ERROR TEST - one reduced chi-squared unit above and below and these
                    #chi-squared bins are for varying pbf width
                    below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                    above = find_nearest(chi_sqs_array[lsqs_pbf_index+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + lsqs_pbf_index + 1

                    tau_low = tau_fin - self.tau_values[pbf_type][below]
                    tau_up = self.tau_values[pbf_type][above] - tau_fin

                    self.fit_plot(pbf_type, 0, lsqs_pbf_index, low_chi = low_chi, iwidth_ind = iwidth_ind, low_pbf = below, high_pbf = above)

                    data_returns = {}
                    data_returns['low_chi'] = low_chi
                    data_returns['tau_fin'] = tau_fin
                    data_returns['fse_effect'] = self.comp_fse(tau_fin)
                    data_returns['intrins_width'] = iwidth
                    data_returns[f'{pbf_type}'] = 'exp'
                    data_returns['tau_low'] = tau_low
                    data_returns['tau_up'] = tau_up

                    return(data_returns)
