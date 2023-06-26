"""
Created June 2023
@author: Abra Geiger abrageiger7

Class for profile fitting with gaussian intrinsic shape.
"""

import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from pypulse.singlepulse import SinglePulse

from fit_functions import *

#fitting templates
with open('gauss_convolved_profiles.pkl', 'rb') as fp:
    gauss_convolved_profiles = pickle.load(fp)

#tau values corresponding to above templates
with open('tau_values.pkl', 'rb') as fp:
    tau_values = pickle.load(fp)

beta_tau_values = tau_values['beta']
exp_tau_values = tau_values['exp']
zeta_tau_values = tau_values['zeta']


class Profile_Gauss:

    def __init__(self, mjd, data, frequencies, dur):
        '''
        mjd (float) - epoch of observation
        data (2D array) - pulse data for epoch
        frequencies (1D array) - frequencies corresponding to the data channels
        dur (float) - observation duration in seconds
        '''

        #initialize the object attributes

        self.mjd = mjd
        self.mjd_round = int(np.around(mjd))
        self.data_orig = data
        self.freq_orig = frequencies
        self.dur = dur


        #subaverages the data for every four frequency channels

        s = subaverages4(mjd, data, frequencies)
        self.num_sub = len(s[1])
        self.subaveraged_info = s


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

        sp = SinglePulse(self.data_suba, opw = np.arange(0, self.start_index))
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

    def chi_plot(self, chi_sq_arr, pbf_type, bzeta = -1, gwidth = -1, pbfwidth = -1):

        '''Plots the inputted chi_sq_arr against the given parameters.

        If gwidth is -1, indicates that the chi-squared surface exists over
        all possible gaussian widths. Same for pbfwidth.

        pbf_type is one of these strings: 'beta', 'zeta', 'exp'

        gwidth and pbfwidth are actual values of intrinsic width and pbf stretch
        factor. bzeta is the actual value of beta or zeta if not exp pbf.'''

        #check of logical function inputs

        if pbf_type != 'beta' or pbf_type != 'zeta' or pbf_type != 'exp':
            raise Exception('Invalid pbf_type.')

        if pbf_type == 'beta' or pbf_type == 'zeta':
            if bzeta == -1:
                raise Exception('Please indicate the b/zeta value used.')

        plt.figure(45)

        if gwidth == -1 and pbfwidth == -1: #neither width set, so 2D chi^2 surface

            plt.title("Fit Chi-sqs")
            plt.xlabel("Gaussian FWHM (microseconds)")
            plt.ylabel("PBF Width")

            #adjust the imshow tick marks
            gauss_ticks = np.zeros(10)
            for ii in range(10):
                gauss_ticks[ii] = str(gauss_fwhm[ii*20])[:3]
            pbf_ticks = np.zeros(10)
            for ii in range(10):
                pbf_ticks[ii] = str(widths[ii*40])[:3]
            plt.xticks(ticks = np.linspace(0,num_gwidth,num=10), labels = gauss_ticks)
            plt.yticks(ticks = np.linspace(0,num_pbfwidth,num=10), labels = pbf_ticks)

            plt.imshow(chi_sq_arr, cmap=plt.cm.viridis_r, origin = 'lower', aspect = 0.25)
            plt.colorbar()

            if pbf_type == 'beta':
                    title = f"BETA={bzeta}|GAUSS|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}.pdf"

            elif pbf_type == 'exp':
                    title = f"EXP|GAUSS|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}.pdf"

            elif pbf_type == 'zeta':
                    title = f"ZETA={bzeta}|GAUSS|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}.pdf"

            plt.savefig(title)
            print(title)
            plt.close(45)

        elif gwidth != -1: #gwidth set, so 1D chi^2 surface

            gwidth_round = int(np.around(gwidth))

            plt.title('Fit Chi-sqs')
            plt.xlabel('PBF Width')
            plt.ylabel('Reduced Chi-Sq')
            plt.plot(widths, chi_sq_arr, drawstyle='steps-pre')

            if pbf_type == 'beta':
                    title = f"BETA={bzeta}|GAUSS|PBF_fit_chisq_setg|MJD={self.mjd_round}|FREQ={self.freq_round}|GWIDTH={gwidth_round}.pdf"

            elif pbf_type == 'exp':
                    title = f"EXP|GAUSS|PBF_fit_chisq_setg|MJD={self.mjd_round}|FREQ={self.freq_round}|GWIDTH={gwidth_round}.pdf"

            elif pbf_type == 'zeta':
                    title = f"ZETA={bzeta}|GAUSS|PBF_fit_chisq_setg|MJD={self.mjd_round}|FREQ={self.freq_round}|GWIDTH={gwidth_round}.pdf"

            plt.savefig(title)
            print(title)
            plt.close(45)


    def fit_plot(self, pbf_type, bzeta_ind, pbfwidth_ind, gwidth_ind, low_chi, low_pbf = -1, high_pbf = -1):

        '''Plots and saves the fit of the profile subaveraged data to the
        template indicated by the argument indexes and the bolean
        indicating if decaying exponential wanted for the broadening function.

        bzeta_ind, pbfwidth_ind, gwidth_ind are ints; exp is boolean

        If low_pbf and or high_pbf != -1, plots additional fits in different color.
        These should be used for demonstrating error on tau and how this makes the
        fit different.'''

        #test that arguement combination is logical

        if pbf_type != 'beta' or pbf_type != 'zeta' or pbf_type != 'exp':
            raise Exception('Invalid pbf_type.')

        if pbf_type == 'beta' or pbf_type == 'zeta':
            if bzeta_ind == -1:
                raise Exception('Please indicate the b/zeta index.')
            elif type(bzeta_ind) != int:
                raise Exception('bzeta_ind must be an integer.')

        #depending on pbf type, get profiles and taus

        if pbf_type == 'beta':
            i = gauss_convolved_profiles[pbf_type][bzeta_ind][pbfwidth_ind][gwidth_ind]
            tau_val = beta_tau_values[bzeta_ind][pbfwidth_ind]

            if low_pbf != -1:
                low_pbf_i = gauss_convolved_profiles[pbf_type][bzeta_ind][low_pbf][gwidth_ind]
                tau_val_low = beta_tau_values[bzeta_ind][low_pbf]
            if high_pbf != -1:
                high_pbf_i = gauss_convolved_profiles[pbf_type][bzeta_ind][high_pbf][gwidth_ind]
                tau_val_high = beta_tau_values[bzeta_ind][high_pbf]

        elif pbf_type == 'exp':
            i = exp_convolved_profiles[pbfwidth_ind][gwidth_ind]
            tau_val = exp_tau_values[pbfwidth_ind]

            if low_pbf != -1:
                low_pbf_i = exp_convolved_profiles[low_pbf][gwidth_ind]
                tau_val_low = exp_tau_values[low_pbf]
            if high_pbf != -1:
                high_pbf_i = exp_convolved_profiles[high_pbf][gwidth_ind]
                tau_val_high = exp_tau_values[high_pbf]

        elif pbf_type == 'zeta':
            i = gauss_convolved_profiles[pbf_type][bzeta_ind][pbfwidth_ind][gwidth_ind]
            tau_val = zeta_tau_values[bzeta_ind][pbfwidth_ind]

            if low_pbf != -1:
                low_pbf_i =gauss_convolved_profiles[pbf_type][bzeta_ind][low_pbf][gwidth_ind]
                tau_val_low = zeta_tau_values[bzeta_ind][low_pbf]
            if high_pbf != -1:
                high_pbf_i = gauss_convolved_profiles[pbf_type][bzeta_ind][high_pbf][gwidth_ind]
                tau_val_high = zeta_tau_values[bzeta_ind][high_pbf]

        profile = i / np.max(i) #fitPulse requires template height of one
        z = np.max(profile)
        zind = np.where(profile == z)[0][0]
        ind_diff = self.xind-zind
        #this lines the profiles up approximately so that Single Pulse finds the
        #true minimum, not just a local min
        profile = np.roll(profile, ind_diff)
        sp = SinglePulse(self.data_suba, opw = np.arange(0, self.start_index))
        fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template
        #matching, scale factor, TOA error, scale factor error, signal to noise
        #ratio, cross-correlation coefficient
        #based on the fitPulse fitting, scale and shift the profile to best fit
        #the inputted data
        #fitPulse figures out the best amplitude itself
        spt = SinglePulse(profile*fitting[2])
        fitted_template = spt.shiftit(fitting[1])

        fitted_template = fitted_template*self.mask

        plt.figure(50)
        fig1 = plt.figure(50)
        #Plot Data-model
        frame1=fig1.add_axes((.1,.3,.8,.6))
        #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
        plt.title('Best Fit Template over Data')

        #plot the lower error profile alongside the best fit for comparison
        if low_pbf != -1:

            profilel = low_pbf_i / np.max(low_pbf_i) #fitPulse requires template height of one
            z = np.max(profilel)
            zind = np.where(profilel == z)[0][0]
            ind_diff = self.xind-zind
            #this lines the profiles up approximately so that Single Pulse finds the
            #true minimum, not just a local min
            profilel = np.roll(profilel, ind_diff)
            sp = SinglePulse(self.data_suba, opw = np.arange(0, self.start_index))
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
            sp = SinglePulse(self.data_suba, opw = np.arange(0, self.start_index))
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
        plt.plot(time, self.data_suba*self.mask, '.', ms = '2.4', label = 'Data')
        if high_pbf != -1:
            plt.plot(time, fitted_templateh, alpha = 0.5, label = fr'Upper Error; $\tau$ = {int(np.around(tau_val_high))} $\mu$s', color = 'orange')
        if low_pbf != -1:
            plt.plot(time, fitted_templatel, alpha = 0.5, label = fr'Lower Error; $\tau$ = {int(np.around(tau_val_low))} $\mu$s', color = 'orange')
        plt.plot(time, fitted_template, label = fr'Best fit; $\tau$ = {int(np.around(tau_val,0))} $\mu$s', color = 'red')
        plt.plot([], [], ' ', label=fr"min $\chi^2$ = {np.around(low_chi,2)}")
        plt.legend(prop={'size': 7})
        frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
        plt.plot()

        #Residual plot
        difference = np.subtract(self.data_suba*self.mask, fitted_template)
        frame2=fig1.add_axes((.1,.1,.8,.2))
        plt.plot(time, difference, '.', ms = '2.4')
        plt.xlabel('Pulse Period (milliseconds)')
        plt.ylabel('Residuals')
        plt.plot()

        gwidth_round = int(np.around(gauss_fwhm[gwidth_ind]))
        pbfwidth_round = int(np.around(widths[pbfwidth_ind]))


        if pbf_type == 'beta':
            title = f'BETA={betaselect[bzeta_ind]}|GAUSS|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}||PBFW={pbfwidth_round}|GW={gwidth_round}.pdf'
        elif pbf_type == 'exp':
            title = f'EXP|GAUSS|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}|PBFW={pbfwidth_round}|GW={gwidth_round}.pdf'
        elif pbf_type == 'zeta':
            title = f'ZETA={zetaselect[bzeta_ind]}|GAUSS|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}|PBFW={pbfwidth_round}|GW={gwidth_round}.pdf'

        plt.savefig(title)
        print(title)
        plt.close(50)

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

    def init_freq_subint(self, freq_subint_index):
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

        if self.freq_suba >= 1600:
            self.start_index = int((700/2048)*phase_bins)
            self.stop_index = int((1548/2048)*phase_bins)
        elif self.freq_suba >= 1400 and self.freq_suba < 1600:
            self.start_index = int((700/2048)*phase_bins)
            self.stop_index = int((1648/2048)*phase_bins)
        elif self.freq_suba >= 1200 and self.freq_suba < 1400:
            self.start_index = int((650/2048)*phase_bins)
            self.stop_index = int((1798/2048)*phase_bins)
        elif self.freq_suba >= 1000 and self.freq_suba < 1200:
            self.start_index = int((600/2048)*phase_bins)
            self.stop_index = int((1948/2048)*phase_bins)
        mask[self.start_index:self.stop_index] = 1.0

        self.bin_num_care = self.stop_index-self.start_index

        self.mask = mask


        #Calculates the root mean square noise of the off pulse.
        #Used later to calculate normalized chi-squared.

        rms_collect = 0
        for i in range(opr_size):
            rms_collect += self.data_suba[i]**2
        rms = math.sqrt(rms_collect/opr_size)

        self.rms_noise = rms


    def fit(self, freq_subint_index, pbf_type, bzeta_ind = -1, gwidth_ind = -1, pbfwidth_ind = -1):
        '''Calculates the best broadening function and corresponding parameters
        for the Profile object.

        pbf_type (str): either 'beta', 'exp', or 'zeta'
        bzeta_ind (int): if nonzero, set beta to this index of betaselect
        gwidth_ind (int): if nonzero, set gauss width to this index of gauss_fwhm
        pbfwidth_ind (int) : if nonzero, set pbf width to this index of widths

        No error calculations for varying more than one parameter
        '''

        if pbf_type != 'beta':
            if pbf_type != 'zeta':
                if pbf_type != 'exp':
                    raise Exception('Invalid pbf_type.')

        if pbf_type == 'beta' or pbf_type == 'zeta':
            if type(bzeta_ind) != int:
                raise Exception('bzeta_ind must be an integer.')

        self.init_freq_subint(freq_subint_index)

        #number of each parameter in the parameter grid
        num_beta = np.size(betaselect)

        #number of each parameter in the parameter grid
        num_zeta = np.size(zetaselect)

        beta_inds = np.arange(num_beta)
        zeta_ind = np.arange(num_zeta)
        gwidth_inds = np.arange(num_gwidth)
        pbfwidth_inds = np.arange(num_pbfwidth)

        #set more convenient names for the data to be fitted to
        data_care = self.data_suba
        freq_care = self.freq_suba

        print('Intrinsic shape is gaussian.')

        if pbf_type == 'beta':

            #case where beta, gwidth, and pbfwidth are free
            if gwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind == -1:

                print('Fitting for beta, gaussian width, and PBF width.')

                num_par = 5 #number of fitted parameters

                chi_sqs_array = np.zeros((num_beta, num_pbfwidth, num_gwidth))
                for i in itertools.product(beta_inds, pbfwidth_inds, gwidth_inds):

                    template = gauss_convolved_profiles[pbf_type][i[0]][i[1]][i[2]]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i[0]][i[1]][i[2]] = chi_sq

                chi_sqs_collect = np.zeros(num_beta)
                pbf_width_collect = np.zeros(num_beta)
                gaussian_width_collect = np.zeros(num_beta)
                taus_collect = np.zeros(num_beta)
                taus_err_collect = np.zeros((2,num_beta))
                ind = 0
                for i in chi_sqs:

                    beta = betaselect[ind]

                    #scale the chi-squared array by the rms value of the profile

                    self.chi_plot(chi_sqs_array, pbf_type, bzeta = beta)

                    #least squares
                    low_chi = find_nearest(chi_sqs_array, 0.0)[0]
                    chi_sqs_collect[ind] = low_chi

                    if chi_sqs_array[0][0] < low_chi+1 and chi_sqs_array[-1][-1] < low_chi+1:
                        raise Exception('NOT CONVERGING ENOUGH') #stops code if not
                        #enough parameters to reach reduced low_chi + 1 before end
                        #of parameter space

                    #lsqs pbf width
                    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
                    lsqs_pbf_val = widths[lsqs_pbf_index]
                    pbf_width_collect[ind] = lsqs_pbf_val

                    #lsqs gaussian width
                    lsqs_gauss_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
                    lsqs_gauss_val = gauss_fwhm[lsqs_gauss_index]
                    gaussian_width_collect[ind] = lsqs_gauss_val

                    taus_collect[ind] = beta_tau_values[ind][lsqs_pbf_index]

                    self.fit_plot(pbf_type, ind, lsqs_pbf_index, lsqs_gauss_index, low_chi)

                    ind+=1

                low_chi = np.min(chi_sqs_collect)
                chi_beta_ind = np.where(chi_sqs_collect == low_chi)[0][0]

                beta_fin = betaselect[chi_beta_ind]
                pbf_width_fin = pbf_width_collect[chi_beta_ind]
                gauss_width_fin = gaussian_width_collect[chi_beta_ind]
                tau_fin = taus_collect[chi_beta_ind]

                pbf_width_ind = np.where(widths == pbf_width_fin)[0][0]
                gauss_width_ind = np.where((gauss_fwhm == gaussian_width_collect[chi_beta_ind]))[0][0]

                #plotting the fit parameters over beta
                for i in range(4):

                    plt.figure(i*4)
                    plt.xlabel('Beta')

                    if i == 0:
                        plt.ylabel('Chi-Squared')
                        plt.plot(betaselect, chi_sqs_collect)
                        param = 'chisqs'

                    if i == 1:
                        plt.ylabel('Overall Best PBF Width')
                        plt.plot(betaselect, pbf_width_collect)
                        param = 'pbfw'

                    if i == 2:
                        plt.ylabel('Overall Best Gaussian Width FWHM (milliseconds)')
                        plt.plot(betaselect, gaussian_width_collect) #already converted to micro fwhm
                        param = 'gwidth'

                    if i == 3:
                        plt.ylabel('Overall Best Tau (microseconds)')
                        plt.plot(betaselect, taus_collect)
                        param = 'tau'

                    title = f'ALLBETA|GAUSS|PBF_fit_overall_{param}|MJD={self.mjd_round}|FREQ={self.freq_round}|bestBETA={betaselect[chi_beta_ind]}.pdf'
                    plt.savefig(title)
                    plt.close(i*4)

                #overall best fit plot
                self.fit_plot(pbf_type, chi_beta_ind, pbf_width_ind, gauss_width_ind, low_chi)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['gauss_width'] = gauss_width_fin
                data_returns['pbf_width'] = pbf_width_fin
                data_returns[f'{pbf_type}'] = beta_fin

                return(data_returns)

            #case where beta is set, but gwidth and pbfwidth free
            elif gwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                print(f'Set beta = {betaselect[bzeta_ind]}. Fitting for gaussian width and PBF width.')

                num_par = 4 #number of fitted parameters

                beta = betaselect[bzeta_ind]

                chi_sqs_array = np.zeros((num_pbfwidth, num_gwidth))
                for i in itertools.product(pbfwidth_inds, gwidth_inds):

                    template = gauss_convolved_profiles[pbf_type][bzeta_ind][i[0]][i[1]]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i[0]][i[1]] = chi_sq

                self.chi_plot(chi_sqs_array, pbf_type, bzeta = beta)

                #least squares
                low_chi = find_nearest(chi_sqs_array, 0.0)[0]

                if chi_sqs_array[0][0] < low_chi+1 or chi_sqs_array[-1][-1] < low_chi+1:
                    raise Exception('NOT CONVERGING ENOUGH') #stops code if not
                    #enough parameters to reach reduced low_chi + 1 before end
                    #of parameter space

                #lsqs pbf width
                lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
                lsqs_pbf_val = widths[lsqs_pbf_index]

                #lsqs gaussian width
                lsqs_gauss_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
                lsqs_gauss_val = gauss_fwhm[lsqs_gauss_index]

                tau_fin = beta_tau_values[bzeta_ind][lsqs_pbf_index]

                self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, lsqs_gauss_index, low_chi)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['gauss_width'] = lsqs_gauss_val
                data_returns['pbf_width'] = lsqs_pbf_val
                data_returns[f'{pbf_type}'] = beta

                return(data_returns)

            #case where beta and gaussian width are set, but fitting for pbf width
            elif gwidth_ind != -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                print(f'Set beta = {betaselect[bzeta_ind]} and gaussian width = {gauss_fwhm[gwidth_ind]} microseconds. Fitting for PBF Width.')

                num_par = 3 # number of fitted parameters

                beta = betaselect[bzeta_ind]
                gwidth = gauss_fwhm[gwidth_ind]

                chi_sqs_array = np.zeros(num_pbfwidth)
                for i in pbfwidth_inds:

                    template = gauss_convolved_profiles[pbf_type][bzeta_ind][i][gwidth_ind]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i] = chi_sq

                self.chi_plot(chi_sqs_array, pbf_type, bzeta = beta, gwidth = gwidth)

                low_chi = find_nearest(chi_sqs_array, 0.0)[0]
                lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
                pbf_width_fin = widths[lsqs_pbf_index]

                if chi_sqs_array[0] < low_chi+(1/(self.bin_num_care-num_par)) or chi_sqs_array[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                    raise Exception('NOT CONVERGING ENOUGH')

                tau_fin = beta_tau_values[bzeta_ind][lsqs_pbf_index]

                #ERROR TEST - one reduced chi-squared unit above and below and these
                #chi-squared bins are for varying pbf width
                below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                above = find_nearest(chi_sqs_array[lsqs_pbf_index+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + lsqs_pbf_index + 1

                tau_arr = beta_tau_values[bzeta_ind]
                tau_low = tau_fin - tau_arr[below]
                tau_up = tau_arr[above] - tau_fin

                self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, gwidth_ind, low_chi, low_pbf = below, high_pbf = above)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['gauss_width'] = gwidth
                data_returns['pbf_width'] = pbf_width_fin
                data_returns[f'{pbf_type}'] = beta
                data_returns['tau_low'] = tau_low
                data_returns['tau_up'] = tau_up

                return(data_returns)


        elif pbf_type == 'zeta':

            #case where zeta, gwidth, and pbfwidth are free
            if gwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind == -1:

                print('Fitting for zeta, gaussian width, and PBF width.')

                num_par = 5 #number of fitted parameters

                chi_sqs_array = np.zeros((num_zeta, num_pbfwidth, num_gwidth))
                for i in itertools.product(zeta_inds, pbfwidth_inds, gwidth_inds):

                    template = gauss_convolved_profiles[pbf_type][i[0]][i[1]][i[2]]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i[0]][i[1]][i[2]] = chi_sq

                chi_sqs_collect = np.zeros(num_zeta)
                pbf_width_collect = np.zeros(num_zeta)
                gaussian_width_collect = np.zeros(num_zeta)
                taus_collect = np.zeros(num_zeta)
                taus_err_collect = np.zeros((2,num_zeta))
                ind = 0
                for i in chi_sqs:

                    zeta = zetaselect[ind]

                    #scale the chi-squared array by the rms value of the profile

                    self.chi_plot(chi_sqs_array, pbf_type, bzeta = zeta)

                    #least squares
                    low_chi = find_nearest(chi_sqs_array, 0.0)[0]
                    chi_sqs_collect[ind] = low_chi

                    if chi_sqs_array[0][0] < low_chi+1 and chi_sqs_array[-1][-1] < low_chi+1:
                        raise Exception('NOT CONVERGING ENOUGH') #stops code if not
                        #enough parameters to reach reduced low_chi + 1 before end
                        #of parameter space

                    #lsqs pbf width
                    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
                    lsqs_pbf_val = widths[lsqs_pbf_index]
                    pbf_width_collect[ind] = lsqs_pbf_val

                    #lsqs gaussian width
                    lsqs_gauss_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
                    lsqs_gauss_val = gauss_fwhm[lsqs_gauss_index]
                    gaussian_width_collect[ind] = lsqs_gauss_val

                    taus_collect[ind] = zeta_tau_values[ind][lsqs_pbf_index]

                    self.fit_plot(pbf_type, ind, lsqs_pbf_index, lsqs_gauss_index, low_chi)

                    ind+=1

                low_chi = np.min(chi_sqs_collect)
                chi_zeta_ind = np.where(chi_sqs_collect == low_chi)[0][0]

                zeta_fin = zetaselect[chi_zeta_ind]
                pbf_width_fin = pbf_width_collect[chi_zeta_ind]
                gauss_width_fin = gaussian_width_collect[chi_zeta_ind]
                tau_fin = taus_collect[chi_zeta_ind]

                pbf_width_ind = np.where(widths == pbf_width_fin)[0][0]
                gauss_width_ind = np.where((gauss_fwhm == gaussian_width_collect[chi_beta_ind]))[0][0]

                #plotting the fit parameters over beta
                for i in range(4):

                    plt.figure(i*4)
                    plt.xlabel('Zeta')

                    if i == 0:
                        plt.ylabel('Chi-Squared')
                        plt.plot(zetaselect, chi_sqs_collect)
                        param = 'chisqs'

                    if i == 1:
                        plt.ylabel('Overall Best PBF Width')
                        plt.plot(zetaselect, pbf_width_collect)
                        param = 'pbfw'

                    if i == 2:
                        plt.ylabel('Overall Best Gaussian Width FWHM (milliseconds)')
                        plt.plot(zetaselect, gaussian_width_collect) #already converted to micro fwhm
                        param = 'gwidth'

                    if i == 3:
                        plt.ylabel('Overall Best Tau (microseconds)')
                        plt.plot(zetaselect, taus_collect)
                        param = 'tau'

                    title = f'ALLZETA|PBF_fit_overall_{param}|MJD={self.mjd_round}|FREQ={self.freq_round}|bestZETA={zetaselect[chi_zeta_ind]}.pdf'
                    plt.savefig(title)
                    plt.close(i*4)

                self.fit_plot(pbf_type, chi_zeta_ind, pbf_width_ind, gauss_width_ind, low_chi)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['gauss_width'] = gauss_width_fin
                data_returns['pbf_width'] = pbf_width_fin
                data_returns[f'{pbf_type}'] = zeta_fin

                return(data_returns)

            #case where beta is set, but gwidth and pbfwidth free
            elif gwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                print(f'Set zeta = {zetaselect[bzeta_ind]}. Fitting for gaussian width and PBF width.')

                num_par = 4 #number of fitted parameters

                zeta = zetaselect[bzeta_ind]

                chi_sqs_array = np.zeros((num_pbfwidth, num_gwidth))
                for i in itertools.product(pbfwidth_inds, gwidth_inds):

                    template = gauss_convolved_profiles[pbf_type][bzeta_ind][i[0]][i[1]]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i[0]][i[1]] = chi_sq

                self.chi_plot(chi_sqs_array, pbf_type, bzeta = zeta)

                #least squares
                low_chi = find_nearest(chi_sqs_array, 0.0)[0]

                if chi_sqs_array[0][0] < low_chi+1 or chi_sqs_array[-1][-1] < low_chi+1:
                    raise Exception('NOT CONVERGING ENOUGH') #stops code if not
                    #enough parameters to reach reduced low_chi + 1 before end
                    #of parameter space

                #lsqs pbf width
                lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
                lsqs_pbf_val = widths[lsqs_pbf_index]

                #lsqs gaussian width
                lsqs_gauss_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
                lsqs_gauss_val = gauss_fwhm[lsqs_gauss_index]

                tau_fin = zeta_tau_values[bzeta_ind][lsqs_pbf_index]

                self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, lsqs_gauss_index, low_chi)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['gauss_width'] = lsqs_gauss_val
                data_returns['pbf_width'] = lsqs_pbf_val
                data_returns[f'{pbf_type}'] = zeta

                return(data_returns)

            #case where beta and gaussian width are set, but fitting for pbf width
            elif gwidth_ind != -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                print(f'Set zeta = {zetaselect[bzeta_ind]} and gaussian width = {gauss_fwhm[gwidth_ind]} microseconds. Fitting for PBF Width.')

                num_par = 3 # number of fitted parameters

                zeta = zetaselect[bzeta_ind]
                gwidth = gauss_fwhm[gwidth_ind]

                chi_sqs_array = np.zeros(num_pbfwidth)
                for i in pbfwidth_inds:

                    template = gauss_convolved_profiles[pbf_type][bzeta_ind][i][gwidth_ind]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i] = chi_sq

                self.chi_plot(chi_sqs_array, pbf_type, bzeta = zeta, gwidth = gwidth)

                low_chi = find_nearest(chi_sqs_array, 0.0)[0]
                lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
                pbf_width_fin = widths[lsqs_pbf_index]

                if chi_sqs_array[0] < low_chi+(1/(self.bin_num_care-num_par)) or chi_sqs_array[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                    raise Exception('NOT CONVERGING ENOUGH')

                tau_fin = zeta_tau_values[bzeta_ind][lsqs_pbf_index]

                #ERROR TEST - one reduced chi-squared unit above and below and these
                #chi-squared bins are for varying pbf width
                below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                above = find_nearest(chi_sqs_array[lsqs_pbf_index+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + lsqs_pbf_index + 1

                tau_arr = zeta_tau_values[bzeta_ind]
                tau_low = tau_fin - tau_arr[below]
                tau_up = tau_arr[above] - tau_fin

                self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, gwidth_ind, low_chi, low_pbf = below, high_pbf = above)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['gauss_width'] = gwidth
                data_returns['pbf_width'] = pbf_width_fin
                data_returns[f'{pbf_type}'] = zeta
                data_returns['tau_low'] = tau_low
                data_returns['tau_up'] = tau_up

                return(data_returns)


        elif pbf_type == 'exp':

            #case where gwidth and pbfwidth are free
            if gwidth_ind == -1 and pbfwidth_ind == -1:

                print(f'Decaying exponential pbf. Fitting for gaussian width and PBF width.')

                num_par = 4 #number of fitted parameters

                chi_sqs_array = np.zeros((num_pbfwidth, num_gwidth))
                for i in itertools.product(pbfwidth_inds, gwidth_inds):

                    template = gauss_convolved_profiles[pbf_type][i[0]][i[1]]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i[0]][i[1]] = chi_sq

                self.chi_plot(chi_sqs_array, pbf_type)

                #least squares
                low_chi = find_nearest(chi_sqs_array, 0.0)[0]

                if chi_sqs_array[0][0] < low_chi+1 or chi_sqs_array[-1][-1] < low_chi+1:
                    raise Exception('NOT CONVERGING ENOUGH') #stops code if not
                    #enough parameters to reach reduced low_chi + 1 before end
                    #of parameter space

                #lsqs pbf width
                lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
                lsqs_pbf_val = widths[lsqs_pbf_index]

                #lsqs gaussian width
                lsqs_gauss_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
                lsqs_gauss_val = gauss_fwhm[lsqs_gauss_index]

                tau_fin = exp_tau_values[lsqs_pbf_index]

                self.fit_plot(pbf_type, 0, lsqs_pbf_index, lsqs_gauss_index, low_chi)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['gauss_width'] = lsqs_gauss_val
                data_returns['pbf_width'] = lsqs_pbf_val
                data_returns[f'{pbf_type}'] = 'exp'

                return(data_returns)

            #case where gaussian width is set but fitting for pbf width
            elif gwidth_ind != -1 and pbfwidth_ind == -1:

                print(f'Decaying exponential pbf. Set gaussian width = {gauss_fwhm[gwidth_ind]} microseconds. Fitting for PBF Width.')

                num_par = 3 # number of fitted parameters

                gwidth = gauss_fwhm[gwidth_ind]

                chi_sqs_array = np.zeros(num_pbfwidth)
                for i in pbfwidth_inds:

                    template = gauss_convolved_profiles[pbf_type][i][gwidth_ind]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i] = chi_sq

                self.chi_plot(chi_sqs_array, pbf_type, gwidth = gwidth)

                low_chi = find_nearest(chi_sqs_array, 0.0)[0]
                lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
                pbf_width_fin = widths[lsqs_pbf_index]

                if chi_sqs_array[0] < low_chi+(1/(self.bin_num_care-num_par)) or chi_sqs_array[-1] < low_chi+(1/(self.bin_num_care-num_par)):
                    raise Exception('NOT CONVERGING ENOUGH')

                tau_fin = exp_tau_values[lsqs_pbf_index]

                #ERROR TEST - one reduced chi-squared unit above and below and these
                #chi-squared bins are for varying pbf width
                below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0]
                above = find_nearest(chi_sqs_array[lsqs_pbf_index+1:], low_chi+(1/(self.bin_num_care-num_par)))[1][0][0] + lsqs_pbf_index + 1

                tau_low = tau_fin - exp_tau_values[below]
                tau_up = exp_tau_values[above] - tau_fin

                self.fit_plot(pbf_type, 0, lsqs_pbf_index, gwidth_ind, low_chi, low_pbf = below, high_pbf = above)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['gauss_width'] = gwidth
                data_returns['pbf_width'] = pbf_width_fin
                data_returns[f'{pbf_type}'] = 'exp'
                data_returns['tau_low'] = tau_low
                data_returns['tau_up'] = tau_up

                return(data_returns)


#THEN MOVE ONTO INTRINSIC VERSION, KEEPING AS CONCISE AS POSSIBLE AND ORGANIZED
#THEN SEE ABOUT POWER LAWS, BEST FIT WIDTHS, ETC AND TESTING THE FUNCTIONS
#THEN REVAMP FIN FIT PLOTTING
#THEN TOA DELAY THINGS



    # def fit_pwr_law_g(self, intrins = False, beta_ind = -1):
    #     '''This method tests a number of different power law indices for the
    #     varaition of gaussian width over frequency.
    #
    #     Tested this over several MJDs and the best fit power law is about 0.9.
    #
    #     Did the same thing but with intrinsic pulse from s-band and best fit
    #     power law is about 0.0.'''
    #
    #     v_0_dece = 1742.0
    #
    #     if beta_ind != -1 and intrins:
    #         v_0_gwifth_0_dece = 11.7
    #     elif intrins:
    #         v_0_gwifth_0_dece = 11.7
    #     else:
    #         v_0_gwifth_0_dece = 70.0 # reference width in microseconds for v_0_dece freq
    #
    #     pwr_ind = np.arange(0,2,0.1) # range of gaussian width power law indexes to choose from
    #
    #     chi_sqs_collect = np.zeros(len(pwr_ind))
    #
    #     for i in range(len(pwr_ind)):
    #
    #         for ii in range(self.num_sub):
    #
    #             self.init_freq_subint(ii)
    #
    #             gwidth_set = v_0_gwifth_0_dece * np.power((self.freq_suba / v_0_dece), -pwr_ind[i])
    #             gwidth_ind = find_nearest(gauss_fwhm, gwidth_set)[1][0][0]
    #
    #             if beta_ind != -1 and intrins:
    #                 dataret = self.fit(ii, beta_ind = beta_ind, gwidth_ind = gwidth_ind, intrins = True)
    #             elif intrins:
    #                 dataret = self.fit(ii, dec_exp = True, gwidth_ind = gwidth_ind, intrins = True)
    #             else:
    #                 dataret = self.fit(ii, dec_exp = True, gwidth_ind = gwidth_ind)
    #
    #             chi_sqs_collect[i] += dataret[0]
    #
    #     plt.figure(10)
    #     plt.plot(pwr_ind, chi_sqs_collect, drawstyle = 'steps')
    #     plt.xlabel('Gaussian Width Power Law Indices')
    #     plt.ylabel('Summed Chi Squared')
    #     plt.title('Chi-Squared for Gwidth Power Law Index (over all freqs)')
    #     title = f'EXP||gwidth_pwrlaw|chisq_plot|MJD={self.mjd_round}.pdf'
    #     if beta_ind != -1 and intrins:
    #         title = f'BETA={betaselect[beta_ind]}|INTRINS|gwidth_pwrlaw|chisq_plot|MJD={self.mjd_round}.pdf'
    #     elif intrins:
    #         title = f'EXP|INTRINS|gwidth_pwrlaw|chisq_plot|MJD={self.mjd_round}.pdf'
    #     plt.savefig(title)
    #     plt.close(10)
    #     print(title)
    #
    #     lowest_chi_ind = np.where((chi_sqs_collect == np.min(chi_sqs_collect)))[0][0]
    #
    #     return(pwr_ind[lowest_chi_ind])


# rid of gwidth power law for now

# if gwidth_pwr_law == True and dec_exp == True:
#
#     v_0_dece = 1742.0
#
#     if intrins:
#         v_0_gwifth_0_dece = 11.8 # reference width in microseconds for v_0_dece freq
#     else:
#         v_0_gwifth_0_dece = 70.0 # reference width in microseconds for v_0_dece freq
#
#     if intrins:
#         pwr_ind = 0.0 #the tested best fit power law index for g width
#     else:
#         pwr_ind = 0.9 #the tested best fit - reference function below
#
#     gwidth_set = v_0_gwifth_0_dece * np.power((freq_care / v_0_dece), -pwr_ind)
#     gwidth_ind = find_nearest(gauss_fwhm, gwidth_set)[1][0][0]
#
# elif gwidth_pwr_law and dec_exp == False and zind == -1 and intrins:
#
#     v_0_dece = 1742.0
#
#     v_0_gwifth_0_dece = 11.8 # reference width in microseconds for v_0_dece freq
#
#     pwr_ind = 0.0 #need to fill in with correct power law index
#
#     gwidth_set = v_0_gwifth_0_dece * np.power((freq_care / v_0_dece), -pwr_ind)
#     gwidth_ind = find_nearest(gauss_fwhm, gwidth_set)[1][0][0]
#
