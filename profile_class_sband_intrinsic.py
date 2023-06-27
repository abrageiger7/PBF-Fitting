"""
Created June 2023
@author: Abra Geiger abrageiger7

Class for profile fitting intrinsic shape of sband profile.
"""

import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
from pypulse.singlepulse import SinglePulse
import pickle

from fit_functions import *

from profile_class_gaussian import Profile_Gauss

#fitting templates
with open(f'sband_intrins_convolved_profiles_phasebins={phase_bins}.pkl', 'rb') as fp:
    sband_intrins_convolved_profiles = pickle.load(fp)

#tau values corresponding to above templates
with open('tau_values.pkl', 'rb') as fp:
    tau_values = pickle.load(fp)

beta_tau_values = tau_values['beta']
exp_tau_values = tau_values['exp']
zeta_tau_values = tau_values['zeta']

#intrinsic widths
intrinss_fwhm = np.load('sband_intrins_fwhm.npy')



class Profile_Intrinss(Profile_Gauss):

    #inherits the initializer, fit_sing, comp_fse, init_freq_subint,

    def chi_plot(self, chi_sq_arr, pbf_type, bzeta = -1, iwidth = -1, pbfwidth = -1):

        '''Plots the inputted chi_sq_arr against the given parameters.

        If iwidth is -1, indicates that the chi-squared surface exists over
        all possible intrinsic widths. Same for pbfwidth.

        pbf_type is one of these strings: 'beta', 'zeta', 'exp'

        iwidth and pbfwidth are actual values of intrinsic width and pbf stretch
        factor. bzeta is the actual value of beta or zeta if not exp pbf.

        Differences from parent method:
        - different intrinsic width plot ticks
        - different plot titles'''

        #check of logical function inputs

        if pbf_type != 'beta' and pbf_type != 'zeta' and pbf_type != 'exp':
            raise Exception('Invalid pbf_type.')

        if pbf_type == 'beta' or pbf_type == 'zeta':
            if bzeta == -1:
                raise Exception('Please indicate the b/zeta value used.')

        plt.figure(45)

        if iwidth == -1 and pbfwidth == -1: #neither width set, so 2D chi^2 surface

            plt.title("Fit Chi-sqs")
            plt.xlabel("Intrinsic FWHM (microseconds)")
            plt.ylabel("PBF Width")

            #adjust the imshow tick marks
            iwidth_ticks = np.zeros(10)
            for ii in range(10):
                iwidth_ticks[ii] = str(intrinss_fwhm[ii*20])[:3]
            pbf_ticks = np.zeros(10)
            for ii in range(10):
                pbf_ticks[ii] = str(widths[ii*40])[:3]
            plt.xticks(ticks = np.linspace(0,num_gwidth,num=10), labels = iwidth_ticks)
            plt.yticks(ticks = np.linspace(0,num_pbfwidth,num=10), labels = pbf_ticks)

            plt.imshow(chi_sq_arr, cmap=plt.cm.viridis_r, origin = 'lower', aspect = 0.25)
            plt.colorbar()

            if pbf_type == 'beta':
                    title = f"BETA={bzeta}|INTRINSS|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}.pdf"

            elif pbf_type == 'exp':
                    title = f"EXP|INTRINSS|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}.pdf"

            elif pbf_type == 'zeta':
                    title = f"ZETA={bzeta}|INTRINSS|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}.pdf"

            plt.savefig(title)
            print(title)
            plt.close(45)

        elif iwidth != -1: #iwidth set, so 1D chi^2 surface

            iwidth_round = int(np.around(iwidth))

            plt.title('Fit Chi-sqs')
            plt.xlabel('PBF Width')
            plt.ylabel('Reduced Chi-Sq')
            plt.plot(widths, chi_sq_arr, drawstyle='steps-pre')

            if pbf_type == 'beta':
                    title = f"BETA={bzeta}|INTRINSS|PBF_fit_chisq_setg|MJD={self.mjd_round}|FREQ={self.freq_round}|IW={iwidth_round}.pdf"

            elif pbf_type == 'exp':
                    title = f"EXP|INTRINSS|PBF_fit_chisq_setg|MJD={self.mjd_round}|FREQ={self.freq_round}|IW={iwidth_round}.pdf"

            elif pbf_type == 'zeta':
                    title = f"ZETA={bzeta}|INTRINSS|PBF_fit_chisq_setg|MJD={self.mjd_round}|FREQ={self.freq_round}|IW={iwidth_round}.pdf"

            plt.savefig(title)
            print(title)
            plt.close(45)

    def fit_plot(self, pbf_type, bzeta_ind, pbfwidth_ind, iwidth_ind, low_chi, low_pbf = -1, high_pbf = -1):

        '''Plots and saves the fit of the profile subaveraged data to the
        template indicated by the argument indexes and the bolean
        indicating if decaying exponential wanted for the broadening function.

        beta_ind, pbfwidth_ind, iwidth_ind are ints; exp is boolean

        If low_pbf and or high_pbf != -1, plots additional fits in different color.
        These should be used for demonstrating error on tau and how this makes the
        fit different.

        Differences from parent method:

        - different profiles - sband_intrins_convolved_profiles ['zeta'] or ['beta'] or ['exp']
        - different fwhm array - intrinss_fwhm vs gauss_fwhm
        - different plot titles'''

        #test that arguement combination is logical

        if pbf_type != 'beta' and pbf_type != 'zeta' and pbf_type != 'exp':
            raise Exception('Invalid pbf_type.')

        if pbf_type == 'beta' or pbf_type == 'zeta':
            if bzeta_ind == -1:
                raise Exception('Please indicate the b/zeta index.')
            elif type(bzeta_ind) != int:
                raise Exception('bzeta_ind must be an integer.')

        #depending on pbf type, get profiles and taus

        if pbf_type == 'beta':
            i = sband_intrins_convolved_profiles[pbf_type][bzeta_ind][pbfwidth_ind][iwidth_ind]
            tau_val = beta_tau_values[bzeta_ind][pbfwidth_ind]

            if low_pbf != -1:
                low_pbf_i = sband_intrins_convolved_profiles[pbf_type][bzeta_ind][low_pbf][iwidth_ind]
                tau_val_low = beta_tau_values[bzeta_ind][low_pbf]
            if high_pbf != -1:
                high_pbf_i = sband_intrins_convolved_profiles[pbf_type][bzeta_ind][high_pbf][iwidth_ind]
                tau_val_high = beta_tau_values[bzeta_ind][high_pbf]

        elif pbf_type == 'exp':
            i = sband_intrins_convolved_profiles[pbf_type][pbfwidth_ind][iwidth_ind]
            tau_val = exp_tau_values[pbfwidth_ind]

            if low_pbf != -1:
                low_pbf_i = sband_intrins_convolved_profiles[pbf_type][low_pbf][iwidth_ind]
                tau_val_low = exp_tau_values[low_pbf]
            if high_pbf != -1:
                high_pbf_i = sband_intrins_convolved_profiles[pbf_type][high_pbf][iwidth_ind]
                tau_val_high = exp_tau_values[high_pbf]

        elif pbf_type == 'zeta':
            i = sband_intrins_convolved_profiles[pbf_type][bzeta_ind][pbfwidth_ind][iwidth_ind]
            tau_val = zeta_tau_values[bzeta_ind][pbfwidth_ind]

            if low_pbf != -1:
                low_pbf_i = sband_intrins_convolved_profiles[pbf_type][bzeta_ind][low_pbf][iwidth_ind]
                tau_val_low = zeta_tau_values[bzeta_ind][low_pbf]
            if high_pbf != -1:
                high_pbf_i = sband_intrins_convolved_profiles[pbf_type][bzeta_ind][high_pbf][iwidth_ind]
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
            plt.plot(time, fitted_templateh, alpha = 0.5, label = fr'Upper Error; $\tau$ = {int(np.around(tau_val_high))} $\mu$s', color = 'orange', lw = 3)
        if low_pbf != -1:
            plt.plot(time, fitted_templatel, alpha = 0.5, label = fr'Lower Error; $\tau$ = {int(np.around(tau_val_low))} $\mu$s', color = 'orange', lw = 3)
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

        iwidth_round = int(np.around(intrinss_fwhm[iwidth_ind]))
        pbfwidth_round = int(np.around(widths[pbfwidth_ind]))

        if pbf_type == 'beta':
            title = f'BETA={betaselect[bzeta_ind]}|INTRINSS|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}||PBFW={pbfwidth_round}|IW={iwidth_round}.pdf'
        elif pbf_type == 'exp':
            title = f'EXP|INTRINSS|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}|PBFW={pbfwidth_round}|IW={iwidth_round}.pdf'
        elif pbf_type == 'zeta':
            title = f'ZETA={zetaselect[bzeta_ind]}|INTRINSS|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}|PBFW={pbfwidth_round}|IW={iwidth_round}.pdf'

        plt.savefig(title)
        print(title)
        plt.close(50)


    def fit(self, freq_subint_index, pbf_type, bzeta_ind = -1, iwidth_ind = -1, pbfwidth_ind = -1):
        '''Calculates the best broadening function and corresponding parameters
        for the Profile object.

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

        self.init_freq_subint(freq_subint_index)

        #number of each parameter in the parameter grid
        num_beta = np.size(betaselect)

        #number of each parameter in the parameter grid
        num_zeta = np.size(zetaselect)

        beta_inds = np.arange(num_beta)
        zeta_ind = np.arange(num_zeta)
        iwidth_inds = np.arange(num_gwidth)
        pbfwidth_inds = np.arange(num_pbfwidth)

        #set more convenient names for the data to be fitted to
        data_care = self.data_suba
        freq_care = self.freq_suba

        print('Intrinsic shape is sband average.')

        if pbf_type == 'beta':

            #case where beta, iwidth, and pbfwidth are free
            if iwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind == -1:

                print('Fitting for beta, intrinsic width, and PBF width.')

                num_par = 5 #number of fitted parameters

                chi_sqs_array = np.zeros((num_beta, num_pbfwidth, num_gwidth))
                for i in itertools.product(beta_inds, pbfwidth_inds, iwidth_inds):

                    template = sband_intrins_convolved_profiles[pbf_type][i[0]][i[1]][i[2]]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i[0]][i[1]][i[2]] = chi_sq

                chi_sqs_collect = np.zeros(num_beta)
                pbf_width_collect = np.zeros(num_beta)
                intrinsic_width_collect = np.zeros(num_beta)
                taus_collect = np.zeros(num_beta)
                taus_err_collect = np.zeros((2,num_beta))
                ind = 0
                for i in chi_sqs_array:

                    beta = betaselect[ind]

                    #scale the chi-squared array by the rms value of the profile

                    self.chi_plot(i, pbf_type, bzeta = beta)

                    #least squares
                    low_chi = find_nearest(i, 0.0)[0]
                    chi_sqs_collect[ind] = low_chi

                    if i[0][0] < low_chi+1 and i[-1][-1] < low_chi+1:
                        raise Exception('NOT CONVERGING ENOUGH') #stops code if not
                        #enough parameters to reach reduced low_chi + 1 before end
                        #of parameter space

                    #lsqs pbf width
                    lsqs_pbf_index = find_nearest(i, 0.0)[1][0][0]
                    lsqs_pbf_val = widths[lsqs_pbf_index]
                    pbf_width_collect[ind] = lsqs_pbf_val

                    #lsqs intrinsic width
                    lsqs_intrins_index = find_nearest(i, 0.0)[1][1][0]
                    lsqs_intrins_val = intrinss_fwhm[lsqs_intrins_index]
                    intrinsic_width_collect[ind] = lsqs_intrins_val

                    taus_collect[ind] = beta_tau_values[ind][lsqs_pbf_index]

                    self.fit_plot(pbf_type, ind, lsqs_pbf_index, lsqs_intrins_index, low_chi)

                    ind+=1

                low_chi = np.min(chi_sqs_collect)
                chi_beta_ind = np.where(chi_sqs_collect == low_chi)[0][0]

                beta_fin = betaselect[chi_beta_ind]
                pbf_width_fin = pbf_width_collect[chi_beta_ind]
                intrins_width_fin = intrinsic_width_collect[chi_beta_ind]
                tau_fin = taus_collect[chi_beta_ind]

                pbf_width_ind = np.where(widths == pbf_width_fin)[0][0]
                intrins_width_ind = np.where((intrinss_fwhm == intrinsic_width_collect[chi_beta_ind]))[0][0]

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
                        plt.ylabel('Overall Best Intrinsic Width FWHM (milliseconds)')
                        plt.plot(betaselect, intrinsic_width_collect) #already converted to micro fwhm
                        param = 'iwidth'

                    if i == 3:
                        plt.ylabel('Overall Best Tau (microseconds)')
                        plt.plot(betaselect, taus_collect)
                        param = 'tau'

                    title = f'ALLBETA|INTRINSS|PBF_fit_overall_{param}|MJD={self.mjd_round}|FREQ={self.freq_round}|bestBETA={betaselect[chi_beta_ind]}.pdf'
                    plt.savefig(title)
                    plt.close(i*4)

                #overall best fit plot
                self.fit_plot(pbf_type, chi_beta_ind, pbf_width_ind, intrins_width_ind, low_chi)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['intrins_width'] = intrins_width_fin
                data_returns['pbf_width'] = pbf_width_fin
                data_returns[f'{pbf_type}'] = beta_fin

                return(data_returns)

            #case where beta is set, but iwidth and pbfwidth free
            elif iwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                print(f'Set beta = {betaselect[bzeta_ind]}. Fitting for intrinsic width and PBF width.')

                num_par = 4 #number of fitted parameters

                beta = betaselect[bzeta_ind]

                chi_sqs_array = np.zeros((num_pbfwidth, num_gwidth))
                for i in itertools.product(pbfwidth_inds, iwidth_inds):

                    template = sband_intrins_convolved_profiles[pbf_type][bzeta_ind][i[0]][i[1]]
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

                #lsqs intrinsic width
                lsqs_intrins_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
                lsqs_intrins_val = intrinss_fwhm[lsqs_intrins_index]

                tau_fin = beta_tau_values[bzeta_ind][lsqs_pbf_index]

                self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, lsqs_intrins_index, low_chi)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['intrins_width'] = lsqs_intrins_val
                data_returns['pbf_width'] = lsqs_pbf_val
                data_returns[f'{pbf_type}'] = beta

                return(data_returns)

            #case where beta and intrinsix width are set, but fitting for pbf width
            elif iwidth_ind != -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                print(f'Set beta = {betaselect[bzeta_ind]} and intrinsic width = {intrinss_fwhm[iwidth_ind]} microseconds. Fitting for PBF Width.')

                num_par = 3 # number of fitted parameters

                beta = betaselect[bzeta_ind]
                iwidth = intrinss_fwhm[iwidth_ind]

                chi_sqs_array = np.zeros(num_pbfwidth)
                for i in pbfwidth_inds:

                    template = sband_intrins_convolved_profiles[pbf_type][bzeta_ind][i][iwidth_ind]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i] = chi_sq

                self.chi_plot(chi_sqs_array, pbf_type, bzeta = beta, iwidth = iwidth)

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

                self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, iwidth_ind, low_chi, low_pbf = below, high_pbf = above)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['intrins_width'] = iwidth
                data_returns['pbf_width'] = pbf_width_fin
                data_returns[f'{pbf_type}'] = beta
                data_returns['tau_low'] = tau_low
                data_returns['tau_up'] = tau_up

                return(data_returns)


        elif pbf_type == 'zeta':

            #case where zeta, iwidth, and pbfwidth are free
            if iwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind == -1:

                print('Fitting for zeta, intrinsic width, and PBF width.')

                num_par = 5 #number of fitted parameters

                chi_sqs_array = np.zeros((num_zeta, num_pbfwidth, num_gwidth))
                for i in itertools.product(zeta_inds, pbfwidth_inds, iwidth_inds):

                    template = sband_intrins_convolved_profiles[pbf_type][i[0]][i[1]][i[2]]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i[0]][i[1]][i[2]] = chi_sq

                chi_sqs_collect = np.zeros(num_zeta)
                pbf_width_collect = np.zeros(num_zeta)
                intrinsic_width_collect = np.zeros(num_zeta)
                taus_collect = np.zeros(num_zeta)
                taus_err_collect = np.zeros((2,num_zeta))
                ind = 0
                for i in chi_sqs_array:

                    zeta = zetaselect[ind]

                    #scale the chi-squared array by the rms value of the profile

                    self.chi_plot(i, pbf_type, bzeta = zeta)

                    #least squares
                    low_chi = find_nearest(i, 0.0)[0]
                    chi_sqs_collect[ind] = low_chi

                    if i[0][0] < low_chi+1 and i[-1][-1] < low_chi+1:
                        raise Exception('NOT CONVERGING ENOUGH') #stops code if not
                        #enough parameters to reach reduced low_chi + 1 before end
                        #of parameter space

                    #lsqs pbf width
                    lsqs_pbf_index = find_nearest(i, 0.0)[1][0][0]
                    lsqs_pbf_val = widths[lsqs_pbf_index]
                    pbf_width_collect[ind] = lsqs_pbf_val

                    #lsqs intrinsic width
                    lsqs_intrins_index = find_nearest(i, 0.0)[1][1][0]
                    lsqs_intrins_val = intrinss_fwhm[lsqs_intrins_index]
                    intrinsic_width_collect[ind] = lsqs_intrins_val

                    taus_collect[ind] = zeta_tau_values[ind][lsqs_pbf_index]

                    self.fit_plot(pbf_type, ind, lsqs_pbf_index, lsqs_intrins_index, low_chi)

                    ind+=1

                low_chi = np.min(chi_sqs_collect)
                chi_zeta_ind = np.where(chi_sqs_collect == low_chi)[0][0]

                zeta_fin = zetaselect[chi_zeta_ind]
                pbf_width_fin = pbf_width_collect[chi_zeta_ind]
                intrins_width_fin = intrinsic_width_collect[chi_zeta_ind]
                tau_fin = taus_collect[chi_zeta_ind]

                pbf_width_ind = np.where(widths == pbf_width_fin)[0][0]
                intrins_width_ind = np.where((intrinss_fwhm == intrinsic_width_collect[chi_beta_ind]))[0][0]

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
                        plt.ylabel('Overall Best Intrinsic Width FWHM (milliseconds)')
                        plt.plot(zetaselect, intrinsic_width_collect) #already converted to micro fwhm
                        param = 'iwidth'

                    if i == 3:
                        plt.ylabel('Overall Best Tau (microseconds)')
                        plt.plot(zetaselect, taus_collect)
                        param = 'tau'

                    title = f'ALLZETA|PBF_fit_overall_{param}|MJD={self.mjd_round}|FREQ={self.freq_round}|bestZETA={zetaselect[chi_zeta_ind]}.pdf'
                    plt.savefig(title)
                    plt.close(i*4)

                self.fit_plot(pbf_type, chi_zeta_ind, pbf_width_ind, intrins_width_ind, low_chi)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['intrins_width'] = intrins_width_fin
                data_returns['pbf_width'] = pbf_width_fin
                data_returns[f'{pbf_type}'] = zeta_fin

                return(data_returns)

            #case where beta is set, but iwidth and pbfwidth free
            elif iwidth_ind == -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                print(f'Set zeta = {zetaselect[bzeta_ind]}. Fitting for intrinsic width and PBF width.')

                num_par = 4 #number of fitted parameters

                zeta = zetaselect[bzeta_ind]

                chi_sqs_array = np.zeros((num_pbfwidth, num_gwidth))
                for i in itertools.product(pbfwidth_inds, iwidth_inds):

                    template = sband_intrins_convolved_profiles[pbf_type][bzeta_ind][i[0]][i[1]]
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

                #lsqs intrinsic width
                lsqs_intrins_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
                lsqs_intrins_val = intrinss_fwhm[lsqs_intrins_index]

                tau_fin = zeta_tau_values[bzeta_ind][lsqs_pbf_index]

                self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, lsqs_intrins_index, low_chi)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['intrins_width'] = lsqs_intrins_val
                data_returns['pbf_width'] = lsqs_pbf_val
                data_returns[f'{pbf_type}'] = zeta

                return(data_returns)

            #case where beta and intrinsic width are set, but fitting for pbf width
            elif iwidth_ind != -1 and pbfwidth_ind == -1 and bzeta_ind != -1:

                print(f'Set zeta = {zetaselect[bzeta_ind]} and intrinsic width = {intrinss_fwhm[iwidth_ind]} microseconds. Fitting for PBF Width.')

                num_par = 3 # number of fitted parameters

                zeta = zetaselect[bzeta_ind]
                iwidth = intrinss_fwhm[iwidth_ind]

                chi_sqs_array = np.zeros(num_pbfwidth)
                for i in pbfwidth_inds:

                    template = sband_intrins_convolved_profiles[pbf_type][bzeta_ind][i][iwidth_ind]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i] = chi_sq

                self.chi_plot(chi_sqs_array, pbf_type, bzeta = zeta, iwidth = iwidth)

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

                self.fit_plot(pbf_type, bzeta_ind, lsqs_pbf_index, iwidth_ind, low_chi, low_pbf = below, high_pbf = above)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['intrins_width'] = iwidth
                data_returns['pbf_width'] = pbf_width_fin
                data_returns[f'{pbf_type}'] = zeta
                data_returns['tau_low'] = tau_low
                data_returns['tau_up'] = tau_up

                return(data_returns)


        elif pbf_type == 'exp':

            #case where iwidth and pbfwidth are free
            if iwidth_ind == -1 and pbfwidth_ind == -1:

                print(f'Decaying exponential pbf. Fitting for intrinsic width and PBF width.')

                num_par = 4 #number of fitted parameters

                chi_sqs_array = np.zeros((num_pbfwidth, num_gwidth))
                for i in itertools.product(pbfwidth_inds, iwidth_inds):

                    template = sband_intrins_convolved_profiles[pbf_type][i[0]][i[1]]
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

                #lsqs intrinsic width
                lsqs_intrins_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
                lsqs_intrins_val = intrinss_fwhm[lsqs_intrins_index]

                tau_fin = exp_tau_values[lsqs_pbf_index]

                self.fit_plot(pbf_type, 0, lsqs_pbf_index, lsqs_intrins_index, low_chi)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['intrins_width'] = lsqs_intrins_val
                data_returns['pbf_width'] = lsqs_pbf_val
                data_returns[f'{pbf_type}'] = 'exp'

                return(data_returns)

            #case where intrinsic width is set but fitting for pbf width
            elif iwidth_ind != -1 and pbfwidth_ind == -1:

                print(f'Decaying exponential pbf. Set intrinsic width = {intrinss_fwhm[iwidth_ind]} microseconds. Fitting for PBF Width.')

                num_par = 3 # number of fitted parameters

                iwidth = intrinss_fwhm[iwidth_ind]

                chi_sqs_array = np.zeros(num_pbfwidth)
                for i in pbfwidth_inds:

                    template = sband_intrins_convolved_profiles[pbf_type][i][iwidth_ind]
                    chi_sq = self.fit_sing(template, num_par)
                    chi_sqs_array[i] = chi_sq

                self.chi_plot(chi_sqs_array, pbf_type, iwidth = iwidth)

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

                self.fit_plot(pbf_type, 0, lsqs_pbf_index, iwidth_ind, low_chi, low_pbf = below, high_pbf = above)

                data_returns = {}
                data_returns['low_chi'] = low_chi
                data_returns['tau_fin'] = tau_fin
                data_returns['fse_effect'] = self.comp_fse(tau_fin)
                data_returns['intrins_width'] = iwidth
                data_returns['pbf_width'] = pbf_width_fin
                data_returns[f'{pbf_type}'] = 'exp'
                data_returns['tau_low'] = tau_low
                data_returns['tau_up'] = tau_up

                return(data_returns)
