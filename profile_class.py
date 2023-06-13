"""
Created June 2023
Last Edited on Sun Jun 11 2023
@author: Abra Geiger abrageiger7

Class for profile fitting
"""

from fit_functions import *

class Profile:

    num_phase_bins = 2048 #number of phase bins for pulse period in data
    opr_size = 600 #number of phase bins for offpulse noise calculation

    def __init__(self, mjd, data, frequencies, dur):
        '''
        mjd (float) - epoch of observation
        data (2D array) - pulse data for epoch
        frequencies (1D array) - frequencies corresponding to the data channels
        dur (float) - observation duration in seconds
        '''

        #initialize the object attributes

        self.mjd = mjd
        self.mjd_round = np.round(mjd)
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

        num_masked = Profile.num_phase_bins - (self.stop_index-self.start_index)

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

    def chi_plot(self, chi_sq_arr, beta = -1, exp = False, gwidth = -1, pbfwidth = -1, zeta = -1):

        '''Plots the inputted chi_sq_arr against the given parameters.

        If gwidth is -1, indicates that the chi-squared surface exists over
        all possible gaussian widths. Same for pbfwidth.

        If beta is -1, must be for decaying exponential and exp != -1. Vice
        versa.'''

        plt.figure(45)

        if gwidth == -1 and pbfwidth == -1:

            plt.title("Fit Chi-sqs")
            plt.xlabel("Gaussian FWHM (microseconds)")
            plt.ylabel("PBF Width")

            #adjust the imshow tick marks
            gauss_ticks = np.zeros(10)
            for ii in range(10):
                gauss_ticks[ii] = str(gauss_fwhm[ii*5])[:3]
            pbf_ticks = np.zeros(10)
            for ii in range(10):
                pbf_ticks[ii] = str(widths[ii*20])[:3]
            plt.xticks(ticks = np.linspace(0,50,num=10), labels = gauss_ticks)
            plt.yticks(ticks = np.linspace(0,200,num=10), labels = pbf_ticks)

            plt.imshow(chi_sq_arr, cmap=plt.cm.viridis_r, origin = 'lower', aspect = 0.25)
            plt.colorbar()

            if beta != -1:
                title = f"ONEB|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}|BETA={beta}.png"

            elif exp:
                title = f"EXP|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}.png"

            plt.savefig(title)
            plt.close(45)

        elif gwidth != -1:

            gwidth_round = str(gwidth)[:3]

            plt.title('Fit Chi-sqs')
            plt.xlabel('PBF Width')
            plt.ylabel('Reduced Chi-Sq')
            plt.plot(widths, chi_sq_arr)

            if beta != -1:
                title = f"ONEBSETG|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}|BETA={beta}|GWIDTH={gwidth_round}.png"

            elif exp:
                title = f"EXPSETG|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}|GWIDTH={gwidth_round}.png"

            elif zeta != -1:
                title = f"ONEZSETG|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}|ZETA={zeta}|GWIDTH={gwidth_round}.png"

            plt.savefig(title)
            plt.close(45)


    def fit_plot(self, zbeta_ind, pbfwidth_ind, gwidth_ind, exp = False, zeta = False):

        '''Plots and saves the fit of the profile subaveraged data to the
        template indicated by the argument indexes and the bolean
        indicating if decaying exponential wanted for the broadening function.

        beta_ind, pbfwidth_ind, gwidth_ind are ints; exp is boolean'''

        if not exp and not zeta:
            i = convolved_profiles[zbeta_ind][pbfwidth_ind][gwidth_ind]
        elif exp:
            i = convolved_profiles_exp[pbfwidth_ind][gwidth_ind]
        elif zeta:
            i = zeta_convolved_profiles[zbeta_ind][pbfwidth_ind][gwidth_ind]

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
        if not exp and not zeta:
            plt.title(f'Best Fit Template over Data with Beta = {betaselect[zbeta_ind]}')
        elif exp:
            plt.title('Best Fit Template over Data')
        elif zeta:
            plt.title(f'Best Fit Template over Data with Zeta = {zetaselect[zbeta_ind]}')

        plt.ylabel('Pulse Intensity')
        plt.plot(time, self.data_suba*self.mask, '.', ms = '2.4')
        plt.plot(time, fitted_template)
        frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
        plt.plot()

        #Residual plot
        difference = np.subtract(self.data_suba*self.mask, fitted_template)
        frame2=fig1.add_axes((.1,.1,.8,.2))
        plt.plot(time, difference, '.', ms = '2.4')
        plt.xlabel('Pulse Period (milliseconds)')
        plt.ylabel('Residuals')
        plt.plot()

        gwidth_round = str(gauss_fwhm[gwidth_ind])[:3]
        pbfwidth_round = str(widths[pbfwidth_ind])[:3]

        if not exp and not zeta:
            title = f'FIT|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}|BETA={betaselect[beta_ind]}|PBFW={pbfwidth_round}|GW={gwidth_round}.png'
        elif exp:
            title = f'FIT|EXP|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}|PBFW={pbfwidth_round}|GW={gwidth_round}.png'
        if not exp and not zeta:
            title = f'FIT|ZETA|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}|ZETA={zetaselect[zbeta_ind]}|PBFW={pbfwidth_round}|GW={gwidth_round}.png'

        plt.savefig(title)
        plt.close(50)

    def comp_fse(self, tau):
        '''Calculates the error due to the finite scintile effect. Reference
        Michael's email for details. Tau is the fitted time delay in microseconds.'''

        T = self.dur
        B = 12.5 #approximate frequency range for each channel in MHz
        D = 0.64 #approximate distance to the pulsar in kpc
        nt = 0.2 #filling factor over time
        nv = 0.2 #filling factor over bandwidth

        v = self.freq_suba / 1000.0
        vd = (1.16)/(2.0*(math.pi)*tau) #MHz
        #td = ((math.sqrt(D*(vd*1000.0)))/v)*(1338.62433862) #seconds
        td = ((math.sqrt(D*(vd)))/v)*(1338.62433862) #seconds
        nscint = (1.0 + nt*(T/td))*(1.0 + nv*(B/vd))
        error = tau/(math.sqrt(nscint)) #microseconds
        #print(error) -> seems to be a very small number of microseconds (order of 1)
        return(error)

    def fit(self, freq_subint_index, beta_ind = -1, gwidth_ind = -1, pbfwidth_ind = -1, dec_exp = False, zind = -1):
        '''Calculates the best broadening function and corresponding parameters
        for the Profile object.

        beta_ind (int): if nonzero, set beta to this index of betaselect
        gwidth_ind (int): if nonzero, set gauss width to this index of gauss_widths
        pbf_width_ind (int) : if nonzero, set pbf width to this index of widths
        dec_exp (bool) : if True, fit decaying exponential broadening functions

        No error calculations for varying more than one parameter
        '''

        #number of each parameter in the parameter grid
        num_beta = np.size(betaselect)
        num_gwidth = np.size(gauss_widths)
        num_pbfwidth = np.size(widths)

        beta_inds = np.arange(num_beta)
        gwidth_inds = np.arange(num_gwidth)
        pbfwidth_inds = np.arange(num_pbfwidth)


        #isolate the data profile at the frequency desired for this fit
        self.data_suba = self.subaveraged_info[0][freq_subint_index]
        self.freq_suba = self.subaveraged_info[1][freq_subint_index]
        self.freq_round = np.round(self.freq_suba)


        #Calculates mode of data profile to shift template to
        x = np.max(self.data_suba)
        self.xind = np.where(self.data_suba == x)[0][0]


        #Set the offpulse regions to zero for fitting because essentially
        #oscillating there.
        #This region size varies depending on frequency

        mask = np.zeros(Profile.num_phase_bins)

        if self.freq_suba >= 1600:
            self.start_index = 700
            self.stop_index = 1548
        elif self.freq_suba >= 1400 and self.freq_suba < 1600:
            self.start_index = 700
            self.stop_index = 1648
        elif self.freq_suba >= 1200 and self.freq_suba < 1400:
            self.start_index = 650
            self.stop_index = 1798
        elif self.freq_suba >= 1000 and self.freq_suba < 1200:
            self.start_index = 600
            self.stop_index = 1948
        mask[self.start_index:self.stop_index] = 1.0

        self.mask = mask


        #Calculates the root mean square noise of the off pulse.
        #Used later to calculate normalized chi-squared.

        rms_collect = 0
        for i in range(Profile.opr_size):
            rms_collect += self.data_suba[i]**2
        rms = math.sqrt(rms_collect/Profile.opr_size)

        self.rms_noise = rms


        #set more convenient names for the data to be fitted to
        data_care = self.data_suba
        freq_care = self.freq_suba


        #case where beta, gaussian width, and pbf width are not set
        if beta_ind == -1 and gwidth_ind == -1 and pbfwidth_ind == -1 and dec_exp == False:

            num_par = 5 #number of fitted parameters

            chi_sqs_array = np.zeros((num_beta, num_pbfwidth, num_gwidth))
            for i in itertools.product(beta_inds, pbfwidth_inds, gwidth_inds):

                template = convolved_profiles[i[0]][i[1]][i[2]]
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

                self.chi_plot(chi_sqs_array, beta = beta)

                #least squares
                low_chi = find_nearest(chi_sqs_array, 0.0)[0]
                chi_sqs_collect[ind] = low_chi

                if chi_sqs_array[0] < low_chi+1 and chi_sqs_array[-1] < low_chi+1:
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

                taus_collect[ind] = tau.tau_values[ind][lsqs_pbf_index]

                #ERROR TEST - one reduced chi-squared unit above and below and these
                #chi-squared bins are for varying pbf width

                #low_bound_chi_val = low_chi+.99
                #up_bound_chi_val = low_chi+1.01

                #NEED TO DO FIND NEAREST INSTEAD AND CONSIDER 2D SURFACE
                #ALSO CAREFUL NOT TO ASSUME SMOOTHNESS FOR CHI CURVE

                #below = find_nearest(chi_sqs_array, low_chi+1)[]
                #above = find_nearest(chi_sqs_array[:low_chi_index])

                #below = np.where((chi_sqs_array >= low_bound_chi_val \
                #& chi_sqs_array <= up_bound_chi_val))
                #above = np.where((chi_sqs_array[low_chi_index:] >= \
                #low_bound_chi_val & chi_sqs_array[:low_chi_index:] <=
                #up_bound_chi_val))[0][0] + low_chi_index

                #tau_arr = tau_values[ind]
                #tau_low = tau_arr[below]
                #tau_up = tau_arr[above]

                #taus_err_collect[0][ind] = tau_low
                #taus_err_collect[1][ind] = tau_up

                if below <= 40:
                    raise Exception('Different Tau Error Conversion May Be Needed')

                self.fit_plot(ind, lsqs_pbf_index, lsqs_gauss_index)

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

                title = f'ALL|PBF_fit_overall_{param}|MJD={self.mjd_round}|FREQ={self.freq_round}|bestBETA={betaselect[chi_beta_ind]}.png'
                plt.savefig(title)
                plt.close(i*4)

            self.fit_plot(chi_beta_ind, pbf_width_ind, gauss_width_ind)

            return(low_chi, tau_fin, self.comp_fse(tau_fin), gauss_width_fin, pbf_width_fin, beta_fin)

        #case where beta and gaussian width are set, but fitting for pbf width
        elif beta_ind != -1 and gwidth_ind != -1 and dec_exp == False:

            num_par = 3 # number of fitted parameters

            beta = betaselect[beta_ind]
            gwidth = gauss_fwhm[gwidth_ind]

            chi_sqs_array = np.zeros(num_pbfwidth)
            for i in pbfwidth_inds:

                template = convolved_profiles[beta_ind][i][gwidth_ind]
                chi_sq = self.fit_sing(template, num_par)
                chi_sqs_array[i] = chi_sq

            self.chi_plot(chi_sqs_array, beta = beta, gwidth = gwidth)

            low_chi = find_nearest(chi_sqs_array, 0.0)[0]
            lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
            pbf_width_fin = widths[lsqs_pbf_index]

            if chi_sqs_array[0] < low_chi+1 and chi_sqs_array[-1] < low_chi+1:
                raise Exception('NOT CONVERGING ENOUGH')

            tau_fin = tau_values[beta_ind][lsqs_pbf_index]

            #ERROR TEST - one reduced chi-squared unit above and below and these
            #chi-squared bins are for varying pbf width
            below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+1)[1][0][0]
            above = find_nearest(chi_sqs_array[lsqs_pbf_index:], low_chi+1)[1][0][0] + lsqs_pbf_index

            tau_arr = tau_values[beta_ind]
            tau_low = tau_fin - tau_arr[below]
            tau_up = tau_arr[above] - tau_fin

            self.fit_plot(beta_ind, lsqs_pbf_index, gwidth_ind)

            return(low_chi, tau_fin, tau_low, tau_up, self.comp_fse(tau_fin), gwidth, pbf_width_fin, beta)

        #case where gaussian width is set and fitting amongst decaying exponential templates
        elif dec_exp == True and gwidth_ind != -1:

            num_par = 3 #number of fitted parameters

            gwidth = gauss_fwhm[gwidth_ind]
            chi_sqs_array = np.zeros(num_pbfwidth)

            for i in pbfwidth_inds:

                template = convolved_profiles_exp[i][gwidth_ind]
                chi_sq = self.fit_sing(template, num_par)
                chi_sqs_array[i] = chi_sq

            self.chi_plot(chi_sqs_array, exp = True, gwidth = gwidth)

            low_chi = find_nearest(chi_sqs_array, 0.0)[0]
            lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
            pbf_width_fin = widths[lsqs_pbf_index]

            if chi_sqs_array[0] < low_chi+1 and chi_sqs_array[-1] < low_chi+1:
                raise Exception('NOT CONVERGING ENOUGH')

            tau_fin = tau_values_exp[lsqs_pbf_index]

            #ERROR TEST - one reduced chi-squared unit above and below and these
            #chi-squared bins are for varying pbf width

            below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+1)[1][0][0]
            above = find_nearest(chi_sqs_array[lsqs_pbf_index:], low_chi+1)[1][0][0] + lsqs_pbf_index

            tau_low = tau_fin - tau_values_exp[below]
            tau_up = tau_values_exp[above] - tau_fin

            self.fit_plot(0, lsqs_pbf_index, gwidth_ind, exp = True)

            return(low_chi, tau_fin, tau_low, tau_up, self.comp_fse(tau_fin), gwidth, pbf_width_fin)

        #case where fitting for decaying exponential, but gauss width and pbf width not set
        elif gwidth_ind == -1 and pbfwidth_ind == -1 and dec_exp == True:

            num_par = 4 #number of fitted parameters

            chi_sqs_array = np.zeros((num_pbfwidth, num_gwidth))
            for i in itertools.product(pbfwidth_inds, gwidth_inds):

                template = convolved_profiles_exp[i[0]][i[1]]
                chi_sq = self.fit_sing(template, num_par)
                chi_sqs_array[i[0]][i[1]] = chi_sq

            self.chi_plot(chi_sqs_array, exp = True)

            low_chi = find_nearest(chi_sqs_array, 0.0)[0]
            lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
            lsqs_pbf_val = widths[lsqs_pbf_index]
            lsqs_gauss_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
            lsqs_gauss_val = widths_gaussian[lsqs_gauss_index]

            tau_fin = tau_values_exp[lsqs_pbf_index]

            self.fit_plot(0, lsqs_pbf_index, lsqs_gauss_index, exp = True)

            return(low_chi, lsqs_gauss_val, lsqs_pbf_val, tau_fin)

        elif zind != -1 and gwidth_ind != -1 and dec_exp == False:

            num_par = 3 # number of fitted parameters

            bzta = zetaselect[zeta_ind]
            gwidth = gauss_fwhm[gwidth_ind]

            chi_sqs_array = np.zeros(num_pbfwidth)
            for i in pbfwidth_inds:

                template = zeta_convolved_profiles[zeta_ind][i][gwidth_ind]
                chi_sq = self.fit_sing(template, num_par)
                chi_sqs_array[i] = chi_sq

            self.chi_plot(chi_sqs_array, zeta = zeta, gwidth = gwidth)

            low_chi = find_nearest(chi_sqs_array, 0.0)[0]
            lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
            pbf_width_fin = widths[lsqs_pbf_index]

            if chi_sqs_array[0] < low_chi+1 and chi_sqs_array[-1] < low_chi+1:
                raise Exception('NOT CONVERGING ENOUGH')

            tau_fin = zeta_tau_values[zeta_ind][lsqs_pbf_index]

            #ERROR TEST - one reduced chi-squared unit above and below and these
            #chi-squared bins are for varying pbf width
            below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+1)[1][0][0]
            above = find_nearest(chi_sqs_array[lsqs_pbf_index:], low_chi+1)[1][0][0] + lsqs_pbf_index

            tau_arr = zeta_tau_values[beta_ind]
            tau_low = tau_fin - tau_arr[below]
            tau_up = tau_arr[above] - tau_fin

            self.fit_plot(zeta_ind, lsqs_pbf_index, gwidth_ind, zeta=True)

            return(low_chi, tau_fin, tau_low, tau_up, self.comp_fse(tau_fin), gwidth, pbf_width_fin, zeta)





#===============================================================================
# Time averaging every 8 data points
# ==============================================================================

# t = np.linspace(0, 2.15, 2048//8)
#
# #print(convolved_profiles)
# print(np.shape(convolved_profiles))
# #print(convolved_profiles_exp)
# print(np.shape(convolved_profiles_exp))
#
# convolved_profiles_timea = np.zeros((12,200,50,2048//8))
#
# for i in range(12):
#     for ii in range(200):
#         for iii in range(50):
#             not_2048 = np.zeros(2048//8)
#             for iv in range(2048//8):
#                 not_2048[iv] = np.average(convolved_profiles[i][ii][iii][(8*iv):(iv*8)+8])
#             convolved_profiles_timea[i][ii][iii] = not_2048
#
# convolved_profiles_exp_timea = np.zeros((200,50, 2048//8))
#
# for i in range(200):
#     for ii in range(50):
#         not_2048 = np.zeros(2048//8)
#         for iii in range(2048//8):
#             not_2048[iii] = np.average(convolved_profiles_exp[i][ii][(8*iii):(iii*8)+8])
#         convolved_profiles_exp_timea[i][ii] = not_2048
#
# convolved_profiles = convolved_profiles_timea
# convolved_profiles_exp = convolved_profiles_exp_timea
#
#
# def subaverages4(mjdi, data, freqsm, plot = False):
#     '''Takes an epoch of pulsar data and subaverages every four frequency
#     channels
#
#     Pre-condition:
#     mjdi (float): the epoch mjd
#     data (numpy array): a 2D array of the epoch data overfrequency and time
#     freqsm (list): the 1D frequency array corresponding the channels within the data
#     plot (bool): if True, will plot the data in frequency and time and plot the
#     four highest frequency channels
#
#     Returns the subaveraged data (numpy array), the average frequencies for
#     this subaveraged data (list), and mjdi (float)'''
#
#     num_phase_bins = np.size(data[0])
#
#     if plot == True:
#         #plots the pulse over time and frequency
#         plt.imshow(data, aspect='26.0', origin='lower')
#         plt.ylabel('Frequency (MHz)')
#         plt.xlabel('Pulse Period (ms)')
#         plt.title('J1903+0327 Observation on MJD ' + str(mjdi)[:5])
#         xlabels_start = np.linspace(0, 2.15, 10)
#         xlabels = np.zeros(10)
#         for i in range(10):
#             xlabels[i] = str(xlabels_start[i])[:4]
#         ylabels_start = np.linspace(max(freqsm), min(freqsm), 10)
#         ylabels = np.zeros(10)
#         for i in range(10):
#             ylabels[i] = str(ylabels_start[i])[:4]
#         plt.xticks(ticks = np.linspace(0,num_phase_bins,10), labels = xlabels)
#         plt.yticks(ticks = np.linspace(0,len(freqsm),10), labels = ylabels)
#         plt.colorbar().set_label('Pulse Intensity')
#         plt.show()
#         #print("The number of subintegrations for this data file initially \
#         #      is" + str(ar.getNsubint()))
#
#     #see if the number of frequency channels is evenly divisible by 4
#     # if len(freqsm)%4 == 0:
#     #     subs = np.zeros((len(freqsm)//4,num_phase_bins))
#     #     center_freqs = np.zeros(len(freqsm)//4)
#     # else:
#     #     subs = np.zeros((len(freqsm)//4+1,num_phase_bins))
#     #     center_freqs = np.zeros((len(freqsm)//4)+1)
#
#     subs = np.zeros((len(freqsm)//4,num_phase_bins))
#     center_freqs = np.zeros(len(freqsm)//4)
#
#     #floor division for subintegrations all of 4 frequency channels
#     #also compute the average frequencies for each subintegration
#     for i in range(len(freqsm)//4):
#         datad = data[4*i:(4*i)+4]
#         dataf = freqsm[4*i:(4*i)+4]
#         subs[i] = np.average(datad, axis = 0)
#         center_freqs[i] = np.average(dataf)
#
#     #if number of frequency channels not divisible by 4
#     # if len(freqsm)%4 != 0:
#         #print('All subintegrations have 4 frequency channels except final\
#         #    subintegration has ' + str(len(freqsm)%4) + ' frequencie(s)')
#     #     data_d = data[len(freqsm)-(len(freqsm)%4):]
#     #     subs[-1] = np.average(data_d, axis = 0)
#     #     dataf = freqsm[len(freqsm)-(len(freqsm)%4):]
#     #     center_freqs[-1] = np.average(dataf)
#     #else:
#         #print('All subintegrations have 4 frequency channels')
#
#     #now subaveraging in time
#     subs_time_avg = np.zeros((len(freqsm)//4,num_phase_bins//8))
#
#     for i in range(len(freqsm)//4):
#         for ii in range(num_phase_bins//8):
#             subs_time_avg[i][ii] = np.average(subs[i][(8*ii):(8*ii)+8])
#
#     #plots the 4 highest frequency channels of the epoch, which are
#     #subaveraged for the highest frequency pulse
#     if plot == True:
#         fig, ax = plt.subplots(2,2)
#         fig.suptitle('MJD 57537 High Frequency Pulses')
#         fig.size = (16,24)
#         title = 'Pulse at Frequency ' + str(np.round(freqsm[0])) + 'MHz'
#         ax[0,0].plot(time, data[0])
#         ax[0,0].set_title(title)
#
#         title = 'Pulse at Frequency ' + str(np.round(freqsm[1])) + 'MHz'
#         ax[0,1].plot(time, data[1])
#         ax[0,1].set_title(title)
#
#         title = 'Pulse at Frequency ' + str(np.round(freqsm[2])) + 'MHz'
#         ax[1,0].plot(time, data[2])
#         ax[1,0].set_title(title)
#
#         title = 'Pulse at Frequency ' + str(np.round(freqsm[3])) + 'MHz'
#         ax[1,1].plot(time, data[3])
#         ax[1,1].set_title(title)
#
#         for ax1 in ax.flat:
#             ax1.set(xlabel='Pulse Phase (ms)', ylabel='Pulse Phase (ms)')
#
#         plt.tight_layout()
#         plt.show()
#
#         for i in range(np.size(center_freqs)):
#             plt.xlabel('Pulse Phase')
#             plt.ylabel('Pulse Intensity')
#             plt.title('Subintegration at ' + str(center_freqs[i]) + 'MHz')
#             plt.plot(subs_time_avg[i], '.')
#             plt.show()
#
#     #print the total number of subaverages
#     #print('Number of subaverages is ' + str(len(center_freqs)))
#
#     return(subs_time_avg, center_freqs, mjdi)
#
#
#
# class Profile:
#
#     num_phase_bins = 2048//8 #number of phase bins for pulse period in data (divided by 8 for averaging)
#     opr_size = 600//8 #number of phase bins for offpulse noise calculation
#
#     def __init__(self, mjd, data, frequencies, dur):
#         '''
#         mjd (float) - epoch of observation
#         data (2D array) - pulse data for epoch
#         frequencies (1D array) - frequencies corresponding to the data channels
#         dur (float) - observation duration in seconds
#         '''
#
#         #initialize the object attributes
#
#         self.mjd = mjd
#         self.mjd_round = np.round(mjd)
#         self.data_orig = data
#         self.freq_orig = frequencies
#         self.dur = dur
#
#
#         #subaverages the data for every four frequency channels
#
#         s = subaverages4(mjd, data, frequencies)
#         self.num_sub = len(s[1])
#         self.subaveraged_info = s
#
#
#     def fit_sing(self, profile, num_par):
#         '''Fits a data profile to a template
#         Helper function for all fitting functions below
#
#         Pre-conditions:
#         profile (numpy array): the template
#         num_par (int): the number of fitted parameters
#
#         Returns the fit chi-squared value (float)'''
#
#         #decide where to cut off noise depending on the frequency (matches with
#         #data as well)
#
#         num_masked = Profile.num_phase_bins - (self.stop_index-self.start_index)
#
#         profile = profile / np.max(profile) #fitPulse requires template height of one
#         z = np.max(profile)
#         zind = np.where(profile == z)[0][0]
#         ind_diff = self.xind-zind
#         #this lines the profiles up approximately so that Single Pulse finds the
#         #true minimum, not just a local min
#         profile = np.roll(profile, ind_diff)
#
#         sp = SinglePulse(self.data_suba, opw = np.arange(0, self.start_index))
#         fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template
#         #matching, scale factor, TOA error, scale factor error, signal to noise
#         #ratio, cross-correlation coefficient
#         #based on the fitPulse fitting, scale and shift the profile to best fit
#         #the inputted data
#         #fitPulse figures out the best amplitude itself
#         spt = SinglePulse(profile*fitting[2])
#         fitted_template = spt.shiftit(fitting[1])
#
#         chi_sq_measure = chi2_distance((self.data_suba*self.mask), (fitted_template*self.mask), self.rms_noise, num_par+num_masked)
#
#         return(chi_sq_measure)
#
#     def chi_plot(self, chi_sq_arr, beta = -1, exp = False, gwidth = -1, pbfwidth = -1):
#
#         '''Plots the inputted chi_sq_arr against the given parameters.
#
#         If gwidth is -1, indicates that the chi-squared surface exists over
#         all possible gaussian widths. Same for pbfwidth.
#
#         If beta is -1, must be for decaying exponential and exp != -1. Vice
#         versa.'''
#
#         plt.figure(45)
#
#         if gwidth == -1 and pbfwidth == -1:
#
#             plt.title("Fit Chi-sqs")
#             plt.xlabel("Gaussian FWHM (microseconds)")
#             plt.ylabel("PBF Width")
#
#             #adjust the imshow tick marks
#             gauss_ticks = np.zeros(10)
#             for ii in range(10):
#                 gauss_ticks[ii] = str(gauss_fwhm[ii*5])[:3]
#             pbf_ticks = np.zeros(10)
#             for ii in range(10):
#                 pbf_ticks[ii] = str(widths[ii*20])[:3]
#             plt.xticks(ticks = np.linspace(0,50,num=10), labels = gauss_ticks)
#             plt.yticks(ticks = np.linspace(0,200,num=10), labels = pbf_ticks)
#
#             plt.imshow(chi_sq_arr, cmap=plt.cm.viridis_r, origin = 'lower', aspect = 0.25)
#             plt.colorbar()
#
#             if beta != -1:
#                 title = f"TIMEA|ONEB|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}|BETA={beta}.png"
#
#             elif exp:
#                 title = f"TIMEA|EXP|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}.png"
#
#             plt.savefig(title)
#             plt.close(45)
#
#         elif gwidth != -1:
#
#             gwidth_round = str(gwidth)[:3]
#
#             plt.title('Fit Chi-sqs')
#             plt.xlabel('PBF Width')
#             plt.ylabel('Reduced Chi-Sq')
#             plt.plot(widths, chi_sq_arr)
#
#             if beta != -1:
#                 title = f"TIMEA|ONEBSETG|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}|BETA={beta}|GWIDTH={gwidth_round}.png"
#
#             elif exp:
#                 title = f"TIMEA|EXPSETG|PBF_fit_chisq|MJD={self.mjd_round}|FREQ={self.freq_round}|GWIDTH={gwidth_round}.png"
#
#             plt.savefig(title)
#             plt.close(45)
#
#
#     def fit_plot(self, beta_ind, pbfwidth_ind, gwidth_ind, exp = False):
#
#         '''Plots and saves the fit of the profile subaveraged data to the
#         template indicated by the argument indexes and the bolean
#         indicating if decaying exponential wanted for the broadening function.
#
#         beta_ind, pbfwidth_ind, gwidth_ind are ints; exp is boolean'''
#
#         if not exp:
#             i = convolved_profiles[beta_ind][pbfwidth_ind][gwidth_ind]
#         elif exp:
#             i = convolved_profiles_exp[pbfwidth_ind][gwidth_ind]
#
#         profile = i / np.max(i) #fitPulse requires template height of one
#         z = np.max(profile)
#         zind = np.where(profile == z)[0][0]
#         ind_diff = self.xind-zind
#         #this lines the profiles up approximately so that Single Pulse finds the
#         #true minimum, not just a local min
#         profile = np.roll(profile, ind_diff)
#         sp = SinglePulse(self.data_suba, opw = np.arange(0, self.start_index))
#         fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template
#         #matching, scale factor, TOA error, scale factor error, signal to noise
#         #ratio, cross-correlation coefficient
#         #based on the fitPulse fitting, scale and shift the profile to best fit
#         #the inputted data
#         #fitPulse figures out the best amplitude itself
#         spt = SinglePulse(profile*fitting[2])
#         fitted_template = spt.shiftit(fitting[1])
#
#         fitted_template = fitted_template*self.mask
#
#         plt.figure(50)
#         fig1 = plt.figure(50)
#         #Plot Data-model
#         frame1=fig1.add_axes((.1,.3,.8,.6))
#         #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
#         if not exp:
#             plt.title(f'Best Fit Template over Data with Beta = {betaselect[beta_ind]}')
#         elif exp:
#             plt.title('Best Fit Template over Data')
#         plt.ylabel('Pulse Intensity')
#         plt.plot(t, self.data_suba*self.mask, '.', ms = '2.4')
#         plt.plot(t, fitted_template)
#         frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
#         #above_noise = np.zeros(Profile.num_phase_bins)
#         #above_noise.fill(self.rms_noise)
#         #below_noise = np.zeros(Profile.num_phase_bins)
#         #below_noise.fill(-self.rms_noise)
#         #plt.plot(t[:Profile.opr_size], self.data_suba[:Profile.opr_size], '.', ms = '2.4', color = 'green')
#         #plt.plot(t, above_noise, color = 'blue')
#         #plt.plot(t, below_noise, color = 'blue')
#
#         #Residual plot
#         difference = np.subtract(self.data_suba*self.mask, fitted_template)
#         frame2=fig1.add_axes((.1,.1,.8,.2))
#         plt.plot(t, difference, '.', ms = '2.4')
#         plt.xlabel('Pulse Period (milliseconds)')
#         plt.ylabel('Residuals')
#
#         gwidth_round = str(gauss_fwhm[gwidth_ind])[:3]
#         pbfwidth_round = str(widths[pbfwidth_ind])[:3]
#
#         if not exp:
#             title = f'TIMEA|FIT|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}|BETA={betaselect[beta_ind]}|PBFW={pbfwidth_round}|GW={gwidth_round}.png'
#         elif exp:
#             title = f'TIMEA|FIT|EXP|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}|PBFW={pbfwidth_round}|GW={gwidth_round}.png'
#         plt.savefig(title)
#         plt.close(50)
#
#     def comp_fse(self, tau):
#         '''Calculates the error due to the finite scintile effect. Reference
#         Michael's email for details. Tau is the fitted time delay in microseconds.'''
#
#         T = self.dur
#         B = 12.5 #approximate frequency range for each channel in MHz
#         D = 0.64 #approximate distance to the pulsar in kpc
#         nt = 0.2 #filling factor over time
#         nv = 0.2 #filling factor over bandwidth
#
#         v = self.freq_suba / 1000.0
#         vd = (1.16)/(2.0*(math.pi)*tau) #MHz
#         #td = ((math.sqrt(D*(vd*1000.0)))/v)*(1338.62433862) #seconds
#         td = ((math.sqrt(D*(vd)))/v)*(1338.62433862) #seconds
#         nscint = (1.0 + nt*(T/td))*(1.0 + nv*(B/vd))
#         error = tau/(math.sqrt(nscint)) #microseconds
#         #print(error) -> seems to be a very small number of microseconds (order of 1)
#         return(error)
#
#     def fit(self, freq_subint_index, beta_ind = -1, gwidth_ind = -1, pbfwidth_ind = -1, dec_exp = False):
#         '''Calculates the best broadening function and corresponding parameters
#         for the Profile object.
#
#         beta_ind (int): if nonzero, set beta to this index of betaselect
#         gwidth_ind (int): if nonzero, set gauss width to this index of gauss_widths
#         pbf_width_ind (int) : if nonzero, set pbf width to this index of widths
#         dec_exp (bool) : if True, fit decaying exponential broadening functions
#
#         No error calculations for varying more than one parameter
#         '''
#
#         #number of each parameter in the parameter grid
#         num_beta = np.size(betaselect)
#         num_gwidth = np.size(gauss_widths)
#         num_pbfwidth = np.size(widths)
#
#         beta_inds = np.arange(num_beta)
#         gwidth_inds = np.arange(num_gwidth)
#         pbfwidth_inds = np.arange(num_pbfwidth)
#
#
#         #isolate the data profile at the frequency desired for this fit
#         self.data_suba = self.subaveraged_info[0][freq_subint_index]
#         self.freq_suba = self.subaveraged_info[1][freq_subint_index]
#         self.freq_round = np.round(self.freq_suba)
#
#
#         #Calculates mode of data profile to shift template to
#         x = np.max(self.data_suba)
#         self.xind = np.where(self.data_suba == x)[0][0]
#
#
#         #Set the offpulse regions to zero for fitting because essentially
#         #oscillating there.
#         #This region size varies depending on frequency
#
#         mask = np.zeros(Profile.num_phase_bins)
#
#         if self.freq_suba >= 1600:
#             self.start_index = 700
#             self.stop_index = 1548
#         elif self.freq_suba >= 1400 and self.freq_suba < 1600:
#             self.start_index = 700
#             self.stop_index = 1648
#         elif self.freq_suba >= 1200 and self.freq_suba < 1400:
#             self.start_index = 650
#             self.stop_index = 1798
#         elif self.freq_suba >= 1000 and self.freq_suba < 1200:
#             self.start_index = 600
#             self.stop_index = 1948
#
#         self.start_index = self.start_index//8
#         self.stop_index = self.stop_index//8
#         mask[self.start_index:self.stop_index] = 1.0
#
#         self.mask = mask
#
#
#         #Calculates the root mean square noise of the off pulse.
#         #Used later to calculate normalized chi-squared.
#
#         rms_collect = 0
#         for i in range(Profile.opr_size):
#             rms_collect += self.data_suba[i]**2
#         rms = math.sqrt(rms_collect/Profile.opr_size)
#
#         self.rms_noise = rms
#
#
#         #set more convenient names for the data to be fitted to
#         data_care = self.data_suba
#         freq_care = self.freq_suba
#
#
#         #case where beta and gaussian width are set, but fitting for pbf width
#         if beta_ind != -1 and gwidth_ind != -1 and dec_exp == False:
#
#             num_par = 3 # number of fitted parameters
#
#             beta = betaselect[beta_ind]
#             gwidth = gauss_fwhm[gwidth_ind]
#
#             chi_sqs_array = np.zeros(num_pbfwidth)
#             for i in pbfwidth_inds:
#
#                 template = convolved_profiles[beta_ind][i][gwidth_ind]
#                 chi_sq = self.fit_sing(template, num_par)
#                 chi_sqs_array[i] = chi_sq
#
#             self.chi_plot(chi_sqs_array, beta = beta, gwidth = gwidth)
#
#             low_chi = find_nearest(chi_sqs_array, 0.0)[0]
#             lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
#             pbf_width_fin = widths[lsqs_pbf_index]
#
#             #if chi_sqs_array[0] < low_chi+1 and chi_sqs_array[-1] < low_chi+1:
#             #    raise Exception('NOT CONVERGING ENOUGH')
#
#             tau_fin = tau_values[beta_ind][lsqs_pbf_index]
#
#             #ERROR TEST - one reduced chi-squared unit above and below and these
#             #chi-squared bins are for varying pbf width
#             below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+1)[1][0][0]
#             above = find_nearest(chi_sqs_array[lsqs_pbf_index:], low_chi+1)[1][0][0] + lsqs_pbf_index
#
#             tau_arr = tau_values[beta_ind]
#             tau_low = tau_fin - tau_arr[below]
#             tau_up = tau_arr[above] - tau_fin
#
#             #print(f'min chi is {low_chi}')
#             #print(f'chi next to low tau err is {chi_sqs_array[below-1]} and {chi_sqs_array[below+1]}')
#             #print(f'chi next to up tau err is {chi_sqs_array[above-1]} and {chi_sqs_array[above+1]}')
#
#
#             #plt.plot(tau_arr, chi_sqs_array)
#             #plt.plot(tau_fin-tau_low, chi_sqs_array[below], 'o', color = 'green')
#             #plt.plot(tau_fin+tau_up, chi_sqs_array[above], 'o', color = 'purple')
#             #plt.plot(tau_fin, low_chi, 'o', color = 'red')
#             #plt.plot(tau_arr, np.full((np.size(tau_values_exp)),low_chi), color = 'red')
#             #plt.plot(tau_arr, np.full((np.size(tau_values_exp)),low_chi+1), color = 'orange')
#             #plt.xlabel(r'$\tau$ ($\mu$s)')
#             #plt.ylabel('Reduced Chi-Squared')
#             #plt.show()
#
#             #print(f'Least Squares Tau: {tau_fin} microsec')
#             #print(f'Tau Lower Error: {tau_low} microsec')
#             #print(f'Tau Upper Error: {tau_up}  microsec')
#
#             self.fit_plot(beta_ind, lsqs_pbf_index, gwidth_ind)
#             #print(f'RMS Noise = {self.rms_noise}')
#
#             return(low_chi, tau_fin, tau_low, tau_up, self.comp_fse(tau_fin), gwidth, pbf_width_fin, beta)
#
#         #case where gaussian width is set and fitting amongst decaying exponential templates
#         elif dec_exp == True and gwidth_ind != -1:
#
#             num_par = 3 #number of fitted parameters
#
#             gwidth = gauss_fwhm[gwidth_ind]
#             chi_sqs_array = np.zeros(num_pbfwidth)
#
#             for i in pbfwidth_inds:
#
#                 template = convolved_profiles_exp[i][gwidth_ind]
#                 chi_sq = self.fit_sing(template, num_par)
#                 chi_sqs_array[i] = chi_sq
#
#             self.chi_plot(chi_sqs_array, exp = True, gwidth = gwidth)
#
#             low_chi = find_nearest(chi_sqs_array, 0.0)[0]
#             lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
#             pbf_width_fin = widths[lsqs_pbf_index]
#
#             #if chi_sqs_array[0] < low_chi+1 and chi_sqs_array[-1] < low_chi+1:
#             #    raise Exception('NOT CONVERGING ENOUGH')
#
#             tau_fin = tau_values_exp[lsqs_pbf_index]
#
#             #ERROR TEST - one reduced chi-squared unit above and below and these
#             #chi-squared bins are for varying pbf width
#
#             below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+1)[1][0][0]
#             above = find_nearest(chi_sqs_array[lsqs_pbf_index:], low_chi+1)[1][0][0] + lsqs_pbf_index
#
#             tau_low = tau_fin - tau_values_exp[below]
#             tau_up = tau_values_exp[above] - tau_fin
#
#             #print(f'min chi is {low_chi}')
#             #print(f'chi next to low tau err is {chi_sqs_array[below-1]} and {chi_sqs_array[below+1]}')
#             #print(f'chi next to up tau err is {chi_sqs_array[above-1]} and {chi_sqs_array[above+1]}')
#
#
#             #plt.plot(tau_values_exp, chi_sqs_array)
#             #plt.plot(tau_fin-tau_low, chi_sqs_array[below], 'o', color = 'green')
#             #plt.plot(tau_fin+tau_up, chi_sqs_array[above], 'o', color = 'purple')
#             #plt.plot(tau_fin, low_chi, 'o', color = 'black')
#             #plt.plot(tau_values_exp, np.full((np.size(tau_values_exp)),low_chi), color = 'red')
#             #plt.plot(tau_values_exp, np.full((np.size(tau_values_exp)),low_chi+1), color = 'orange')
#             #plt.xlabel(r'$\tau$ ($\mu$s)')
#             #plt.ylabel('Reduced Chi-Squared')
#             #plt.show()
#
#             #print(f'Least Squares Tau: {tau_fin} microsec')
#             #print(f'Tau Lower Error: {tau_low} microsec')
#             #print(f'Tau Upper Error: {tau_up}  microsec')
#
#             self.fit_plot(0, lsqs_pbf_index, gwidth_ind, exp = True)
#             #print(f'RMS Noise = {self.rms_noise}')
#
#             return(low_chi, tau_fin, tau_low, tau_up, self.comp_fse(tau_fin), gwidth, pbf_width_fin)
