"""
Created April 2023
Last Edited on Mon May 22 2023
@author: Abra Geiger abrageiger7

File of Functions for Fitting Broadening Functions
"""

#Functions contained: find_nearest(tau helper function), ecdf and pdf_to_cdf
#(likelihood helper functions), likelihood_evaluator, chi2_distance
#(helper function), subaverages4, fit_sing(fitting helper function),
#fit_all_profile (for fitting for best beta), fit_all_profile_set_gwidth
#(for fitting for best best for constant gauss width), fit_cons_beta_ipfd
#(for fitting with constant beta with no gaussian convolution, but estimated
#intrinsic pulse convolution), fit_cons_beta_profile (for fitting with constant beta),
#fit_dec_exp (for fitting decaying exponential)

#imports
from pypulse.archive import Archive
from pypulse.singlepulse import SinglePulse
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import CubicSpline
from scipy import special
import itertools
import convolved_pbfs as conv
import intrinsic_pbfs as intrins
import tau
import convolved_exp as cexp

#imports
convolved_profiles = conv.convolved_profiles
widths = conv.widths
gauss_widths = conv.widths_gaussian
widths_gaussian = conv.widths_gaussian
betaselect = conv.betaselect
time = conv.time
convolved_w_dataintrins = intrins.convolved_w_dataintrins
parameters = conv.parameters
convolved_profiles_exp = cexp.convolved_profiles_exp
tau_values_exp = tau.tau_values_exp
tau_values = tau.tau_values

phase_bins = 2048
t = np.linspace(0, phase_bins, phase_bins)
#gauss widths converted to fwhm microseconds
gauss_fwhm = gauss_widths * ((0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))

def find_nearest(a, a0):
    '''Element in nd array `a` closest to the scalar value `a0`

    Preconditions: a is an n dimensional array and a0 is a scalar value

    Returns index, value'''
    idx = np.abs(a - a0).argmin()
    return a.flat[idx], np.where((a == a.flat[idx]))

#Prof Lam's code for likelihood evaluator
#Creates empirical cdf
def ecdf(values, sort=True):
    if sort:
        values = np.sort(values)
    return values, np.linspace(0, 1, len(values))

EPS = special.erf(1.0/np.sqrt(2))/2.0

def pdf_to_cdf(pdf, dt=1):
    return np.cumsum(pdf)*dt

def likelihood_evaluator(x, y, cdf=False, median=False, pm=True, values=None):
    """
    cdf: if True, x,y describe the cdf
    median: if True, use the median value, otherwise the peak of the pdf (assuming cdf=False
    pm: xminus and xplus are the plus/minus range, not the actual values
    Future: give it values to grab off the CDF (e.g. 2 sigma, 99%, etc)
    values: use this array
    """
    if not cdf:
        y = y/np.trapz(y, x=x)
        ycdf = pdf_to_cdf(y, dt=(x[1]-x[0]))
    else: #else given a cdf
        ycdf = y

    if not values:
        if median:
            yb = 0.50   #Now take the median!
        else:
            indb = np.argmax(y)
            yb = ycdf[indb]
        ya = yb - EPS
        yc = yb + EPS
        yd = 0.95

        inda = np.argmin(np.abs(ycdf - ya))
        if median:
            indb = np.argmin(np.abs(ycdf - yb))
        indc = np.argmin(np.abs(ycdf - yc))
        indd = np.argmin(np.abs(ycdf - yd))

        inds = np.arange(inda, indc+1) #including indc
        #print indc-inda, np.trapz(L[inds], x=Vrs[inds])
        xval = x[indb]
        if pm:
            xminus = x[indb] - x[inda]
            xplus = x[indc] - x[indb]
        else:
            xminus = x[inda]
            xplus = x[indc]
        x95 = x[indd]

        return xval, xminus, xplus, x95
    else:
        retval = np.zeros_like(values)
        for i, v in enumerate(values):
            indv = np.argmin(np.abs(ycdf - v))
            retval[i] = x[indv]
        return retval

def chi2_distance(A, B, num_fitted = 0):

    '''Takes two vectors and calculates their comparative chi-squared value

    Pre-condition: A and B are 2 vectors of the same length and num_fitted (int) is
    the number of fitted parameters

    Returns chi-squared value'''

    squared_residuals = []
    for (a, b) in zip(A, B):
        sq_res = (a-b)**2
        squared_residuals.append(sq_res)

    chi_squared = sum(squared_residuals) / \
    (len(squared_residuals) - num_fitted)

    return(chi_squared)

def subaverages4(mjdi, data, freqsm, plot = False):
    '''Takes an epoch of pulsar data and subaverages every four frequency
    channels

    Pre-condition:
    mjdi (float): the epoch mjd
    data (numpy array): a 2D array of the epoch data overfrequency and time
    freqsm (list): the 1D frequency array corresponding the channels within the data
    plot (bool): if True, will plot the data in frequency and time and plot the
    four highest frequency channels

    Returns the subaveraged data (numpy array), the average frequencies for
    this subaveraged data (list), and mjdi (float)'''

    if plot == True:
        #plots the pulse over time and frequency
        plt.imshow(data, aspect='26.0', origin='lower')
        plt.ylabel('Frequency (MHz)')
        plt.xlabel('Pulse Period (ms)')
        plt.title('J1903+0327 Observation on MJD ' + str(mjdi)[:5])
        xlabels_start = np.linspace(0, 2.15, 10)
        xlabels = np.zeros(10)
        for i in range(10):
            xlabels[i] = str(xlabels_start[i])[:4]
        ylabels_start = np.linspace(max(freqsm), min(freqsm), 10)
        ylabels = np.zeros(10)
        for i in range(10):
            ylabels[i] = str(ylabels_start[i])[:4]
        plt.xticks(ticks = np.linspace(0,2048,10), labels = xlabels)
        plt.yticks(ticks = np.linspace(0,len(freqsm),10), labels = ylabels)
        plt.colorbar().set_label('Pulse Intensity')
        plt.show()
        #print("The number of subintegrations for this data file initially \
        #      is" + str(ar.getNsubint()))

    #see if the number of frequency channels is evenly divisible by 4
    if len(freqsm)%4 == 0:
        subs = np.zeros((len(freqsm)//4,2048))
        center_freqs = np.zeros(len(freqsm)//4)
    else:
        subs = np.zeros((len(freqsm)//4+1,2048))
        center_freqs = np.zeros((len(freqsm)//4)+1)

    #floor division for subintegrations all of 4 frequency channels
    #also compute the average frequencies for each subintegration
    for i in range(len(freqsm)//4):
        datad = data[4*i:(4*i)+4]
        dataf = freqsm[4*i:(4*i)+4]
        subs[i] = np.average(datad, axis = 0)
        center_freqs[i] = np.average(dataf)

    #if number of frequency channels not divisible by 4
    if len(freqsm)%4 != 0:
        #print('All subintegrations have 4 frequency channels except final\
        #    subintegration has ' + str(len(freqsm)%4) + ' frequencie(s)')
        data_d = data[len(freqsm)-(len(freqsm)%4):]
        subs[-1] = np.average(data_d, axis = 0)
        dataf = freqsm[len(freqsm)-(len(freqsm)%4):]
        center_freqs[-1] = np.average(dataf)
    #else:
        #print('All subintegrations have 4 frequency channels')

    #plots the 4 highest frequency channels of the epoch, which are
    #subaveraged for the highest frequency pulse
    if plot == True:
        fig, ax = plt.subplots(2,2)
        fig.suptitle('MJD 57537 High Frequency Pulses')
        fig.size = (16,24)
        title = 'Pulse at Frequency ' + str(np.round(freqsm[0])) + 'MHz'
        ax[0,0].plot(time, data[0])
        ax[0,0].set_title(title)

        title = 'Pulse at Frequency ' + str(np.round(freqsm[1])) + 'MHz'
        ax[0,1].plot(time, data[1])
        ax[0,1].set_title(title)

        title = 'Pulse at Frequency ' + str(np.round(freqsm[2])) + 'MHz'
        ax[1,0].plot(time, data[2])
        ax[1,0].set_title(title)

        title = 'Pulse at Frequency ' + str(np.round(freqsm[3])) + 'MHz'
        ax[1,1].plot(time, data[3])
        ax[1,1].set_title(title)

        for ax1 in ax.flat:
            ax1.set(xlabel='Pulse Phase (ms)', ylabel='Pulse Phase (ms)')

        plt.show()

        for i in range(np.size(center_freqs)):
            plt.xlabel('Pulse Phase')
            plt.ylabel('Pulse Intensity')
            plt.title('Subintegration at ' + str(center_freqs[i]) + 'MHz')
            plt.plot(subs[i])
            plt.show()

    #print the total number of subaverages
    print('Number of subaverages is ' + str(len(center_freqs)))

    return(subs, center_freqs, mjdi)

def fit_sing(i, xind, data_care, freqsy, num_fitted, plot = False, beta_ind = 0, gwidth_ind = 0, pbfwidth_ind = 0):
    '''Fits a data profile to a template
    Helper function for all fitting functions below

    Pre-conditions:
    i (numpy array): the template
    xind (int): the mode location to shift the profile to in phase phase_bins
    data_care (numpy array): the data profile
    freqsy (float): the frequency of observation for data_care
    num_fitted (int): the number of fitted parameters

    Returns the fit chi-squared value (float)'''

    #decide where to cut off noise depending on the frequency (matches with
    #data as well)

    mask = np.zeros(2048)
    if freqsy >= 1600:
        start_index = 700
        stop_index = 1548
        num_masked = phase_bins - (stop_index-start_index)
    elif freqsy >= 1400 and freqsy < 1600:
        start_index = 700
        stop_index = 1648
        num_masked = phase_bins - (stop_index-start_index)
    elif freqsy >= 1200 and freqsy < 1400:
        start_index = 650
        stop_index = 1798
        num_masked = phase_bins - (stop_index-start_index)
    elif freqsy >= 1000 and freqsy < 1200:
        start_index = 600
        stop_index = 1948
        num_masked = phase_bins - (stop_index-start_index)
    mask[start_index:stop_index] = 1.0

    profile = i / np.max(i) #fitPulse requires template height of one
    z = np.max(profile)
    zind = np.where(profile == z)[0][0]
    ind_diff = xind-zind
    #this lines the profiles up approximately so that Single Pulse finds the
    #true minimum, not just a local min
    profile = np.roll(profile, ind_diff)
    sp = SinglePulse(data_care, opw = np.arange(0, start_index))
    fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template
    #matching, scale factor, TOA error, scale factor error, signal to noise
    #ratio, cross-correlation coefficient
    #based on the fitPulse fitting, scale and shift the profile to best fit
    #the inputted data
    #fitPulse figures out the best amplitude itself
    spt = SinglePulse(profile*fitting[2])
    fitted_template = spt.shiftit(fitting[1])

    chi_sq_measure = chi2_distance((data_care*mask), (fitted_template*mask), num_fitted+num_masked)

    return(chi_sq_measure)


class Profile:

    num_phase_bins = 2048 #number of phase bins for pulse period in data
    opr_size = 600 #number of phase bins for offpulse noise calculation

    def __init__(self, mjd, data, frequencies, freq_subint_index, dur):
        '''
        mjd (float) - epoch of observation
        data (2D array) - pulse data for epoch
        frequencies (1D array) - frequencies corresponding to the data channels
        freq_subint_index (int) - index subintegrated data focused on
        '''

        self.mjd = mjd
        self.mjd_round = np.round(mjd)
        self.data_orig = data
        self.freq_orig = frequencies
        self.dur = dur

        #subaverages the data for every four frequency channels

        s = subaverages4(mjd, data, frequencies)
        self.num_sub = len(s[1])
        self.data_suba = s[0][freq_subint_index]
        self.freq_suba = s[1][freq_subint_index]
        self.freq_round = np.round(self.freq_suba)

        #Calculates the root mean square noise of the off pulse.
        #Used later to calculate normalized chi-squared.

        rms_collect = 0
        for i in range(Profile.opr_size):
            rms_collect += self.data_suba[i]**2
        rms = math.sqrt(rms_collect/Profile.opr_size)

        self.rms_noise = rms

        #mode of data profile to shift template to
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

        self.data_forfit = self.data_suba*mask

    def chi_plot(self, chi_sq_arr, beta = -1, exp = False, gwidth = -1, pbfwidth = -1):

        '''Plots the inputted chi_sq_arr against the given parameters.

        If gwidth is -1, indicates that the chi-squared surface exists over
        all possible gaussian widths. Same for pbfwidth.

        If bets is -1, must be for decaying exponential and exp != -1. Vice
        versa.'''

        plt.figure(45)

        if gwidth == -1 and pbfwidth == -1:

            plt.title("Fit Chi-sqs")
            plt.xlabel("Gaussian FWHM (microseconds)")
            plt.ylabel("PBF Width")

            #adjust the imshow tick marks
            gauss_ticks = np.zeros(10)
            for ii in range(10):
                gauss_ticks[ii] = str(gauss_fwhm[ii*20])[:3]
            pbf_ticks = np.zeros(10)
            for ii in range(10):
                pbf_ticks[ii] = str(widths[ii*20])[:3]
            plt.xticks(ticks = np.linspace(0,200,num=10), labels = gauss_ticks)
            plt.yticks(ticks = np.linspace(0,200,num=10), labels = pbf_ticks)

            plt.imshow(chi_sq_arr, cmap=plt.cm.viridis_r, origin = 'lower')
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

            plt.savefig(title)
            plt.close(45)


    def fit_plot(self, beta_ind, pbfwidth_ind, gwidth_ind, exp = False):

        if not exp:
            i = convolved_profiles[beta_ind][pbfwidth_ind][gwidth_ind]
        elif exp:
            i = convolved_profiles_exp[pbfwidth_ind][gwidth_ind]

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
        if not exp:
            plt.title(f'Best Fit Template over Data with Beta = {betaselect[beta_ind]}')
        elif exp:
            plt.title('Best Fit Template over Data')
        plt.ylabel('Pulse Intensity')
        plt.plot(time, self.data_forfit, '.', ms = '2.4')
        plt.plot(time, fitted_template)
        frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
        plt.plot()

        #Residual plot
        difference = np.subtract(self.data_forfit, fitted_template)
        frame2=fig1.add_axes((.1,.1,.8,.2))
        plt.plot(time, difference, '.', ms = '2.4')
        plt.xlabel('Pulse Period (milliseconds)')
        plt.ylabel('Residuals')
        plt.plot()

        gwidth_round = str(gauss_fwhm[gwidth_ind])[:3]
        pbfwidth_round = str(widths[pbfwidth_ind])[:3]

        if not exp:
            title = f'FIT|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}|BETA={betaselect[beta_ind]}|PBFW={pbfwidth_round}|GW={gwidth_round}.png'
        elif exp:
            title = f'FIT|EXP|PBF_fit_plot|MJD={self.mjd_round}|FREQ={self.freq_round}|PBFW={pbfwidth_round}|GW={gwidth_round}.png'
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
        error1 = tau/(math.sqrt(nscint)) #microseconds
        return(error)

    def fit(self, beta_ind = -1, gwidth_ind = -1, pbfwidth_ind = -1, dec_exp = False):
        '''Calculates the best broadening function and corresponding parameters
        for the Profile object.

        beta_ind (int): if nonzero, set beta to this index of betaselect
        gwidth_ind (int): if nonzero, set gauss width to this index of gauss_widths
        pbf_width_ind (int) : if nonzero, set pbf width to this index of widths
        dec_exp (bool) : if True, fit decaying exponential broadening functions
        '''

        #number of each parameter in the parameter grid
        num_beta = np.size(betaselect)
        num_gwidth = np.size(gauss_widths)
        num_pbfwidth = np.size(widths)

        beta_inds = np.arange(num_beta)
        gwidth_inds = np.arange(num_gwidth)
        pbfwidth_inds = np.arange(num_pbfwidth)

        data_care = self.data_suba
        freq_care = self.freq_suba

        if beta_ind == -1 & gwidth_ind == -1 and dec_exp == False:

            num_par = 5 #number of fitted parameters

            chi_sqs = np.zeros((num_beta, num_pbfwidth, num_gwidth))
            for i in itertools.product(beta_inds, pbfwidth_inds, gwidth_inds):

                template = convolved_profiles[i[0]][i[1]][i[2]]
                chi_sq = fit_sing(template, selfxind, data_care, freq_care, num_par)
                chi_sqs[i[0]][i[1]][i[2]] = chi_sq

            chi_sqs_collect = np.zeros(num_beta)
            pbf_width_collect = np.zeros(num_beta)
            gaussian_width_collect = np.zeros(num_beta)
            taus_collect = np.zeros(num_beta)
            taus_err_collect = np.zeros((2,num_beta))
            ind = 0
            for i in chi_sqs:

                beta = betaselect[ind]

                #scale the chi-squared array by the rms value of the profile
                chi_sqs_array = np.divide(i,(self.rms_noise**2))
                self.chi_plot(chi_sqs_array, beta = beta)

                #least squares
                low_chi = find_nearest(chi_sqs_array, 0.0)[0]
                chi_sqs_collect[ind] = low_chi

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
            tau_err_fin = taus_err_collect[:,chi_beta_ind]

            pbf_width_ind = np.where(widths == pbf_width_fin)[0][0]
            gauss_width_ind = np.where(((gauss_widths  * (2.0*math.sqrt(2*math.log(2))) * (0.0021499/2048) * 1e6) == gaussian_width_collect[chi_beta_ind]))[0][0]

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

            return(low_chi, tau_fin, tau_err_fin, self.comp_fse(tau_fin), gauss_width_fin, pbf_width_fin, beta_fin)


        elif beta_ind != -1 and gwidth_ind != -1 and dec_exp == False:

            num_par = 3 # number of fitted parameters

            beta = betaselect[beta_ind]
            gwidth = gauss_fwhm[gwidth_ind]

            chi_sqs = np.zeros(num_pbfwidth)
            for i in pbfwidth_inds:

                template = convolved_profiles[beta_ind][i][gwidth_ind]
                chi_sq = fit_sing(template, self.xind, data_care, freq_care, num_par)
                chi_sqs[i] = chi_sq

            chi_sqs_array = np.divide(chi_sqs,(self.rms_noise**2))
            self.chi_plot(chi_sqs_array, beta = beta, gwidth = gwidth)

            low_chi = find_nearest(chi_sqs_array, 0.0)[0]
            lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
            lsqs_pbf_val = widths[lsqs_pbf_index]

            tau_fin = tau_values[beta_ind][lsqs_pbf_index]
            pbf_width_fin = lsqs_pbf_val

            #ERROR TEST - one reduced chi-squared unit above and below and these
            #chi-squared bins are for varying pbf width
            below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+1)[1][0][0]
            above = find_nearest(chi_sqs_array[lsqs_pbf_index:], low_chi+1)[1][0][0] + lsqs_pbf_index

            tau_arr = tau_values[beta_ind]
            tau_low = tau_fin - tau_arr[below]
            tau_up = tau_arr[above] - tau_fin

            self.fit_plot(beta_ind, lsqs_pbf_index, gwidth_ind)

            return(low_chi, tau_fin, tau_low, tau_up, self.comp_fse(tau_fin), gwidth, pbf_width_fin, beta)


        elif dec_exp == True and gwidth_ind != -1:

            num_par = 3 #number of fitted parameters

            gwidth = gauss_fwhm[gwidth_ind]
            chi_sqs = np.zeros(num_pbfwidth)

            for i in pbfwidth_inds:

                template = convolved_profiles_exp[i][gwidth_ind]
                chi_sq = fit_sing(template, self.xind, data_care, freq_care, num_par)
                chi_sqs[i] = chi_sq

            chi_sqs_array = np.divide(chi_sqs,(self.rms_noise**2))
            self.chi_plot(chi_sqs_array, exp = True, gwidth = gwidth)

            low_chi = find_nearest(chi_sqs_array, 0.0)[0]
            lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
            lsqs_pbf_val = widths[lsqs_pbf_index]

            tau_fin = tau_values_exp[lsqs_pbf_index]
            pbf_width_fin = lsqs_pbf_val

            #ERROR TEST - one reduced chi-squared unit above and below and these
            #chi-squared bins are for varying pbf width
            below = find_nearest(chi_sqs_array[:lsqs_pbf_index], low_chi+1)[1][0][0]
            above = find_nearest(chi_sqs_array[lsqs_pbf_index:], low_chi+1)[1][0][0] + lsqs_pbf_index

            tau_low = tau_fin - tau_values_exp[below]
            tau_up = tau_values_exp[above] - tau_fin

            self.fit_plot(0, lsqs_pbf_index, gwidth_ind, exp = True)

            return(low_chi, tau_fin, tau_low, tau_up, self.comp_fse(tau_fin), gwidth, pbf_width_fin)



def fit_all_profile(mjdi, data, freqsm, freq_subint_index):

    '''For a given data profile, finds the best template across varying beta,
    pbf_width, and gaussian width'''

    s = subaverages4(mjdi, data, freqsm)

    data_care = s[0][freq_subint_index]

    #calculate the root mean square noise of the off pulse in order to
    #normalize chi-squared
    rms_collect = 0
    for i in range(600):
        rms_collect += data_care[i]*data_care[i]
    rms = math.sqrt(rms_collect/600)

    freqs_care = s[1][freq_subint_index]

    x = np.max(data_care)
    xind = np.where(data_care == x)[0][0]

    #set the offpulse regions to zero because essentially oscillating there
    #vary the offpulse region depending on frequency
    mask = np.zeros(2048)
    if freqs_care >= 1600:
        start_index = 700
        stop_index = 1548
    elif freqs_care >= 1400 and freqs_care < 1600:
        start_index = 700
        stop_index = 1648
    elif freqs_care >= 1200 and freqs_care < 1400:
        start_index = 650
        stop_index = 1798
    elif freqs_care >= 1000 and freqs_care < 1200:
        start_index = 600
        stop_index = 1948
    mask[start_index:stop_index] = 1.0

    data_care = data_care*mask

    chi_sqs_collect = np.zeros(np.size(betaselect))
    pbf_width_collect = np.zeros(np.size(betaselect))
    gaussian_width_collect = np.zeros(np.size(betaselect))
    taus_collect = np.zeros(np.size(betaselect))

    for beta_index in range(12):

        print("Fitting for Beta = " + str(betaselect[beta_index]))
        chi_sqs_array = np.zeros((np.size(widths), np.size(gauss_widths)))

        data_index1 = 0
        #for the varying pbf_widths
        for ii in convolved_profiles[beta_index]:
            data_index2 = 0
            #for the varying gaussian widths
            for i in ii:
                chi_sqs_array[data_index1][data_index2] = fit_sing(i, xind, data_care, freqs_care, 5)
                data_index2 = data_index2+1
            data_index1 = data_index1+1

        #print(betaselect[beta_index])
        plt.figure(1*beta_index)
        chisqs = chi_sqs_array - np.amin(chi_sqs_array)
        chisqs = np.exp((-0.5)*chisqs)
        plt.title('Fit Chi-sqs')
        plt.xlabel('Rounded Gaussian FWHM (microseconds)')
        plt.ylabel('PBF Width')
        #scale the chi-squared array by the rms value of the profile
        chi_sqs_array = np.divide(chi_sqs_array,(rms*rms))
        plt.imshow(chi_sqs_array, cmap=plt.cm.viridis_r, origin = 'lower')
        gauss_ticks = np.zeros(10)
        for i in range(10):
            gauss_ticks[i] = str(int(gauss_widths[i*5] * (0.0021499/2048) * 1e6 \
            * (2.0*math.sqrt(2*math.log(2))))) #convert to FWHM microseconds
        pbf_ticks = np.zeros(10)
        for i in range(10):
            pbf_ticks[i] = str(widths[i*5])[:3]
        plt.xticks(ticks = np.linspace(0,50,num=10), labels = gauss_ticks)
        plt.yticks(ticks = np.linspace(0,50,num=10), labels = pbf_ticks)
        plt.colorbar()
        title = 'PBF_fit_chisq_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + \
        str(freqs_care)[:4] + '_BETA=' + str(betaselect[beta_index]) + '.png'
        plt.savefig(title)
        plt.close(1*beta_index)
        low_chi = find_nearest(chi_sqs_array, 0.0)[0]
        #print("Minimum chi-squared: " + str(low_chi))

        chi_sqs_collect[beta_index] = low_chi

        lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
        lsqs_pbf_val = widths[lsqs_pbf_index]
        lsqs_gauss_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
        lsqs_gauss_val = widths_gaussian[lsqs_gauss_index] * \
        (2.0*math.sqrt(2*math.log(2))) * (0.0021499/2048) * 1e6 #microseconds FWHM
        gaussian_width_collect[beta_index] = lsqs_gauss_val
        pbf_width_collect[beta_index] = lsqs_pbf_val

        #probabilitiesx = np.sum(chisqs, axis=1)
        #p_pbfwidth = np.where(probabilitiesx == np.max(probabilitiesx))[0]
        #probabilitiesy = np.sum(chisqs, axis=0)
        #p_gausswidth = np.where(probabilitiesy == np.max(probabilitiesy))[0]
        #likelihoodx = likelihood_evaluator(widths, probabilitiesx)
        #likelihoody = likelihood_evaluator(widths_gaussian, probabilitiesy)

        #b_pbfwidth_index = find_nearest(widths, likelihoodx[0])[1][0][0]
        #b_gausswidth_index = find_nearest(widths_gaussian, likelihoody[0])[1][0][0]

        #plt.title('PBF Width Likelihood')\n",
        #plt.xlabel('PBF Width')\n",
        #plt.ylabel('Likelihood')\n",
        #plt.plot(widths, probabilitiesx)\n",
        #plt.show()\n",
        #print('Likelihood:')\n",
        #print(\"Best fit PBF width: \"+ str(likelihoodx[0]))\n",
        #print(\"Error: minus \" + str(likelihoodx[1]) + \" plus \" + str(likelihoodx[2]))\n",
        #print('Lowest Chi-sqs:')\n",
        #print(\"Best fit PBF width: \"+ str(lsqs_pbf_val))\n",
        #plt.title('Gaussian Width Likelihood')\n",
        #plt.xlabel('Gaussian Width')\n",
        #plt.ylabel('Likelihood')\n",
        #plt.plot(widths_gaussian, probabilitiesy)\n",
        #plt.show()\n",
        #print('Likelihood:')\n",
        #print(\"Best fit Gaussian width: \"+ str(likelihoody[0]) + ' microseconds')\n",
        #print(\"Error: minus \" + str(likelihoody[1]) + \" microseconds plus \" + str(likelihoody[2]) + ' microseconds')\n",
        # print('Lowest Chi-sqs:')
        # print("Best fit Gaussian width: "+ str(lsqs_gauss_val))
        # print('Best fit tau value: ' + str(tau.tau_values[beta_index][b_pbfwidth_index]) + ' microseconds')
        taus_collect[beta_index] = tau.tau_values[beta_index][lsqs_pbf_index]

        #plt.title('Tau Values')\n",
        #plt.xlabel('Gaussian Width (milliseconds)')\n",
        #plt.ylabel('PBF Width')\n",
        #plt.imshow(taus_collect, cmap=plt.cm.viridis_r)\n",
        #plt.colorbar()\n",
        #plt.show()\n",

         #plt.title('Likelihood: Best Fit Template over Data')\n",
         #plt.ylabel('Pulse Intensity')\n",
         #plt.xlabel('Pulse Period (milliseconds)')\n",
         #plt.plot(time, data_care, '.')\n",
         #plt.plot(time, fitted_templates[b_pbfwidth_index][b_gausswidth_index])\n",
         #plt.show()\n",

         #plt.title('Least Squares: Best Fit Template over Data')\n",
         #plt.ylabel('Pulse Intensity')\n",
         #plt.xlabel('Pulse Period (milliseconds)')\n",
         #plt.plot(time, data_care, '.')\n",
         #plt.plot(time, fitted_templates[lsqs_pbf_index][lsqs_gauss_index])\n",
         #plt.show()\n",

        #fig1 = plt.figure(1)\n",
        ##Plot Data-model\n",
        #frame1=fig1.add_axes((.1,.3,.8,.6))\n",
        ##xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]\n",
        #plt.title('Likelihood: Best Fit Template over Data')\n",
        #plt.ylabel('Pulse Intensity')\n",
        #plt.plot(time, data_care, '.')\n",
        #plt.plot(time, fitted_templates[b_pbfwidth_index][b_gausswidth_index]) \n",
        #frame1.set_xticklabels([]) #Remove x-tic labels for the first frame\n",
        #plt.plot()\n",

        ##Residual plot\n",
        #difference = np.subtract(data_care, fitted_templates[b_pbfwidth_index][b_gausswidth_index])\n",
        #frame2=fig1.add_axes((.1,.1,.8,.2))        \n",
        #plt.plot(time, difference, '.')\n",
        #plt.xlabel('Pulse Period (milliseconds)')\n",
        #plt.ylabel('Residuals')\n",
        #plt.plot()\n",

        #Best Fit Template fitting
        profile = convolved_profiles[beta_index][lsqs_pbf_index][lsqs_gauss_index] / np.max(convolved_profiles[beta_index][lsqs_pbf_index][lsqs_gauss_index])
        #fitPulse requires template height of one
        z = np.max(profile)
        zind = np.where(profile == z)[0][0]
        ind_diff = xind-zind
        #this lines the profiles up approximately so that Single Pulse finds the true minimum, not just a local min\n",
        profile = np.roll(profile, ind_diff)
        sp = SinglePulse(data_care, opw = np.arange(0, 800))
        fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template matching, scale factor, TOA error, \n",
        #scale factor error, signal to noise ratio, cross-correlation coefficient\n",
        #based on the fitPulse fitting, scale and shift the profile to best fit the inputted data\n",
        #fitPulse figures out the best amplitude itself
        spt = SinglePulse(profile*fitting[2])
        fitted_template = spt.shiftit(fitting[1])
        fitted_template = fitted_template*mask

        plt.figure(2*beta_index)
        fig1 = plt.figure(2*beta_index)
        #Plot Data-model
        frame1=fig1.add_axes((.1,.3,.8,.6))
        #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
        plt.title('Best Fit Template over Data with Beta = ' + str(betaselect[beta_index]))
        plt.ylabel('Pulse Intensity')
        plt.plot(time, data_care, '.', ms = '2.4')
        plt.plot(time, fitted_template)
        frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
        plt.plot()

        #Residual plot
        difference = np.subtract(data_care, fitted_template)
        frame2=fig1.add_axes((.1,.1,.8,.2))
        plt.plot(time, difference, '.', ms = '2.4')
        plt.xlabel('Pulse Period (milliseconds)')
        plt.ylabel('Residuals')
        plt.plot()

        title = 'PBF_fit_plot_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_BETA=' + str(betaselect[beta_index]) + '.png'
        plt.savefig(title)
        plt.close(2*beta_index)

    plt.figure(25)
    plt.xlabel('Beta')
    plt.ylabel('Chi-Squared')
    plt.plot(betaselect, chi_sqs_collect)
    title = 'PBF_fit_overall_chisqs_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_bestBETA=' + str(betaselect[beta_index]) + '.png'
    plt.savefig(title)
    plt.close(25)
    plt.figure(26)
    plt.xlabel('Beta')
    plt.ylabel('Overall Best PBF Width')
    plt.plot(betaselect, pbf_width_collect)
    title = 'PBF_fit_overall_widths_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_bestBETA=' + str(betaselect[beta_index]) + '.png'
    plt.savefig(title)
    plt.close(26)
    plt.figure(27)
    plt.xlabel('Beta')
    plt.ylabel('Overall Best Gaussian Width FWHM (milliseconds)')
    plt.errorbar(betaselect, gaussian_width_collect) #already converted to micro fwhm
    title = 'PBF_fit_overall_gauss_widths_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_bestBETA=' + str(betaselect[beta_index]) + '.png'
    plt.savefig(title)
    plt.close(27)
    plt.figure(28)
    plt.xlabel('Beta')
    plt.ylabel('Overall Best Tau (microseconds)')
    plt.errorbar(betaselect, taus_collect)
    title = 'PBF_fit_overall_tau_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_bestBETA=' + str(betaselect[beta_index]) + '.png'
    plt.savefig(title)
    plt.close(28)

    low_chi = np.min(chi_sqs_collect)
    chi_beta_ind = np.where(chi_sqs_collect == low_chi)[0][0]
    beta_fin = betaselect[chi_beta_ind]
    pbf_width_fin = pbf_width_collect[chi_beta_ind]
    gauss_width_fin = gaussian_width_collect[chi_beta_ind]
    tau_fin = taus_collect[chi_beta_ind]

    pbf_width_ind = np.where(widths == pbf_width_fin)[0][0]
    gauss_width_ind = np.where(((gauss_widths  * (2.0*math.sqrt(2*math.log(2))) * (0.0021499/2048) * 1e6) == gaussian_width_collect[chi_beta_ind]))[0][0]

    #now plot the final best template over the data
    profile = convolved_profiles[chi_beta_ind][pbf_width_ind][gauss_width_ind] / np.max(convolved_profiles[chi_beta_ind][pbf_width_ind][gauss_width_ind])
    #fitPulse requires template height of one
    z = np.max(profile)
    zind = np.where(profile == z)[0][0]
    ind_diff = xind-zind
    #this lines the profiles up approximately so that Single Pulse finds the true minimum, not just a local min\n",
    profile = np.roll(profile, ind_diff)
    sp = SinglePulse(data_care, opw = np.arange(0, 800))
    fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template matching, scale factor, TOA error, \n",
    #scale factor error, signal to noise ratio, cross-correlation coefficient\n",
    #based on the fitPulse fitting, scale and shift the profile to best fit the inputted data\n",
    #fitPulse figures out the best amplitude itself\n",
    #fitted_shift[data_index1][data_index2] = fitting[1]\n"
    #fitted_scale_factor[data_index1][data_index2] = fitting[2]
    spt = SinglePulse(profile*fitting[2])
    fitted_template = spt.shiftit(fitting[1])
    #fitted_templates[data_index1][data_index2] = fitted_template
    fitted_template = fitted_template*mask

    plt.figure(29)
    fig1 = plt.figure(29)
    #Plot Data-model\n",
    frame1=fig1.add_axes((.1,.3,.8,.6))
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.title('Best Fit Template over Data with Beta = ' + str(betaselect[beta_index]) + '; chisq = ' + str(low_chi))[:5]
    plt.ylabel('Pulse Intensity')
    plt.plot(time, data_care, '.', ms = '2.4')
    plt.plot(time, fitted_template)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.show()

    #Residual plot
    difference = np.subtract(data_care, fitted_template)
    frame2=fig1.add_axes((.1,.1,.8,.2))
    plt.plot(time, difference, '.', ms = '2.4')
    plt.xlabel('Pulse Period (milliseconds)')
    plt.ylabel('Residuals')
    plt.show()

    title = 'PBF_fit_overall_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_GAUSSW=' + str(gauss_width_fin)[:4] + '_PBFW=' + str(pbf_width_fin) + '.png'
    plt.savefig(title)
    plt.close(29)
    print(title)

    print('Min Chi-sq = ' + str(low_chi) + '\n'+'Best tau = ' + str(tau_fin) \
          + '\n'+'Best Gauss Width = ' + str(gauss_width_fin) + '\n'+'Best PBF Width = ' \
              + str(pbf_width_fin) + '\n'+'Best Fit Beta = ' + str(beta_fin) + '\n'+'Frequency = '\
                  + str(freqs_care))
    return(low_chi, tau_fin, gauss_width_fin, pbf_width_fin, beta_fin, freqs_care)

def fit_all_profile_set_gwidth(mjdi, data, freqsm, freq_subint_index, gwidth_index):

    s = subaverages4(mjdi, data, freqsm)

    data_care = s[0][freq_subint_index]

    #calculate the root mean square noise of the off pulse in order to
    #normalize chi-squared
    rms_collect = 0
    for i in range(600):
        rms_collect += data_care[i]*data_care[i]
    rms = math.sqrt(rms_collect/600)

    freqs_care = s[1][freq_subint_index]

    x = np.max(data_care)
    xind = np.where(data_care == x)[0][0]

    #set the offpulse regions to zero because essentially oscillating there
    #vary the offpulse region depending on frequency
    mask = np.zeros(2048)
    if freqs_care >= 1600:
        start_index = 700
        stop_index = 1548
    elif freqs_care >= 1400 and freqs_care < 1600:
        start_index = 700
        stop_index = 1648
    elif freqs_care >= 1200 and freqs_care < 1400:
        start_index = 650
        stop_index = 1798
    elif freqs_care >= 1000 and freqs_care < 1200:
        start_index = 600
        stop_index = 1948
    mask[start_index:stop_index] = 1.0

    data_care = data_care*mask

    chi_sqs_collect = np.zeros(np.size(betaselect))
    pbf_width_collect = np.zeros(np.size(betaselect))
    taus_collect = np.zeros(np.size(betaselect))

    for beta_index in range(12):

        print("Fitting for Beta = " + str(betaselect[beta_index]))
        chi_sqs_array = np.zeros(np.size(widths))

        data_index1 = 0
        #for the varying pbf_widths
        for ii in convolved_profiles[beta_index]:
            #for the gaussian width
            i = ii[gwidth_index]
            chi_sqs_array[data_index1] = fit_sing(i, xind, data_care, freqs_care, 4)
            data_index1 = data_index1+1

        #print(betaselect[beta_index])
        plt.figure(1*beta_index)
        chisqs = chi_sqs_array - np.amin(chi_sqs_array)
        chisqs = np.exp((-0.5)*chisqs)
        plt.title('Fit Chi-sqs for Constant Gaussian FWHM = ' + str(gauss_widths[gwidth_index] * (0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:5])
        plt.ylabel('Chisq')
        plt.xlabel('PBF Width')
        #scale the chi-squared array by the rms value of the profile
        chi_sqs_array = np.divide(chi_sqs_array,(rms*rms))
        plt.plot(widths, chi_sqs_array)
        title = 'SETGwidth='+str(gauss_widths[gwidth_index] * (0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4]+'_PBF_fit_chisq_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_BETA=' + str(betaselect[beta_index]) + '.png'
        plt.savefig(title)
        plt.close(1*beta_index)
        low_chi = find_nearest(chi_sqs_array, 0.0)[0]

        chi_sqs_collect[beta_index] = low_chi

        lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
        lsqs_pbf_val = widths[lsqs_pbf_index]
        pbf_width_collect[beta_index] = lsqs_pbf_val

        #probabilitiesx = np.sum(chisqs, axis=1)
        #p_pbfwidth = np.where(probabilitiesx == np.max(probabilitiesx))[0]
        #probabilitiesy = np.sum(chisqs, axis=0)
        #p_gausswidth = np.where(probabilitiesy == np.max(probabilitiesy))[0]
        #likelihoodx = likelihood_evaluator(widths, probabilitiesx)
        #likelihoody = likelihood_evaluator(widths_gaussian, probabilitiesy)

        #b_pbfwidth_index = find_nearest(widths, likelihoodx[0])[1][0][0]
        #b_gausswidth_index = find_nearest(widths_gaussian, likelihoody[0])[1][0][0]

        #plt.title('PBF Width Likelihood')\n",
        #plt.xlabel('PBF Width')\n",
        #plt.ylabel('Likelihood')\n",
        #plt.plot(widths, probabilitiesx)\n",
        #plt.show()\n",
        #print('Likelihood:')\n",
        #print(\"Best fit PBF width: \"+ str(likelihoodx[0]))\n",
        #print(\"Error: minus \" + str(likelihoodx[1]) + \" plus \" + str(likelihoodx[2]))\n",
        #print('Lowest Chi-sqs:')\n",
        #print(\"Best fit PBF width: \"+ str(lsqs_pbf_val))\n",
        #plt.title('Gaussian Width Likelihood')\n",
        #plt.xlabel('Gaussian Width')\n",
        #plt.ylabel('Likelihood')\n",
        #plt.plot(widths_gaussian, probabilitiesy)\n",
        #plt.show()\n",
        #print('Likelihood:')\n",
        #print(\"Best fit Gaussian width: \"+ str(likelihoody[0]) + ' microseconds')\n",
        #print(\"Error: minus \" + str(likelihoody[1]) + \" microseconds plus \" + str(likelihoody[2]) + ' microseconds')\n",
        # print('Lowest Chi-sqs:')
        # print("Best fit Gaussian width: "+ str(lsqs_gauss_val))
        # print('Best fit tau value: ' + str(tau.tau_values[beta_index][b_pbfwidth_index]) + ' microseconds')
        taus_collect[beta_index] = tau.tau_values[beta_index][lsqs_pbf_index]

        #plt.title('Tau Values')\n",
        #plt.xlabel('Gaussian Width (milliseconds)')\n",
        #plt.ylabel('PBF Width')\n",
        #plt.imshow(taus_collect, cmap=plt.cm.viridis_r)\n",
        #plt.colorbar()\n",
        #plt.show()\n",

         #plt.title('Likelihood: Best Fit Template over Data')\n",
         #plt.ylabel('Pulse Intensity')\n",
         #plt.xlabel('Pulse Period (milliseconds)')\n",
         #plt.plot(time, data_care, '.')\n",
         #plt.plot(time, fitted_templates[b_pbfwidth_index][b_gausswidth_index])\n",
         #plt.show()\n",

         #plt.title('Least Squares: Best Fit Template over Data')\n",
         #plt.ylabel('Pulse Intensity')\n",
         #plt.xlabel('Pulse Period (milliseconds)')\n",
         #plt.plot(time, data_care, '.')\n",
         #plt.plot(time, fitted_templates[lsqs_pbf_index][lsqs_gauss_index])\n",
         #plt.show()\n",

        #fig1 = plt.figure(1)\n",
        ##Plot Data-model\n",
        #frame1=fig1.add_axes((.1,.3,.8,.6))\n",
        ##xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]\n",
        #plt.title('Likelihood: Best Fit Template over Data')\n",
        #plt.ylabel('Pulse Intensity')\n",
        #plt.plot(time, data_care, '.')\n",
        #plt.plot(time, fitted_templates[b_pbfwidth_index][b_gausswidth_index]) \n",
        #frame1.set_xticklabels([]) #Remove x-tic labels for the first frame\n",
        #plt.plot()\n",

        ##Residual plot\n",
        #difference = np.subtract(data_care, fitted_templates[b_pbfwidth_index][b_gausswidth_index])\n",
        #frame2=fig1.add_axes((.1,.1,.8,.2))        \n",
        #plt.plot(time, difference, '.')\n",
        #plt.xlabel('Pulse Period (milliseconds)')\n",
        #plt.ylabel('Residuals')\n",
        #plt.plot()\n",

        #Best Fit Template fitting
        profile = convolved_profiles[beta_index][lsqs_pbf_index][gwidth_index] / np.max(convolved_profiles[beta_index][lsqs_pbf_index][gwidth_index])
        #fitPulse requires template height of one
        z = np.max(profile)
        zind = np.where(profile == z)[0][0]
        ind_diff = xind-zind
        #this lines the profiles up approximately so that Single Pulse finds the true minimum, not just a local min\n",
        profile = np.roll(profile, ind_diff)
        sp = SinglePulse(data_care, opw = np.arange(0, 800))
        fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template matching, scale factor, TOA error, \n",
        #scale factor error, signal to noise ratio, cross-correlation coefficient\n",
        #based on the fitPulse fitting, scale and shift the profile to best fit the inputted data\n",
        #fitPulse figures out the best amplitude itself
        spt = SinglePulse(profile*fitting[2])
        fitted_template = spt.shiftit(fitting[1])
        fitted_template = fitted_template*mask

        plt.figure(2*beta_index)
        fig1 = plt.figure(2*beta_index)
        #Plot Data-model
        frame1=fig1.add_axes((.1,.3,.8,.6))
        #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
        plt.title('SETGwidth='+str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] + 'Best Fit Template over Data w/ Beta = ' + str(betaselect[beta_index]))
        plt.ylabel('Pulse Intensity')
        plt.plot(time, data_care, '.', ms = '2.4')
        plt.plot(time, fitted_template)
        frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
        plt.plot()

        #Residual plot
        difference = np.subtract(data_care, fitted_template)
        frame2=fig1.add_axes((.1,.1,.8,.2))
        plt.plot(time, difference, '.', ms = '2.4')
        plt.xlabel('Pulse Period (milliseconds)')
        plt.ylabel('Residuals')
        plt.plot()

        title = 'SETGwidth='+str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] + 'PBF_fit_plot_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_BETA=' + str(betaselect[beta_index]) + '.png'
        plt.savefig(title)
        plt.close(2*beta_index)

    plt.figure(25)
    plt.xlabel('Beta')
    plt.ylabel('Chi-Squared')
    plt.plot(betaselect, chi_sqs_collect)
    title = 'SETGwidth='+str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] +'PBF_fit_overall_chisqs_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_bestBETA=' + str(betaselect[beta_index]) + '.png'
    plt.savefig(title)
    plt.close(25)
    plt.figure(26)
    plt.xlabel('Beta')
    plt.ylabel('Overall Best PBF Width')
    plt.plot(betaselect, pbf_width_collect)
    title = 'SETGwidth='+str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] +'PBF_fit_overall_widths_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_bestBETA=' + str(betaselect[beta_index]) + '.png'
    plt.savefig(title)
    plt.close(26)
    plt.figure(28)
    plt.xlabel('Beta')
    plt.ylabel('Overall Best Tau (microseconds)')
    plt.errorbar(betaselect, taus_collect)
    title = 'SETGwidth='+str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] +'PBF_fit_overall_tau_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_bestBETA=' + str(betaselect[beta_index]) + '.png'
    plt.savefig(title)
    plt.close(28)

    low_chi = np.min(chi_sqs_collect)
    chi_beta_ind = np.where(chi_sqs_collect == low_chi)[0][0]
    beta_fin = betaselect[chi_beta_ind]
    pbf_width_fin = pbf_width_collect[chi_beta_ind]
    tau_fin = taus_collect[chi_beta_ind]

    pbf_width_ind = np.where(widths == pbf_width_fin)[0][0]
    #now plot the final best template over the data
    profile = convolved_profiles[chi_beta_ind][pbf_width_ind][gwidth_index] / np.max(convolved_profiles[chi_beta_ind][pbf_width_ind][gwidth_index])
    #fitPulse requires template height of one
    z = np.max(profile)
    zind = np.where(profile == z)[0][0]
    ind_diff = xind-zind
    #this lines the profiles up approximately so that Single Pulse finds the true minimum, not just a local min\n",
    profile = np.roll(profile, ind_diff)
    sp = SinglePulse(data_care, opw = np.arange(0, 800))
    fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template matching, scale factor, TOA error, \n",
    #scale factor error, signal to noise ratio, cross-correlation coefficient\n",
    #based on the fitPulse fitting, scale and shift the profile to best fit the inputted data\n",
    #fitPulse figures out the best amplitude itself\n",
    #fitted_shift[data_index1][data_index2] = fitting[1]\n"
    #fitted_scale_factor[data_index1][data_index2] = fitting[2]
    spt = SinglePulse(profile*fitting[2])
    fitted_template = spt.shiftit(fitting[1])
    #fitted_templates[data_index1][data_index2] = fitted_template
    fitted_template = fitted_template*mask

    plt.figure(29)
    fig1 = plt.figure(29)
    #Plot Data-model\n",
    frame1=fig1.add_axes((.1,.3,.8,.6))
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.title('SETGwidth='+str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] +'Best Fit Template over Data w/ Beta = ' + str(betaselect[beta_index]) + '; chisq = ' + str(low_chi)[:5])
    plt.ylabel('Pulse Intensity')
    plt.plot(time, data_care, '.', ms = '2.4')
    plt.plot(time, fitted_template)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.plot()

    #Residual plot
    difference = np.subtract(data_care, fitted_template)
    frame2=fig1.add_axes((.1,.1,.8,.2))
    plt.plot(time, difference, '.', ms = '2.4')
    plt.xlabel('Pulse Period (milliseconds)')
    plt.ylabel('Residuals')
    plt.plot()

    title = 'SETGwidth='+str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] +'PBF_fit_overall_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_PBFW=' + str(pbf_width_fin)[:5] + '.png'
    plt.savefig(title)
    plt.close(29)
    print(title)

    gauss_width_set = gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2)))

    print('SETGwidth='+str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] +'\n'+'Min Chi-sq = ' + str(low_chi) + '\n'+'Best tau = ' + str(tau_fin) \
          + '\n' + '\n'+'Best PBF Width = ' \
              + str(pbf_width_fin) + '\n'+'Best Fit Beta = ' + str(beta_fin) + '\n'+'Frequency = '\
                  + str(freqs_care))
    return(low_chi, tau_fin, gauss_width_set, pbf_width_fin, beta_fin, freqs_care)

def fit_dec_exp(mjdi, data, freqsm, freq_subint_index):

    print('Fitting decaying exponential')

    s = subaverages4(mjdi, data, freqsm)

    data_care = s[0][freq_subint_index]

    #calculate the root mean square noise of the off pulse in order to
    #normalize chi-squared
    rms_collect = 0
    for i in range(600):
        rms_collect += data_care[i]*data_care[i]
    rms = math.sqrt(rms_collect/600)

    freqs_care = s[1][freq_subint_index]

    x = np.max(data_care)
    xind = np.where(data_care == x)[0][0]

    #set the offpulse regions to zero because essentially oscillating there
    #vary the offpulse region depending on frequency
    mask = np.zeros(2048)
    if freqs_care >= 1600:
        start_index = 700
        stop_index = 1548
    elif freqs_care >= 1400 and freqs_care < 1600:
        start_index = 700
        stop_index = 1648
    elif freqs_care >= 1200 and freqs_care < 1400:
        start_index = 650
        stop_index = 1798
    elif freqs_care >= 1000 and freqs_care < 1200:
        start_index = 600
        stop_index = 1948
    mask[start_index:stop_index] = 1.0

    data_care = data_care*mask

    chi_sqs_array = np.zeros((np.size(widths),np.size(parameters[:,0])))

    data_index1 = 0
    #for the varying pbf_widths
    for ii in convolved_profiles_exp:
        data_index2 = 0
        #for the varying gaussian widths
        for i in ii:
            # profile = i / np.max(i) #fitPulse requires template height of one
            # z = np.max(profile)
            # #print(z)\n",
            # zind = np.where(profile == z)[0][0]
            # #print(zind)\n",
            # ind_diff = xind-zind
            # #this lines the profiles up approximately so that Single Pulse finds the true minimum, not just a local min
            # profile = np.roll(profile, ind_diff)
            # sp = SinglePulse(data_care, opw = np.arange(0, 800))
            # fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template matching, scale factor, TOA error,
            # #scale factor error, signal to noise ratio, cross-correlation coefficient
            # #based on the fitPulse fitting, scale and shift the profile to best fit the inputted data
            # #fitPulse figures out the best amplitude itself
            # fitted_shift[data_index1][data_index2] = fitting[1]
            # fitted_scale_factor[data_index1][data_index2] = fitting[2]
            # spt = SinglePulse(profile*fitting[2])
            # fitted_template = spt.shiftit(fitting[1])
            # fitted_templates[data_index1][data_index2] = fitted_template
            # fitted_template[:700] = 0.0
            # fitted_template[1548:] = 0.0
            # chi_sq_measure = chi2_distance(data_care, fitted_template)
            chi_sqs_array[data_index1][data_index2] = fit_sing(i, xind, data_care, freqs_care, 4)
            data_index2 = data_index2+1
        data_index1 = data_index1+1

    plt.figure(1)
    chisqs = chi_sqs_array - np.amin(chi_sqs_array)
    chisqs = np.exp((-0.5)*chisqs)
    plt.title('Fit Chi-sqs')
    plt.xlabel('Rounded Gaussian FWHM (microseconds)')
    plt.ylabel('PBF Width')
    #scale the chi-squared array by the rms value of the profile
    chi_sqs_array = np.divide(chi_sqs_array,(rms*rms))
    plt.imshow(chi_sqs_array, cmap=plt.cm.viridis_r, origin = 'lower')
    gauss_ticks = np.zeros(10)
    for i in range(10):
        gauss_ticks[i] = str(int(gauss_widths[i*5] * (0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))) #converted to microseconds FWHM
    pbf_ticks = np.zeros(10)
    for i in range(10):
        pbf_ticks[i] = str(widths[i*5])[:3]
    plt.xticks(ticks = np.linspace(0,50,num=10), labels = gauss_ticks)
    plt.yticks(ticks = np.linspace(0,50,num=10), labels = pbf_ticks)
    plt.colorbar()
    title = 'EXP_fit_chisq_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '.png'
    plt.savefig(title)
    plt.close(1)
    low_chi = find_nearest(chi_sqs_array, 0.0)[0]
    #print("Minimum chi-squared: " + str(low_chi))

    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
    lsqs_pbf_val = widths[lsqs_pbf_index]
    lsqs_gauss_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
    lsqs_gauss_val = widths_gaussian[lsqs_gauss_index]

    #probabilitiesx = np.sum(chisqs, axis=1)
    #p_pbfwidth = np.where(probabilitiesx == np.max(probabilitiesx))[0]
    #probabilitiesy = np.sum(chisqs, axis=0)
    #p_gausswidth = np.where(probabilitiesy == np.max(probabilitiesy))[0]
    #likelihoodx = likelihood_evaluator(widths, probabilitiesx)
    #likelihoody = likelihood_evaluator(widths_gaussian, probabilitiesy)

    #b_pbfwidth_index = find_nearest(widths, likelihoodx[0])[1][0][0]
    #b_gausswidth_index = find_nearest(widths_gaussian, likelihoody[0])[1][0][0]

    tau_fin = tau_values_exp[lsqs_pbf_index]
    gaussian_width_fin = lsqs_gauss_val * (0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))) #converted to microseconds FWHM
    pbf_width_fin = lsqs_pbf_val

    profile = convolved_profiles_exp[lsqs_pbf_index][lsqs_gauss_index] / np.max(convolved_profiles_exp[lsqs_pbf_index][lsqs_gauss_index])
    #fitPulse requires template height of one
    z = np.max(profile)
    zind = np.where(profile == z)[0][0]
    ind_diff = xind-zind
    #this lines the profiles up approximately so that Single Pulse finds the true minimum, not just a local min\n",
    profile = np.roll(profile, ind_diff)
    sp = SinglePulse(data_care, opw = np.arange(0, 800))
    fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template matching, scale factor, TOA error, \n",
    #scale factor error, signal to noise ratio, cross-correlation coefficient\n",
    #based on the fitPulse fitting, scale and shift the profile to best fit the inputted data\n",
    #fitPulse figures out the best amplitude itself\n",
    #fitted_shift[data_index1][data_index2] = fitting[1]\n"
    #fitted_scale_factor[data_index1][data_index2] = fitting[2]
    spt = SinglePulse(profile*fitting[2])
    fitted_template = spt.shiftit(fitting[1])
    #fitted_templates[data_index1][data_index2] = fitted_template
    fitted_template[:700] = 0.0
    fitted_template[1548:] = 0.0

    plt.figure(29)
    fig1 = plt.figure(29)
    #Plot Data-model\n",
    frame1=fig1.add_axes((.1,.3,.8,.6))
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.title('Best Fit Template over Data; chisq = ' + str(low_chi)[:5])
    plt.ylabel('Pulse Intensity')
    plt.plot(time, data_care, '.', ms = '2.4')
    plt.plot(time, fitted_template)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame

    #Residual plot
    difference = np.subtract(data_care, fitted_template)
    frame2=fig1.add_axes((.1,.1,.8,.2))
    plt.plot(time, difference, '.', ms = '2.4')
    plt.xlabel('Pulse Period (milliseconds)')
    plt.ylabel('Residuals')

    title = 'EXP_fit_overall_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_GAUSSW=' + str(gaussian_width_fin)[:4] + '_PBFW=' + str(pbf_width_fin)[:5] + '.png'
    plt.savefig(title)
    plt.close(29)

    #     plt.title('Likelihood: Best Fit Template over Data')
    #     plt.ylabel('Pulse Intensity')
    #     plt.xlabel('Pulse Period (milliseconds)')
    #     plt.plot(time, data_care, '.')
    #     plt.plot(time, fitted_templates[b_pbfwidth_index][b_gausswidth_index])
    #     plt.show()

    #     plt.title('Least Squares: Best Fit Template over Data')
    #     plt.ylabel('Pulse Intensity')
    #     plt.xlabel('Pulse Period (milliseconds)')
    #     plt.plot(time, data_care, '.')
    #     plt.plot(time, fitted_templates[lsqs_pbf_index][lsqs_gauss_index])
    #     plt.show()

    print('Min Chi-sq = ' + str(low_chi) + '\n'+'Best tau = ' + str(tau_fin) \
          + '\n'+'Best Gauss Width = ' + str(gaussian_width_fin) + '\n'+'Best PBF Width = ' \
              + str(pbf_width_fin) + '\n'+'Frequency = '\
                  + str(freqs_care))
    return(low_chi, tau_fin, gaussian_width_fin, pbf_width_fin, freqs_care)


def fit_dec_setgwidth_exp(mjdi, data, freqsm, freq_subint_index, gwidth_index):

    print('Fitting decaying exponential')

    s = subaverages4(mjdi, data, freqsm)

    data_care = s[0][freq_subint_index]

    #calculate the root mean square noise of the off pulse in order to
    #normalize chi-squared
    rms_collect = 0
    for i in range(600):
        rms_collect += data_care[i]*data_care[i]
    rms = math.sqrt(rms_collect/600)

    freqs_care = s[1][freq_subint_index]

    x = np.max(data_care)
    xind = np.where(data_care == x)[0][0]

    #set the offpulse regions to zero because essentially oscillating there
    #vary the offpulse region depending on frequency
    mask = np.zeros(2048)
    if freqs_care >= 1600:
        start_index = 700
        stop_index = 1548
    elif freqs_care >= 1400 and freqs_care < 1600:
        start_index = 700
        stop_index = 1648
    elif freqs_care >= 1200 and freqs_care < 1400:
        start_index = 650
        stop_index = 1798
    elif freqs_care >= 1000 and freqs_care < 1200:
        start_index = 600
        stop_index = 1948
    mask[start_index:stop_index] = 1.0

    data_care = data_care*mask

    chi_sqs_array = np.zeros(np.size(widths))

    data_index1 = 0
    #for the varying pbf_widths
    for ii in convolved_profiles_exp:
        #for the set gaussian width
        i = ii[gwidth_index]
        chi_sqs_array[data_index1] = fit_sing(i, xind, data_care, freqs_care, 3)
        data_index1 = data_index1+1

    plt.figure(1)
    chisqs = chi_sqs_array - np.amin(chi_sqs_array)
    chisqs = np.exp((-0.5)*chisqs)
    plt.title('Fit Chi-sqs')
    plt.ylabel('Chi-sq')
    plt.xlabel('PBF Width')
    #scale the chi-squared array by the rms value of the profile
    chi_sqs_array = np.divide(chi_sqs_array,(rms*rms))
    plt.plot(widths, chi_sqs_array)
    title = 'SETGEXP_fit_chisq_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_GWIDTH=' + str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] + '.png'
    plt.savefig(title)
    plt.show()
    plt.close(1)
    low_chi = find_nearest(chi_sqs_array, 0.0)[0]
    low_chi_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]

    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
    lsqs_pbf_val = widths[lsqs_pbf_index]

    tau_fin = tau_values_exp[lsqs_pbf_index]
    pbf_width_fin = lsqs_pbf_val

    profile = convolved_profiles_exp[lsqs_pbf_index][gwidth_index] / np.max(convolved_profiles_exp[lsqs_pbf_index][gwidth_index])
    #fitPulse requires template height of one
    z = np.max(profile)
    zind = np.where(profile == z)[0][0]
    ind_diff = xind-zind
    #this lines the profiles up approximately so that Single Pulse finds the true minimum, not just a local min\n",
    profile = np.roll(profile, ind_diff)
    sp = SinglePulse(data_care, opw = np.arange(0, 800))
    fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template matching, scale factor, TOA error, \n",
    #scale factor error, signal to noise ratio, cross-correlation coefficient\n",
    #based on the fitPulse fitting, scale and shift the profile to best fit the inputted data\n",
    #fitPulse figures out the best amplitude itself\n",
    #fitted_shift[data_index1][data_index2] = fitting[1]\n"
    #fitted_scale_factor[data_index1][data_index2] = fitting[2]
    spt = SinglePulse(profile*fitting[2])
    fitted_template = spt.shiftit(fitting[1])
    #fitted_templates[data_index1][data_index2] = fitted_template
    fitted_template[:700] = 0.0
    fitted_template[1548:] = 0.0


    #ERROR TEST - one reduced chi-squared unit above and below and these
    #chi-squared bins are for varying pbf width
    below = np.where((chi_sqs_array == low_chi+1))
    above = np.where((chi_sqs_array[low_chi_index:]))[0][0] + low_chi_index

    minus_error = low_chi_index - below
    plus_error = above - low_chi_index

    print(tau_values_exp[60] - tau_values_exp[59])
    print(widths[60] - widths[59])

    print(tau_values_exp[61] - tau_values_exp[60])
    print(widths[61] - widths[60])

    #above lines printed 12.597070312499966, 0.2572327044025151, 12.597070312500023, 0.2572327044025151
    #for decaying exp just have to scale

    print(below)
    print(above)
    print(minus_error)
    print(plus_error)


    plt.figure(29)
    fig1 = plt.figure(29)
    #Plot Data-model\n",
    frame1=fig1.add_axes((.1,.3,.8,.6))
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.title('Best Fit Template over Data; chisq = ' + str(low_chi)[:5])
    plt.ylabel('Pulse Intensity')
    plt.plot(time, data_care, '.', ms = '2.4')
    plt.plot(time, fitted_template)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame

    #Residual plot
    difference = np.subtract(data_care, fitted_template)
    frame2=fig1.add_axes((.1,.1,.8,.2))
    plt.plot(time, difference, '.', ms = '2.4')
    plt.xlabel('Pulse Period (milliseconds)')
    plt.ylabel('Residuals')

    title = 'SETGEXP_fit_overall_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_GAUSSW=' + str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] + '_PBFW=' + str(pbf_width_fin)[:5] + '.png'
    plt.savefig(title)
    plt.close(29)

    #     plt.title('Likelihood: Best Fit Template over Data')
    #     plt.ylabel('Pulse Intensity')
    #     plt.xlabel('Pulse Period (milliseconds)')
    #     plt.plot(time, data_care, '.')
    #     plt.plot(time, fitted_templates[b_pbfwidth_index][b_gausswidth_index])
    #     plt.show()

    #     plt.title('Least Squares: Best Fit Template over Data')
    #     plt.ylabel('Pulse Intensity')
    #     plt.xlabel('Pulse Period (milliseconds)')
    #     plt.plot(time, data_care, '.')
    #     plt.plot(time, fitted_templates[lsqs_pbf_index][lsqs_gauss_index])
    #     plt.show()

    print('Min Chi-sq = ' + str(low_chi) + '\n'+'Best tau = ' + str(tau_fin) \
          + '\n'+'Set Gauss Width = ' + str(gauss_widths[gwidth_index]* \
          (0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] + \
          '\n'+'Best PBF Width = ' + str(pbf_width_fin) + '\n'+'Frequency = '\
          + str(freqs_care))

    gwidth = str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * \
    (2.0*math.sqrt(2*math.log(2))))[:4]

    return(low_chi, tau_fin, gwidth, pbf_width_fin, freqs_care, len(s[1]), minus_error, plus_error)


def fit_cons_beta_profile(mjdi, data, freqsm, freq_subint_index, beta_index, plot_conv=False):

    print("Fitting for Beta = " + str(betaselect[beta_index]))

    s = subaverages4(mjdi, data, freqsm)

    data_care = s[0][freq_subint_index]

    #calculate the root mean square noise of the off pulse in order to
    #normalize chi-squared
    rms_collect = 0
    for i in range(600):
        rms_collect += data_care[i]*data_care[i]
    rms = math.sqrt(rms_collect/600)

    freqs_care = s[1][freq_subint_index]

    x = np.max(data_care)
    xind = np.where(data_care == x)[0][0]

    #set the offpulse regions to zero because essentially oscillating there
    #vary the offpulse region depending on frequency
    mask = np.zeros(2048)
    if freqs_care >= 1600:
        start_index = 700
        stop_index = 1548
    elif freqs_care >= 1400 and freqs_care < 1600:
        start_index = 700
        stop_index = 1648
    elif freqs_care >= 1200 and freqs_care < 1400:
        start_index = 650
        stop_index = 1798
    elif freqs_care >= 1000 and freqs_care < 1200:
        start_index = 600
        stop_index = 1948
    mask[start_index:stop_index] = 1.0

    data_care = data_care*mask

    chi_sqs_array = np.zeros((np.size(widths),np.size(parameters[:,0])))

    data_index1 = 0
    #for the varying pbf_widths
    for ii in convolved_profiles[beta_index]:
        data_index2 = 0
        #for the varying gaussian widths
        for i in ii:
            # profile = i / np.max(i) #fitPulse requires template height of one
            # z = np.max(profile)
            # #print(z)\n",
            # zind = np.where(profile == z)[0][0]
            # #print(zind)\n",
            # ind_diff = xind-zind
            # #this lines the profiles up approximately so that Single Pulse finds the true minimum, not just a local min
            # profile = np.roll(profile, ind_diff)
            # sp = SinglePulse(data_care, opw = np.arange(0, 800))
            # fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template matching, scale factor, TOA error,
            # #scale factor error, signal to noise ratio, cross-correlation coefficient
            # #based on the fitPulse fitting, scale and shift the profile to best fit the inputted data
            # #fitPulse figures out the best amplitude itself
            # fitted_shift[data_index1][data_index2] = fitting[1]
            # fitted_scale_factor[data_index1][data_index2] = fitting[2]
            # spt = SinglePulse(profile*fitting[2])
            # fitted_template = spt.shiftit(fitting[1])
            # fitted_templates[data_index1][data_index2] = fitted_template
            # fitted_template[:700] = 0.0
            # fitted_template[1548:] = 0.0
            # chi_sq_measure = chi2_distance(data_care, fitted_template)
            chi_sqs_array[data_index1][data_index2] = fit_sing(i, xind, data_care, freqs_care, 4)
            data_index2 = data_index2+1
        data_index1 = data_index1+1


    plt.figure(1)
    plt.title('Fit Chi-sqs')
    plt.xlabel('Rounded Gaussian FWHM (microseconds)')
    plt.ylabel('PBF Width')
    chi_sqs_array = np.divide(chi_sqs_array,(rms*rms))
    plt.imshow(chi_sqs_array, cmap=plt.cm.viridis_r, origin = 'lower')
    gauss_ticks = np.zeros(10)
    for i in range(10):
        gauss_ticks[i] = str(int(gauss_widths[i*5] * (0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))) # converted to microseconds FWHM
    pbf_ticks = np.zeros(10)
    for i in range(10):
        pbf_ticks[i] = str(widths[i*5])[:3]
    plt.xticks(ticks = np.linspace(0,50,num=10), labels = gauss_ticks)
    plt.yticks(ticks = np.linspace(0,50,num=10), labels = pbf_ticks)
    plt.colorbar()
    title = 'ONEB_fit_chisq_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_BETA+' + str(betaselect[beta_index]) + '.png'
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig(title)
    plt.close(1)
    low_chi = find_nearest(chi_sqs_array, 0.0)[0]
    #print("Minimum chi-squared: " + str(low_chi))

    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
    lsqs_pbf_val = widths[lsqs_pbf_index]
    lsqs_gauss_index = find_nearest(chi_sqs_array, 0.0)[1][1][0]
    lsqs_gauss_val = widths_gaussian[lsqs_gauss_index]

    # likelihood
    # chisqs = chi_sqs_array - np.amin(chi_sqs_array)
    # chisqs = np.exp((-0.5)*chisqs)
    # probabilitiesx = np.sum(chisqs, axis=1)
    # p_pbfwidth = np.where(probabilitiesx == np.max(probabilitiesx))[0]
    # probabilitiesy = np.sum(chisqs, axis=0)
    # p_gausswidth = np.where(probabilitiesy == np.max(probabilitiesy))[0]
    # likelihoodx = likelihood_evaluator(widths, probabilitiesx)
    # likelihoody = likelihood_evaluator(widths_gaussian, probabilitiesy)

    #b_pbfwidth_index = find_nearest(widths, likelihoodx[0])[1][0][0]
    #b_gausswidth_index = find_nearest(widths_gaussian, likelihoody[0])[1][0][0]

    tau_fin = tau_values[beta_index][lsqs_pbf_index]
    gaussian_width_fin = lsqs_gauss_val * (0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))) #converted to microseconds FWHM
    pbf_width_fin = lsqs_pbf_val



    p = parameters[lsqs_gauss_index]
    gaussian = (p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2])))) / trapz(p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))
    gaussian = gaussian/(np.max(gaussian))
    #gaussian to unit height and lining up to the data
    indg_diff = xind-int(p[1])

    #discrete pbf to unit height and lining up to the data
    discrete_pbf = conv.pbf_data_unitarea[beta_index][lsqs_pbf_index] / np.max(conv.pbf_data_unitarea[beta_index][lsqs_pbf_index])
    discrete_center = np.where((discrete_pbf == np.max(discrete_pbf)))[0][0]
    indp_diff = xind - discrete_center

    profile = convolved_profiles[beta_index][lsqs_pbf_index][lsqs_gauss_index] / np.max(convolved_profiles_exp[beta_index][lsqs_pbf_index][lsqs_gauss_index])
    #fitPulse requires template height of one
    z = np.max(profile)
    zind = np.where(profile == z)[0][0]
    ind_diff = xind-zind
    #this lines the profiles up approximately so that Single Pulse finds the true minimum, not just a local min\n",
    profile = np.roll(profile, ind_diff)
    sp = SinglePulse(data_care, opw = np.arange(0, 800))
    fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template matching, scale factor, TOA error, \n",
    #scale factor error, signal to noise ratio, cross-correlation coefficient\n",
    #based on the fitPulse fitting, scale and shift the profile to best fit the inputted data\n",
    #fitPulse figures out the best amplitude itself\n",
    #fitted_shift[data_index1][data_index2] = fitting[1]\n"
    #fitted_scale_factor[data_index1][data_index2] = fitting[2]
    spt = SinglePulse(profile*fitting[2])
    fitted_template = spt.shiftit(fitting[1])
    #fitted_templates[data_index1][data_index2] = fitted_template
    fitted_template = fitted_template*mask

    plt.figure(2)
    fig1 = plt.figure(2)
    #Plot Data-model\n",
    frame1=fig1.add_axes((.1,.3,.8,.6))
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]\n",
    plt.title('Best Fit Template over Data; chisq = ' + str(low_chi)[:5])
    plt.ylabel('Pulse Intensity')
    if plot_conv == True:
        plt.plot(time, np.roll((gaussian*np.max(fitted_template)),indg_diff), color = 'purple')
        plt.plot(time, np.roll((discrete_pbf*np.max(fitted_template)), indp_diff), color = 'green')
    plt.plot(time, data_care, '.', ms = '2.4')
    plt.plot(time, fitted_template)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame

    #Residual plot
    difference = np.subtract(data_care, fitted_template)
    frame2=fig1.add_axes((.1,.1,.8,.2))
    plt.plot(time, difference, '.', ms = '2.4')
    plt.xlabel('Pulse Period (milliseconds)')
    plt.ylabel('Residuals')
    title = 'ONEB_fit_overall_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_GAUSSW=' + str(gaussian_width_fin)[:4] + '_PBFW=' + str(pbf_width_fin) + '_BETA=' + str(betaselect[beta_index]) + '.png'
    print(title)
    plt.plot()
    plt.show()
    plt.savefig(title)
    plt.close(2)

    #     plt.title('Likelihood: Best Fit Template over Data')
    #     plt.ylabel('Pulse Intensity')
    #     plt.xlabel('Pulse Period (milliseconds)')
    #     plt.plot(time, data_care, '.')
    #     plt.plot(time, fitted_templates[b_pbfwidth_index][b_gausswidth_index])
    #     plt.show()

    #     plt.title('Least Squares: Best Fit Template over Data')
    #     plt.ylabel('Pulse Intensity')
    #     plt.xlabel('Pulse Period (milliseconds)')
    #     plt.plot(time, data_care, '.')
    #     plt.plot(time, fitted_templates[lsqs_pbf_index][lsqs_gauss_index])
    #     plt.show()
    print('Min Chi-sq = ' + str(low_chi) + '\n'+'Best tau = ' + str(tau_fin) \
          + '\n'+'Best Gauss Width = ' + str(gaussian_width_fin) + '\n'+'Best PBF Width = ' \
              + str(pbf_width_fin) + '\n'+'Beta set = ' + str(betaselect[beta_index]) + '\n'+'Frequency = '\
                  + str(freqs_care))
    return(low_chi, tau_fin, gaussian_width_fin, pbf_width_fin, freqs_care)

def fit_cons_beta_gauss_profile(mjdi, data, freqsm, freq_subint_index, beta_index, gwidth_index, plot_conv=False):

    print("Fitting for Beta = " + str(betaselect[beta_index]))

    s = subaverages4(mjdi, data, freqsm)

    data_care = s[0][freq_subint_index]

    #calculate the root mean square noise of the off pulse in order to
    #normalize chi-squared
    rms_collect = 0
    for i in range(600):
        rms_collect += data_care[i]*data_care[i]
    rms = math.sqrt(rms_collect/600)

    freqs_care = s[1][freq_subint_index]

    x = np.max(data_care)
    xind = np.where(data_care == x)[0][0]

    #set the offpulse regions to zero because essentially oscillating there
    #vary the offpulse region depending on frequency
    mask = np.zeros(2048)
    if freqs_care >= 1600:
        start_index = 700
        stop_index = 1548
    elif freqs_care >= 1400 and freqs_care < 1600:
        start_index = 700
        stop_index = 1648
    elif freqs_care >= 1200 and freqs_care < 1400:
        start_index = 650
        stop_index = 1798
    elif freqs_care >= 1000 and freqs_care < 1200:
        start_index = 600
        stop_index = 1948
    mask[start_index:stop_index] = 1.0

    data_care = data_care*mask

    chi_sqs_array = np.zeros(np.size(widths))

    data_index1 = 0
    #for the varying pbf_widths
    for ii in convolved_profiles[beta_index]:
        #for the set gaussian width
        i = ii[gwidth_index]
        chi_sqs_array[data_index1] = fit_sing(i, xind, data_care, freqs_care, 3)
        data_index1 = data_index1+1


    plt.figure(1)
    plt.title('Fit Chi-sqs')
    plt.xlabel('PBF Width')
    plt.ylabel('Chi-sqs')
    chi_sqs_array = np.divide(chi_sqs_array,(rms*rms))
    plt.plot(widths, chi_sqs_array)
    title = 'ONEBSETG_fit_chisq_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_BETA=' + str(betaselect[beta_index]) + '_GWIDTH=' + str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] + '.png'
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig(title)
    plt.close(1)
    low_chi = find_nearest(chi_sqs_array, 0.0)[0]
    #print("Minimum chi-squared: " + str(low_chi))

    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
    lsqs_pbf_val = widths[lsqs_pbf_index]

    tau_fin = tau_values[beta_index][lsqs_pbf_index]
    pbf_width_fin = lsqs_pbf_val

    p = parameters[gwidth_index]
    gaussian = (p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2])))) / trapz(p[0]*np.exp((-1.0/2.0)*(((t-p[1])/p[2])*((t-p[1])/p[2]))))
    gaussian = gaussian/(np.max(gaussian))
    #gaussian to unit height and lining up to the data
    indg_diff = xind-int(p[1])

    #discrete pbf to unit height and lining up to the data
    discrete_pbf = conv.pbf_data_unitarea[beta_index][lsqs_pbf_index] / np.max(conv.pbf_data_unitarea[beta_index][lsqs_pbf_index])
    discrete_center = np.where((discrete_pbf == np.max(discrete_pbf)))[0][0]
    indp_diff = xind - discrete_center

    profile = convolved_profiles[beta_index][lsqs_pbf_index][gwidth_index] / np.max(convolved_profiles_exp[beta_index][lsqs_pbf_index][gwidth_index])
    #fitPulse requires template height of one
    z = np.max(profile)
    zind = np.where(profile == z)[0][0]
    ind_diff = xind-zind
    #this lines the profiles up approximately so that Single Pulse finds the true minimum, not just a local min\n",
    profile = np.roll(profile, ind_diff)
    sp = SinglePulse(data_care, opw = np.arange(0, 800))
    fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template matching, scale factor, TOA error, \n",
    #scale factor error, signal to noise ratio, cross-correlation coefficient\n",
    #based on the fitPulse fitting, scale and shift the profile to best fit the inputted data\n",
    #fitPulse figures out the best amplitude itself\n",
    #fitted_shift[data_index1][data_index2] = fitting[1]\n"
    #fitted_scale_factor[data_index1][data_index2] = fitting[2]
    spt = SinglePulse(profile*fitting[2])
    fitted_template = spt.shiftit(fitting[1])
    #fitted_templates[data_index1][data_index2] = fitted_template
    fitted_template = fitted_template*mask

    plt.figure(2)
    fig1 = plt.figure(2)
    #Plot Data-model\n",
    frame1=fig1.add_axes((.1,.3,.8,.6))
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]\n",
    plt.title('Best Fit Template over Data; chisq = ' + str(low_chi)[:5])
    plt.ylabel('Pulse Intensity')
    if plot_conv == True:
        plt.plot(time, np.roll((gaussian*np.max(fitted_template)),indg_diff), color = 'purple')
        plt.plot(time, np.roll((discrete_pbf*np.max(fitted_template)), indp_diff), color = 'green')
    plt.plot(time, data_care, '.', ms = '2.4')
    plt.plot(time, fitted_template)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame

    #Residual plot
    difference = np.subtract(data_care, fitted_template)
    frame2=fig1.add_axes((.1,.1,.8,.2))
    plt.plot(time, difference, '.', ms = '2.4')
    plt.xlabel('Pulse Period (milliseconds)')
    plt.ylabel('Residuals')
    title = 'ONEBSETG_fit_overall_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_GAUSSW=' + str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] + '_PBFW=' + str(pbf_width_fin) + '_BETA=' + str(betaselect[beta_index]) + '.png'
    print(title)
    plt.plot()
    plt.savefig(title)
    plt.close(2)

    #     plt.title('Likelihood: Best Fit Template over Data')
    #     plt.ylabel('Pulse Intensity')
    #     plt.xlabel('Pulse Period (milliseconds)')
    #     plt.plot(time, data_care, '.')
    #     plt.plot(time, fitted_templates[b_pbfwidth_index][b_gausswidth_index])
    #     plt.show()

    #     plt.title('Least Squares: Best Fit Template over Data')
    #     plt.ylabel('Pulse Intensity')
    #     plt.xlabel('Pulse Period (milliseconds)')
    #     plt.plot(time, data_care, '.')
    #     plt.plot(time, fitted_templates[lsqs_pbf_index][lsqs_gauss_index])
    #     plt.show()
    print('Min Chi-sq = ' + str(low_chi) + '\n'+'Best tau = ' + str(tau_fin) \
          + '\n'+'Set gaussian width = ' + str(gauss_widths[gwidth_index]* \
          (0.0021499/2048) * 1e6 * (2.0*math.sqrt(2*math.log(2))))[:4] + \
          '\n'+'Best PBF Width = ' + str(pbf_width_fin) + '\n'+'Beta set = ' \
          + str(betaselect[beta_index]) + '\n'+'Frequency = ' + str(freqs_care))
    # return the lowest chi-squared value, tau, gwidth, pbf_width, frequency and number of subaverages

    gwidth = str(gauss_widths[gwidth_index]*(0.0021499/2048) * 1e6 * \
    (2.0*math.sqrt(2*math.log(2))))[:4]

    return(low_chi, tau_fin, gwidth, pbf_width_fin, freqs_care, len(s[1]))


def fit_cons_beta_ipfd(mjdi, data, freqsm, freq_subint_index, beta_index): #intrinsic pulse from data

    s = subaverages4(mjdi, data, freqsm)

    data_care = s[0][freq_subint_index]

    #calculate the root mean square noise of the off pulse in order to
    #normalize chi-squared
    rms_collect = 0
    for i in range(600):
        rms_collect += data_care[i]*data_care[i]
    rms = math.sqrt(rms_collect/600)

    freqs_care = s[1][freq_subint_index]

    x = np.max(data_care)
    xind = np.where(data_care == x)[0][0]

    #set the offpulse regions to zero because essentially oscillating there
    #vary the offpulse region depending on frequency
    mask = np.zeros(2048)
    if freqs_care >= 1600:
        start_index = 700
        stop_index = 1548
    elif freqs_care >= 1400 and freqs_care < 1600:
        start_index = 700
        stop_index = 1648
    elif freqs_care >= 1200 and freqs_care < 1400:
        start_index = 650
        stop_index = 1798
    elif freqs_care >= 1000 and freqs_care < 1200:
        start_index = 600
        stop_index = 1948
    mask[start_index:stop_index] = 1.0

    data_care = data_care*mask

    chi_sqs_array = np.zeros((np.size(widths)))

    data_index1 = 0
    #for the varying pbf_widths
    for i in convolved_w_dataintrins[beta_index]:
        chi_sqs_array[data_index1] = fit_sing(i, xind, data_care, freqs_care, 3)
        data_index1 = data_index1+1

    plt.figure(1)
    chi_sqs_array = np.divide(chi_sqs_array,(rms*rms))
    plt.title('Fit Chi-sqs')
    plt.xlabel('PBF Width')
    plt.ylabel('Chi-sqs')
    plt.plot(widths, chi_sqs_array)
    title = 'ONEBINTR_fit_chisq_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_BETA+' + str(betaselect[beta_index]) + '.png'
    plt.savefig(title)
    low_chi = find_nearest(chi_sqs_array, 0.0)[0]
    plt.show()
    plt.close(1)

    lsqs_pbf_index = find_nearest(chi_sqs_array, 0.0)[1][0][0]
    lsqs_pbf_val = widths[lsqs_pbf_index]

    #likelihoodx = likelihood_evaluator(widths, chisqs)

    #b_pbfwidth_index = find_nearest(widths, likelihoodx[0])[1][0][0]

    print('Lowest Chi-sqs:')
    print("Best fit PBF width: "+ str(lsqs_pbf_val))
    print('Best fit tau value: ' + str(tau_values[beta_index][lsqs_pbf_index]) + ' microseconds')


    profile = convolved_w_dataintrins[beta_index][lsqs_pbf_index] / np.max(convolved_w_dataintrins[beta_index][lsqs_pbf_index])
    #fitPulse requires template height of one
    z = np.max(profile)
    zind = np.where(profile == z)[0][0]
    ind_diff = xind-zind
    #this lines the profiles up approximately so that Single Pulse finds the true minimum, not just a local min
    profile = np.roll(profile, ind_diff)
    sp = SinglePulse(data_care, opw = np.arange(0, 800))
    fitting = sp.fitPulse(profile) #TOA cross-correlation, TOA template matching, scale factor, TOA error,
    #scale factor error, signal to noise ratio, cross-correlation coefficient
    #based on the fitPulse fitting, scale and shift the profile to best fit the inputted data
    #fitPulse figures out the best amplitude itself
    spt = SinglePulse(profile*fitting[2])
    fitted_template = spt.shiftit(fitting[1])
    fitted_template = fitted_template*mask

    plt.figure(50*beta_index)
    fig1 = plt.figure(50*beta_index)
    #Plot Data-model
    frame1=fig1.add_axes((.1,.3,.8,.6))
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.title('Best Fit Template over Data; chisq = ' + str(low_chi)[:5])
    plt.ylabel('Pulse Intensity')
    plt.plot(time, data_care, '.', ms = '2.4')
    plt.plot(time, fitted_template)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.plot()

    #Residual plot
    difference = np.subtract(data_care, fitted_template)
    frame2=fig1.add_axes((.1,.1,.8,.2))
    plt.plot(time, difference, '.', ms = '2.4')
    plt.xlabel('Pulse Period (milliseconds)')
    plt.ylabel('Residuals')
    plt.plot()

    title = 'ONEBINTR_fit_chisq_for_MJD=' + str(mjdi)[:5] +'_FREQ=' + str(freqs_care)[:4] + '_BETA=' + str(betaselect[beta_index]) + '_PBFW=' + str(lsqs_pbf_val)[:5] + '.png'
    plt.savefig(title)
    plt.close(50*beta_index)

#must add return! <3
