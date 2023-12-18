from mcmc_profile_fitting_class import *
from fit_functions import *
from intrinsic_component_powerlaw_fit import *
import sys

screen = str(sys.argv[3]) # 'thin' or 'thick' medium pbf
if screen == 'thick':
    beta = str(sys.argv[1]) # set spectral index of wavenumber spectrum
    zeta = str(sys.argv[2]) # set inner scale of waveumber spectrum
else:
    beta = float(sys.argv[1]) # set spectral index of wavenumber spectrum
    zeta = float(sys.argv[2]) # set inner scale of waveumber spectrum
rerun = str(sys.argv[4]) # if 'rerun', recalulate the intrinsic components.
# otherwise, just plot the already calculated data for the above inputs. Throws
# exception is has not already been calcualted
if rerun == 'rerun':
    numruns = int(sys.argv[5]) # number of mcmc runs in order to calculate sband
    # shape parameters
    tau_guess = float(sys.argv[6])

plt.rc('font', family = 'serif')

##IMPORT DATA##

sband = np.load('j1903_high_freq_temp_unsmoothed.npy')
# reference nanograv notebook s_band_j1903_data.ipynb for frequency calcualtion
sband_avg_center_freq = 2132.0

# load in the j1903 subband averages - these are averaged over all epochs
with open('j1903_average_profiles_lband.pkl', 'rb') as fp:
    lband_avgs = pickle.load(fp)

freq_keys = []
lband_freqs = []
for i in lband_avgs.keys():
    freq_keys.append(i)
    lband_freqs.append(float(i))

# frequencies corresponding to the j1903 subband averages
lband_freqs = np.array(lband_freqs)

lband_data_array = np.zeros((len(freq_keys), np.size(lband_avgs[freq_keys[0]])))
ind = 0
for i in freq_keys:
    lband_data_array[ind] = lband_avgs[i]
    ind += 1

if __name__ == "__main__":

    mcmc_fitting_object = MCMC_Profile_Fit(beta, zeta, screen, sband, \
    sband_avg_center_freq, 'mjd_average')

    if rerun == 'rerun':

        params = mcmc_fitting_object.profile_component_fit(numruns, tau_guess)

        powerlaw_fitting_object = Intrinsic_Component_Powerlaw_Fit(beta, zeta, \
        'mjd_average', 'mcmc_params|'+mcmc_fitting_object.plot_tag+'.npz', \
        mcmc_fitting_object.frequency, screen, lband_data_array, lband_freqs, \
        sband)

        # 3.667 0.0 thick
        if (beta == '3.667' and (zeta == '0.0' or zeta == '0')):
            amp1_tests=np.linspace(0.5,1.5,21)
            amp3_tests=np.linspace(-0.4,0.3,8)
            cent3_tests=np.linspace(-0.4,0.3,8)
            width3_tests=np.linspace(-1.3,-0.6,8)

        # standard test
        # amp1_tests=np.linspace(-1.5,1.5,21)
        # amp3_tests=np.linspace(-0.8,0.7,8)
        # cent3_tests=np.linspace(-0.8,0.7,8)
        # width3_tests=np.linspace(-0.8,0.7,8)

        powerlaw_fitting_object.fit_comp3(amp3_tests, cent3_tests, width3_tests)
        powerlaw_fitting_object.fit_amp1(amp1_tests)
        powerlaw_fitting_object.plot_modeled()
        powerlaw_fitting_object.plot_modeled_fitted()
