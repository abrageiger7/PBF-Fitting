from mcmc_profile_fitting_class import *
from scattering_time_delay import *
from fit_functions import *
from intrinsic_component_powerlaw_fit import *

mjd = int(sys.argv[1]) # data epoch for this intrinsic fit calculation
beta = str(sys.argv[2]) # set spectral index of wavenumber spectrum
zeta = str(sys.argv[3]) # set inner scale of waveumber spectrum
screen = str(sys.argv[4]) # 4 'thin' or 'thick' medium pbf
rerun = str(sys.argv[5]) # if 'rerun', recalulate the intrinsic components.
# otherwise, just plot the already calculated data for the above inputs. Throws
# exception is has not already been calcualted
if rerun == 'rerun':
    numruns = int(sys.argv[6]) # number of mcmc runs in order to calculate sband
    # shape parameters
    tau_guess = float(sys.argv[7])

plt.rc('font', family = 'serif')

if __name__ == "__main__":

    mcmc_fitting_object = MCMC_Profile_Fit_Per_Epoch(beta, zeta, mjd, screen)

    if rerun == 'rerun':

        params = mcmc_fitting_object.profile_component_fit(numruns, tau_guess)

        pwrlaws = np.zeros(4)

        powerlaw_fitting_object = Intrinsic_Component_Powerlaw_Fit_Per_Epoch(beta, zeta, \
        mjd, 'mcmc_params|'+mcmc_fitting_object.plot_tag+'.npz', pwrlaws, \
        mcmc_fitting_object.frequency, screen, mcmc_fitting_object.profile)

        amp3_tests = np.linspace(-1.0,1.0,2)
        phase_tests = np.linspace(-1.0,1.0,3)
        width_tests = np.linspace(-1.0,1.0,4)
        amp1_tests = np.linspace(-1.0,1.0,2)

        powerlaw_fitting_object.fit_comp3(amp3_tests, phase_tests, width_tests)
        powerlaw_fitting_object.fit_amp1(amp1_tests)
        powerlaw_fitting_object.plot_modeled()
        powerlaw_fitting_object.plot_modeled_fitted()
