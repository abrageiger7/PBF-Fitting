import sys
sys.path.append('/Users/abrageiger/Documents/research/projects/pbf_fitting')
from epoch_average_intrinsic_fitting_function import *
beta = str(sys.argv[1]) # set spectral index of wavenumber spectrum
zeta = str(sys.argv[2]) # set inner scale of waveumber spectrum
screen = str(sys.argv[3]) # 'thin' or 'thick' medium pbf
rerun = str(sys.argv[4]) # if 'mcmc', recalulate the sband intrinsic params with
# mcmc. If 'powerlaws' recalculate the powerlaws of the intrinsic components
# over lband. If 'both' runs mcmc and powerlaw fitting. Otherwise, just plot the
# already calculated data for the above inputs. Throws exception is has not
# already been calculated.
amp1_tests='default'
amp3_tests='default'
cent3_tests='default'
width3_tests='default'
if rerun == 'mcmc' or rerun == 'powerlaws' or rerun == 'both':
    numruns = int(sys.argv[5]) # number of mcmc runs in order to calculate sband
    # shape parameters
    tau_guess = float(sys.argv[6])
    epoch_average_intrinsic_fitting(beta, zeta, screen, rerun, numruns, tau_guess)
else:
    epoch_average_intrinsic_fitting(beta, zeta, screen, rerun)
