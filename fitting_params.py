"""
Created June 2023
@author: Abra Geiger abrageiger7

File that contains all the parameters necessary to fitting so that everything is
consistent across files.
"""

#===============================================================================
# Phase bins and timing
# ==============================================================================

#number of phase bins for a profile - time average every 8 of the original data
#2048
phase_bins = 2048//8

# number of time bins for original cordes pbfs
cordes_phase_bins = 9549

#phase bins
t = np.linspace(0, phase_bins, phase_bins)

#seconds per pulse period
sec_pulse_per = 0.0021499

#seconds to milliseconds conversion
s_to_ms_conv = 1e3

#time phase bins in milliseconds
time = np.arange(0,phase_bins,1) * (sec_pulse_per/phase_bins) * s_to_ms_conv #milliseconds

opr_size = int((600/2048)*num_phase_bins) #number of phase bins for offpulse noise calculation

#===============================================================================
# Fitting Parameters
# ==============================================================================

# beta values chosen to use
betaselect = np.array([3.1, 3.5, 3.667, 3.8, 3.9, 3.95, 3.975, 3.99, 3.995, 3.9975, 3.999, 3.99999])

# zeta values chosen to use
zetaselect = np.array([0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0])

#array of widths used (pbf stretch factors)
#previously as low as .1 and as high as 42
num_pbfwidth = 400
widths = np.linspace(0.0001, 35.0, num_widths)

#array of gaussian widths (phase bins)
num_gwidth = 200
widths_gaussian = np.linspace((0.01/2048)*phase_bins, (150.0/2048)*phase_bins, num_gwidths)

#gauss widths converted to fwhm microseconds
gauss_fwhm = widths_gaussian * ((0.0021499/phase_bins) * 1e6 * (2.0*math.sqrt(2*math.log(2))))

#gaussian parameters in phase bins and arbitrary intensity comparitive to data
parameters = np.zeros((200, 3))
parameters[:,0] = 0.3619 #general amplitude to be scaled
parameters[:,1] = (1025.0/2048)*phase_bins #general mean
parameters[:,2] = widths_gaussian #independent variable

#===============================================================================
# Finite Scintle Effect
# ==============================================================================

B = 60.0 #approximate frequency range for each channel in MHz - subaverage 4 of about 12.5 MHz each
D = 0.64 #approximate distance to the pulsar in kpc
nt = 0.2 #filling factor over time
nv = 0.2 #filling factor over bandwidth
c_1 = 1.16 #constant for uniform scattering media in fse
vel_cons = 2.53e4 #km/sec velocity constant in relationship between pulsar
#perpendicular velocity and delta tau_d
V = 18.9 #km/sec perpendicular velocity of the pulsar
