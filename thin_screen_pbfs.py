import numpy as np
import matplotlib.pyplot as plt
import math
from fit_functions import calculate_tau, stretch_or_squeeze, phase_bins, time_average, find_nearest
from scipy.interpolate import CubicSpline
from scipy.integrate import trapz

plt.ion()

# Cordes thin screen pbfs
# CAREFUL: time steps spaced logarithmically
pbf_data = np.load('generate_pbf_sets_Nbeta_12_Nli_10.npz', allow_pickle=True)

inner_scale = pbf_data['livec']
beta_values = pbf_data['betavec']
time_values = pbf_data['tvecarray']
pbf_array = pbf_data['pbfarray']
one_over_e_values = pbf_data['tevec']
pdf_tau_values = pbf_data['tavevec']

# because the time steps are spaced logarithmically, start with lots of points
large_phase_bins = 1000000
num_pbfwidth = 400

# for each pbf, calculate number of tau values
tau_values_set = np.linspace(0.1,500,num_pbfwidth)

# parameters to collect for each pbf
betas = beta_values
zetas = inner_scale
tau_values_collect = np.zeros((np.shape(pbf_array)[0], np.shape(pbf_array)[1], num_pbfwidth))
thin_screen_pbfs = np.zeros((np.shape(pbf_array)[0], np.shape(pbf_array)[1], num_pbfwidth, phase_bins))

# keep in memory from first loop for speed
rescaled_pbfs = np.zeros((np.shape(pbf_array)[0], np.shape(pbf_array)[1], phase_bins))
original_tau_values = np.zeros(np.shape(pbf_array))

reference_tau_scale = 50000.0

i = find_nearest(betas, 3.667)[1][0][0]
ii = find_nearest(zetas, 0)[1][0][0]

time = time_values[i][ii]
pbf = pbf_array[i][ii]
plt.plot(time, pbf)

spline = CubicSpline(time, pbf)

# large number of phase bins to accomodate log time spacing
time_linear = np.linspace(time[0], time[-1], large_phase_bins)
pbf_linear = spline(time_linear)
plt.plot(time_linear, pbf_linear)

tau = one_over_e_values[i][ii]
print(f"Scale factor = {reference_tau_scale/tau}")

rescaled_pbf = stretch_or_squeeze(pbf_linear, reference_tau_scale/tau)
plt.plot(time_linear, rescaled_pbf)

del(pbf_linear)
del(tau)
new_pbf = time_average(rescaled_pbf, phase_bins)
rescaled_pbfs[i][ii] = new_pbf
original_tau_values[i][ii] = calculate_tau(new_pbf)[0]
print(f"Final tau = {original_tau_values[i][ii]}")
plt.plot(np.linspace(time[0], time[-1], phase_bins), new_pbf)
plt.show()

np.save('zeta_0_beta_11_3_thin_screen_pbf.npy', new_pbf/np.max(new_pbf))

del(rescaled_pbf)
del(new_pbf)


for i in range(np.shape(pbf_array)[0]): #beta
     for ii in range(np.shape(pbf_array)[1]): #zeta

        time = time_values[i][ii]
        pbf = pbf_array[i][ii]

        spline = CubicSpline(time, pbf)

        # large number of phase bins to accomodate log time spacing
        time_linear = np.linspace(time[0], time[-1], large_phase_bins)
        pbf_linear = spline(time_linear)
        del(time_linear)

        tau = one_over_e_values[i][ii]
        print(f"Scale factor = {reference_tau_scale/tau}")

        rescaled_pbf = stretch_or_squeeze(pbf_linear, reference_tau_scale/tau)

        del(pbf_linear)
        del(tau)
        new_pbf = time_average(rescaled_pbf, phase_bins)
        rescaled_pbfs[i][ii] = new_pbf
        original_tau_values[i][ii] = calculate_tau(new_pbf)[0]
        print(f"Final tau = {original_tau_values[i][ii]}")

        del(rescaled_pbf)
        del(new_pbf)


for i in range(np.shape(pbf_array)[0]): #beta
     for ii in range(np.shape(pbf_array)[1]): #zeta

         pbf = rescaled_pbfs[i][ii]

         for iii in range(num_pbfwidth):

             str_sqz_pbf = stretch_or_squeeze(pbf, ...)
             thin_screen_pbfs[i][ii][iii] = str_sqz_pbf/trapz(str_sqz_pbf)
             tau_values_collect[i][ii][iii] = calculate_tau(str_sqz_pbf)
             del(str_sqz_pbf)

         del(pbf)

np.savez('thin_screen_pbfs', pbfs_unitarea = thin_screen_pbfs, betas = betas, zetas = zetas, tau_mus = tau_values_collect)
