import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
import matplotlib.ticker as tick
from astropy.time import Time
import pickle
import sys

from profile_class import Profile_Fitting
from fit_functions import *

pbf_type = 'zeta'
intrinsic_shape = 'modeled'

#calculate the total duration of observation
with open('j1903_data.pkl', 'rb') as fp:
    data_dict = pickle.load(fp)

mjd_strings = list(data_dict.keys())
mjds = np.zeros(np.size(mjd_strings))
for i in range(np.size(mjd_strings)):
    mjds[i] = data_dict[mjd_strings[i]]['mjd']

dur_collect = 0
for i in mjd_strings:
    dur_collect += data_dict[i]['dur']

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

lband_data = np.zeros((len(freq_keys),2048))

ind = 0
for i in lband_avgs.keys():
    lband_data[ind] = lband_avgs[i]
    ind += 1

data_time_averaged = np.zeros((len(freq_keys),256))
ind = 0
for i in lband_data:
    data_time_averaged[ind] = time_average(i,256)
    ind += 1

index_care = 0
key_care = freq_keys[index_care]

if __name__ == '__main__':

    ua_pbfs = {}
    tau_values = {}

    ua_pbfs['beta'] = np.load(f'beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']
    ua_pbfs['zeta'] = np.load(f'zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']
    ua_pbfs['exp'] = np.load(f'exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']

    tau_values['beta'] = np.load(f'beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']
    tau_values['zeta'] = np.load(f'zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']
    tau_values['exp'] = np.load(f'exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']

    betas = np.load(f'beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['betas']
    zetas = np.load(f'zeta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['zetas']

    fitting_profiles = ua_pbfs
    intrinsic_fwhms = -1

    mjd = 0.0
    data = data_time_averaged
    freqs = lband_freqs
    dur = dur_collect

    prof = Profile_Fitting(mjd, data, freqs, dur, intrinsic_shape, betas, zetas, fitting_profiles, tau_values, intrinsic_fwhms, subaverage=False)

    datab = prof.fit(index_care, pbf_type)

    print(f'Frequency = {prof.freq_round}')

    print('Before Add to Baseline')
    print(f'Best Fit Zeta = ')
    print(datab[pbf_type], '-', datab['zeta_low'], '+', datab['zeta_up'])

    print('After Add to Baseline')

    prof = Profile_Fitting(mjd, data+(.01*np.max(data[-1])), freqs, dur, intrinsic_shape, betas, zetas, fitting_profiles, tau_values, intrinsic_fwhms, subaverage=False)

    datab = prof.fit(index_care, pbf_type)

    print(datab[pbf_type], '-', datab['zeta_low'], '+', datab['zeta_up'])

    print('After Subtract From Baseline')

    prof = Profile_Fitting(mjd, data-(.01*np.max(data[-1])), freqs, dur, intrinsic_shape, betas, zetas, fitting_profiles, tau_values, intrinsic_fwhms, subaverage=False)

    datab = prof.fit(index_care, pbf_type)

    print(datab[pbf_type], '-', datab['zeta_low'], '+', datab['zeta_up'])
