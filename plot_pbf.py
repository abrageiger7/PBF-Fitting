import numpy as np
import pickle
from fit_functions import *
import matplotlib.ticker as tick
from profile_class import Profile_Fitting
from astropy.time import Time
from scipy.stats import pearsonr
import os

time = np.linspace(0,1,phase_bins,endpoint=False)

plt.style.use('dark_background')

plt.figure(1)

#fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (6,4))
fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (5,4))
plt.rc('font', family = 'serif')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')

label1 = r'Extended Medium $\beta$ = 3.667'
label2 = r'Thin Screen $\beta$ = 3.667'
label3 = 'Decaying Exponential'

extended_best_tau_o = 71.0
thin_best_tau_o = 62.0
exp_best_tau_o = 105.0

# extended medium case
ua_pbfs = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']

tau_values = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']

betas = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/beta_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['betas']

ind_beta1 = np.where((betas == 3.667))[0][0]
ind_tau1 = find_nearest(tau_values[ind_beta1], extended_best_tau_o)[1][0][0]
tau1 = find_nearest(tau_values[ind_beta1], extended_best_tau_o)[0]
print(tau1)

pbf1 = ua_pbfs[ind_beta1][ind_tau1]

# thin screen case
unith_pbfs = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS={phase_bins}.npz')['pbfs_unitheight']

tau_values_start = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS={phase_bins}.npz')['tau_mus']

betas = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS={phase_bins}.npz')['betas']
zetas = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/thin_screen_pbfs|PHASEBINS={phase_bins}.npz')['zetas']

ind_zeta2 = np.where((zetas == 0.01))[0][0]
ind_beta2 = np.where((betas == 3.667))[0][0]
ind_tau2 = find_nearest(tau_values_start[ind_beta2][ind_zeta2], thin_best_tau_o)[1][0][0]
tau2 = find_nearest(tau_values_start[ind_beta2][ind_zeta2], thin_best_tau_o)[0]
print(tau2)

pbf2 = unith_pbfs[ind_beta2][ind_zeta2][ind_tau2] / trapz(unith_pbfs[ind_beta2][ind_zeta2][ind_tau2])

ua_pbfs = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['pbfs_unitarea']

tau_values = np.load(f'/Users/abrageiger/Documents/research/projects/pbf_fitting/exp_pbf_data_unitarea|PHASEBINS={phase_bins}.npz')['taus_mus']

ind_tau3 = find_nearest(tau_values, exp_best_tau_o)[1][0][0]
tau3 = find_nearest(tau_values, exp_best_tau_o)[0]
print(tau3)

pbf3 = ua_pbfs[ind_tau3]

# axs.flat[0].plot(time, pbf1, color = 'grey', linewidth = 2, label = label1)
#
# axs.flat[0].plot(time, pbf2, color = 'g', linewidth = 2, label = label2)
#
# axs.flat[0].plot(time, pbf3, color = 'steelblue', linewidth = 2, label = label3)
#
# axs.flat[1].plot(time, pbf1, color = 'grey', linewidth = 2, label = label1)
#
# axs.flat[1].plot(time, pbf2, color = 'g', linewidth = 2, label = label2)
#
# axs.flat[1].plot(time, pbf3, color = 'steelblue', linewidth = 2, label = label3)
#
# axs.flat[1].set_xscale('log')
# axs.flat[1].set_yscale('log')
#
# axs.flat[0].set_xticks([])
# axs.flat[0].set_yticks([])
# axs.flat[0].set_xlabel('Time (arbitrary)')
# axs.flat[0].set_ylabel('PBF (normalized)')
# axs.flat[1].legend()

axs.plot(time, pbf1, color = 'grey', linewidth = 2, label = label1)

axs.plot(time, pbf2, color = 'g', linewidth = 2, label = label2)

axs.plot(time, pbf3, color = 'steelblue', linewidth = 2, label = label3)

axs.set_xticks([])
axs.set_yticks([])
axs.set_xlabel('Time (arbitrary)')
axs.set_ylabel('PBF (normalized)')
axs.legend()

fig.tight_layout()
plt.savefig('pbf_comparison.png', dpi = 300, bbox_inches='tight')
plt.savefig('pbf_comparison.pdf', bbox_inches='tight')

plt.close('all')

plt.figure(1)

phase_bins = init_data_phase_bins

pbf_to_wrap = np.load('zeta_0.0_beta_3.667_pbf.npy')

plt.plot(np.linspace(0,1,phase_bins,endpoint=False), pbf_to_wrap, label = 'Original PBF \n' + r'$\tau$ = ' + f'{np.round(calculate_tau(pbf_to_wrap)[1]/phase_bins, 4)}')

time = np.linspace(0,1,phase_bins//32,endpoint=False)

pbf_wrapped = np.zeros(phase_bins//32)

for i in range(32):
    pbf_wrapped += pbf_to_wrap[(i*phase_bins//32):((i+1)*phase_bins//32)]

spline = CubicSpline(time, pbf_wrapped)

plt.plot(np.linspace(0,1,phase_bins,endpoint=False), spline(np.linspace(0,1,phase_bins,endpoint=False)), label = 'Final PBF \n' + r'$\tau$ = ' + f'{np.round(calculate_tau(spline(np.linspace(0,1,phase_bins,endpoint=False)))[1]/phase_bins, 4)}')

plt.plot([],[], label = 'Wrap', color = 'k')

spline = CubicSpline(time, pbf_to_wrap[:phase_bins//32])

plt.plot(np.linspace(0,1,phase_bins,endpoint=False), spline(np.linspace(0,1,phase_bins,endpoint=False)), label = '0')

pbf_wrapped += pbf_to_wrap[:phase_bins//32]

for i in range(1,32):
    if (i==1) or (i==2) or (i==3) or (i==5) or (i==10) or (i==30):
        plt.plot(time, pbf_to_wrap[(i*phase_bins//32):((i+1)*phase_bins//32)], label = str(i))
plt.xlabel('Time (arbitrary)')
plt.ylabel('PBF')
plt.xticks([])
plt.yticks([])

plt.title('')

plt.legend()
plt.savefig('pbf_wrap_demo.pdf', bbox_inches='tight')
plt.savefig('pbf_wrap_demo.png', dpi = 300, bbox_inches='tight')
plt.show()
