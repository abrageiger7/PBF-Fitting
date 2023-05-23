import fit_functions as fittin
import convolved_pbfs as conv
import numpy as np
import matplotlib.pyplot as plt
import math

convolved_profiles = conv.convolved_profiles
widths = conv.widths
gauss_widths = conv.widths_gaussian
betaselect = conv.betaselect
time = conv.time

num_phasebins = np.shape(convolved_profiles[0][0][0])
j1903_period = 0.0021499 * 1e6 #microseconds

def find_nearest(a, a0):
    '''Element in nd array `a` closest to the scalar value `a0`

    Preconditions: a is an n dimensional array and a0 is a scalar value

    Returns index, value'''
    idx = np.abs(a - a0).argmin()
    return a.flat[idx], np.where((a == a.flat[idx]))

def calculate_tau(profile):
    '''Calculates tau value of J1903 profile by calculating where it decays to the value of its max divided by e

    Preconditions: profile is a 1 dimensional array of length num_phasebins (2048)'''
    iii = np.copy(profile)
    #isolate decaying part of pbf
    maxim = np.where((iii == np.max(iii)))[0][0]
    iii[:maxim] = 0.0
    near = find_nearest(iii, np.max(iii)/math.e)
    tau = (near[1][0][0] - maxim) * j1903_period / num_phasebins #microseconds
    tau_index = near[1][0][0]
    tau_unconvert = near[0]
    return(tau, tau_index, tau_unconvert)

#tau values are the amount of time covered between the mode of the pbf and the mode/e value of the pbf

tau_values = np.zeros((np.size(betaselect), np.size(widths)))

pbf_data_freqscale = conv.pbf_data_freqscale

#plt.figure(1)

data_index = 0
for i in pbf_data_freqscale:
    data_index2 = 0
    for ii in i:
        tau_ii = calculate_tau(ii)
        tau_values[data_index][data_index2] = tau_ii[0]
        #plt.plot(ii)
        #plt.plot(tau_ii[1], tau_ii[2],'.')
        data_index2 = data_index2+1
    data_index = data_index+1
#plt.show()
np.save('tau_values', tau_values)

plt.figure(2)
im = plt.imshow(tau_values, origin = 'lower', aspect = '3.0')
plt.ylabel("Beta Values")
plt.xlabel("PBF Widths Rounded")
plt.title("Tau Values")
plt.yticks(ticks = np.arange(12), labels = betaselect)
pbf_ticks = np.zeros(10)
for i in range(10):
    pbf_ticks[i] = str(widths[i*5])[:4]
plt.xticks(ticks = np.linspace(0,50,num=10), labels = pbf_ticks)
cbar = plt.colorbar(im)
cbar.set_label('Tau in microseconds')
plt.savefig('tau_values_plot')
plt.show()

import convolved_exp as cexp
widths_exp_array = cexp.widths_exp_array

def calculate_tau_exp(profile):
    '''Calculates tau value of J1903 profile by calculating where it decays to the value of its max divided by e

    Preconditions: profile is a 1 dimensional array of length 2048'''
    iii = np.copy(profile)
    #isolate decaying part of pbf
    near = find_nearest(iii, np.max(iii)/math.e)
    #RESCALE THE TAU VALUES HERE FOR CONVOLUTION
    tau = (near[1][0][0]) * j1903_period / num_phasebins #microseconds
    tau_index = near[1][0][0]
    tau_unconvert = near[0]
    return(tau, tau_index, tau_unconvert)


tau_values_exp = np.zeros(np.size(widths))


data_index = 0
for i in widths_exp_array:
    tau_values_exp[data_index] = calculate_tau_exp(i)[0]
    data_index = data_index+1


#plot tau over pbf width for decaying exponential
#plt.xlabel("PBF Width (stretch factor)")
#plt.ylabel("Tau (microseconds)")
#plt.title("Tau versus PBF Width for Decaying Exponential Profiles")
#plt.plot(widths, tau_values_exp)
#plt.show()

#plot tau for a 2 specific values of beta varying over pbf width
#plt.xlabel("PBF Width (stretch factor)")
#plt.ylabel("Tau (microseconds)")
#plt.plot(widths, tau_values[-1], color = 'blue', label = 'Beta=4.0')
#plt.plot(widths, tau_values[0], color = 'orange', label = 'Beta='+str(betaselect[0]))
#plt.legend()
#plt.show()