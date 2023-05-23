import numpy as np
import matplotlib.pyplot as plt
import convolved_pbfs as conv

intrinsic = np.load('Approximate Intrinsic Pulse.npy')
widths = conv.widths
betaselect = conv.betaselect
time = conv.time
pbf_data_unitarea = conv.pbf_data_unitarea

plt.xlabel('Pulse Period')
plt.ylabel('Integrated Intensity')
plt.plot(time, intrinsic)

t = np.linspace(0, 2048, 2048)
convolved_w_dataintrins = np.zeros((np.size(betaselect), np.size(widths), 2048))

data_index0 = 0
for i in pbf_data_unitarea:
    data_index1 = 0
    for ii in i:
        ua_intrinsic_gauss = intrinsic
        new_profile = (np.fft.ifft(np.fft.fft(ua_intrinsic_gauss)*np.fft.fft(ii)))
        new_profile = new_profile.real #take real component of convolution
        convolved_w_dataintrins[data_index0][data_index1] = new_profile
        data_index1 = data_index1+1
    data_index0 = data_index0+1

for i in range(10):
     plt.figure(i)
     plt.plot(convolved_w_dataintrins[i][i*5])
     plt.show()
