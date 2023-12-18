import corner
import numpy as np
import matplotlib.pyplot as plt

samples = np.random.rand(100,9)
labels = [r'$A_1$', r'$\phi_1$', r'$W_1$', r'$\phi_2$', r'$W_2$', r'$A_3$' \
, r'$\phi_3$', r'$W_3$', r'$\tau$']

plt.figure(1)
fig = corner.corner(samples, bins=50, color='dimgrey', smooth=0.6, \
plot_datapoints=False, plot_density=True, plot_contours=True, \
fill_contour=False, show_titles=True, labels = labels)
plt.savefig('corner_plot_test.pdf')
plt.close('all')
