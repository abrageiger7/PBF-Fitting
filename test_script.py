import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pypulse.singlepulse import SinglePulse
from scipy.integrate import trapz
from scipy import optimize
from scipy.interpolate import CubicSpline
import math
from pypulse.singlepulse import SinglePulse
import sys

plt.plot(np.linspace(0,10,11), np.linspace(0,10,11))
plt.savefig('test_plot.pdf')

np.save('test_data', np.linspace(0,10,11))
