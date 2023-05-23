# Reads in npz files outputted by gammaD_from_pde_sw.py 
# for different values of beta. 
#
# Extracts the pulse broadening function and metadata and plots
# all on the same frame

import numpy as np
from numpy import zeros, size
from numpy import interp
import matplotlib.pyplot as plt

import sys
import datetime
import glob

import calcticks as ct

script_name = sys.argv[0]
basename = script_name.split('.')[0]
now = datetime.datetime.now()
plotstamp = 'jmc   ' + basename + '  ' + str(now).split('.')[0]

npzfiles = glob.glob('PBF*zeta_0.000*.npz')
nfiles = np.size(npzfiles)

#input('hit return')

for n, npzfile in enumerate(npzfiles):
    print(npzfile)
    indata = np.load(npzfile, allow_pickle=True)    # True needed to get dicts

    tvec = indata['tvec']
    pbf = indata['PBF']
    GDnu2 = indata['GDnu2']
    bbeta = indata['bb']
    dvs = indata['dvs']
    vsvec = indata['vsvec']
    inargs = indata['args']
    zeta = inargs.all().zeta
    sftype = inargs.all().sftype


    if n==0:
        # use lists to handle input that has different length tvec's:
        tvecarray2 = []
        pbfarray2 = []
        GDnu2array2 = []
        betavec = zeros(nfiles)
        dvsvec = zeros(nfiles)
        zetavec = zeros(nfiles)
        sftypevec = zeros(nfiles, dtype=str)

    # Recalculate tvec because earlier versions of gammaD_from_pde_sw.py
    # calculated it differently and seems to be off by x2 (too large)
    dt_spline = 1 / (2.*vsvec.max())
    # factor of 2\pi makes widths in vs-space and t-space in bin units reasonable

    dt_spline *= 2*np.pi
    Nt2 = vsvec.size
    tvec = np.linspace(0, 2*Nt2-1,2*Nt2-1, endpoint=True) * dt_spline


    tvecarray2.append(tvec)
    pbfarray2.append(pbf)
    GDnu2array2.append(GDnu2[0:size(vsvec)])
    betavec[n] = bbeta
    dvsvec[n] = dvs
    zetavec[n] = zeta
    sftypevec[n] = sftype

# indices to sort in beta:
beta_inds = np.argsort(betavec)
beta_sort = betavec[beta_inds]
zeta_sort = zetavec[beta_inds]
    
# Plot PBFs for selected values of beta

zetaselect = np.array([0.000])

betaselect = np.array([3.1, 3.5, 3.667, 3.8, 3.9, 3.95, 3.975, 3.99, 3.995, 3.9975, 3.999, 3.99999])
#betaselect = np.array([3.1, 3.2, 3.3, 3.4, 3.5, 3.667, 3.8, 3.9, 3.95, 3.975, 3.99999])

tvecplotmax = 30

fig = plt.figure()
ax = fig.add_subplot(111)
    
for n, beta in enumerate(beta_sort):
    #beta = beta_sort[n]
    if beta in betaselect:
        #pbf = pbf_sort[n] 
        pbf = pbfarray2[beta_inds[n]]
        pbf /= pbf.max()                # unit maximum
        tvec = tvecarray2[beta_inds[n]]
        if beta == 3.667:
            label = r'$\rm 11/3$'
            #label = r'$\rm %5.3f$'%(beta)
        elif beta == 3.95:
            label = r'$\rm %4.2f$'%(beta)
        elif beta == 3.975:
            label = r'$\rm %5.3f$'%(beta)
        elif beta == 3.99:
            label = r'$\rm %4.2f$'%(beta)
        elif beta == 3.995:
            label = r'$\rm %5.3f$'%(beta)
        elif beta == 3.9975:
            label = r'$\rm %5.4f$'%(beta)
        elif beta ==  3.999:
            label = r'$\rm %5.3f$'%(beta)
        elif beta < 3.99:
            label = r'$\rm %4.2f$'%(beta)
        else:
            label = r'$\rm %3.0f$'%(beta)

        tinds = np.where(tvec < tvecplotmax)
        plt.plot(tvec[tinds], pbf[tinds], '-', lw=2, label=label)

        ntmax = pbf.argmax()
        pbfmax = pbf.max()
        pbfinds = pbf[ntmax:].argsort()
        pbfsort = pbf[ntmax:][pbfinds]
        tvecsort = tvec[ntmax:][pbfinds]
        t_e = interp((pbfmax/np.e,), pbfsort, tvecsort)[0]
  
        print('%3d  %5.2f   %6.3f   %6.3f   %6.3f'%(n, zeta, 
            tvec[ntmax], t_e, t_e/tvec[ntmax]))

ndecades = 6
plt.yscale('log')
ylocs, ytics = ct.calcticks_log(tlog1=-ndecades, tlog2=0, dt=1) 
plt.axis(ymin=0.8*10**(-ndecades))
#plt.axis(xmax = 30)
plt.yticks(ylocs, ytics)
plt.xlabel(r'$\rm t \ / \ t_c$', fontsize=16)
plt.ylabel(r'$\rm PBF(t)$', fontsize=16)
plt.legend(loc=(0.175,0.75), title=r'$\rm \beta$', ncol=4)
plt.annotate(plotstamp, xy=(0.6, 0.02), xycoords='figure fraction', ha='left', va='center', fontsize=5)
plt.show()

plotfile = basename + '_' + str(now.microsecond)+'.pdf'
plt.savefig(plotfile)
input('hit return')
plt.close('all')
