import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
from astropy.io import fits
from lnprob import lnprob
from guess_inits import guess_inits
import emcee

# data files
prefix  = 'blind2_fo'
imfile  = prefix+'.combo.noisy'
visfile = prefix+'.340GHz'


# bin locations
nbins = 20
b = 0.05 + 0.05*np.arange(nbins)
a = np.roll(b, 1)
rin = 0.01/140.
a[0] = rin
cb = 0.5*(a+b)
bins = rin, b


# guess initial surface brightnesses in these bins, using synthesized image
im_in = 'data/'+imfile+'.image.fits'
guess_sb = guess_inits(im_in, bins)


# initialize walkers
ndim, nwalkers, nthreads = nbins, 80, 8
p0 = [guess_sb*(1.+0.5*np.random.uniform(-1, 1, ndim)) for i in range(nwalkers)]


# plot initial ball of guesses for surface brightness profile
plt.axis([0.01, 1.5, 1e-4, 1e1])
plt.loglog(cb, guess_sb, 'ob')
for i in range(nwalkers):
    plt.loglog(cb, p0[:][i], '-r', alpha=0.1)
plt.xlabel('radius [arcsec]')
plt.ylabel('surface brightness [Jy/arcsec**2]')
plt.savefig(prefix+'.profile.png')
plt.close()


# load the "data" visibilities
data = np.load('data/'+visfile+'.vis.npz')
freq = 340e9
u = 1e-3*data['u']*freq/2.9979e8
v = 1e-3*data['v']*freq/2.9979e8
real = data['Re']
imag = data['Im']
wgt = 10000.*data['Wt']


# deproject
incl = 0.
PA = 0.
offset = np.array([0., 0.])
indata = u, v, real, imag
dvis = deprojectVis(indata, incl=incl, PA=PA, offset=offset)
drho, dreal, dimag = dvis

# initial guess visibilities
ftheta = incl, PA, offset, avg_sb
uvsamples = u, v
fmodelvis = discreteModel(ftheta, uvsamples, bins)
mindata = u, v, fmodelvis.real, fmodelvis.imag
fvis = deprojectVis(mindata, incl=incl, PA=PA, offset=offset)
frho, freal, fimag = fvis

# binned truth
ttheta = incl, PA, offset, tru_sb
tmodelvis = discreteModel(ttheta, uvsamples, bins)
mindata = u, v, tmodelvis.real, tmodelvis.imag
tvis = deprojectVis(mindata, incl=incl, PA=PA, offset=offset)
trho, treal, timag = tvis



plt.axis([0, 2000., -0.025, 0.15])
plt.plot(drho, dreal, '.k', alpha=0.01)
# loop through initialized walkers
for i in range(nwalkers):
    guess_sb = p0[:][i]
    gtheta = incl, PA, offset, guess_sb
    gmodelvis = discreteModel(gtheta, uvsamples, bins)
    gindata = u, v, gmodelvis.real, gmodelvis.imag
    gvis = deprojectVis(gindata, incl=incl, PA=PA, offset=offset)
    grho, greal, gimag = gvis
    plt.plot(grho, greal, '.r', alpha=0.008)
plt.plot(frho, freal, '.b', alpha=0.01)
plt.plot(trho, treal, '.g', alpha=0.01)
plt.xlabel('deprojected baseline length [klambda]')
plt.ylabel('real visibility [Jy]')
plt.savefig('visprof.png')

