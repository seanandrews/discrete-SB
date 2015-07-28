import numpy as np
import os
import time
import sys
from astropy.io import fits
from lnprob import lnprob
import emcee

# - DEFINITIONS

# bin locations
nbins = 20
b = 0.05 + 0.05*np.arange(nbins)
a = np.roll(b, 1)
rin = 0.01/140.
a[0] = rin
cb = 0.5*(a+b)
bins = rin, b

# load and re-pack the "data" visibilities
visdata = np.load('data/blind2_fo.340GHz.vis.npz')
freq = 340e9
u = 1e-3*visdata['u']*freq/2.9979e8
v = 1e-3*visdata['v']*freq/2.9979e8
real = visdata['Re']
imag = visdata['Im']
wgt = 10000.*visdata['Wt']
data = u, v, real, imag, wgt

# initialize walkers
ndim, nwalkers, nthreads = nbins+4, 80, 8

# load the initial surface brightness guesses
p0 = np.load('p0.npz')['p0']


# trial likelihood calculation
print(-2.*lnprob(p0[:][0], data, bins))
