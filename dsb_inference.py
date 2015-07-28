import numpy as np
import os
import time
import sys
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


# - INITIALIZE MCMC

# initialize walkers
ndim, nwalkers, nthreads = nbins+4, 80, 8

# load the initial surface brightness guesses
p0 = np.load('p0.npz')['p0']


# - SAMPLE POSTERIOR

# create a file to store progress information
os.system('rm notes.dat')
f = open("notes.dat", "w")
f.close()

# initialize sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, \
                                args=[data, bins])

# emcee sampler; track time
iter = 100
tic0 = time.time()
sampler.run_mcmc(p0, iter)
toc = time.time()
print(toc-tic0)

# save the results in a binary file
np.save('chain', sampler.chain)

# add a note
f = open("notes.dat", "a")
f.write("{0:f}   {1:f}   {2:f}\n".format((toc-tic0)/3600., (toc-tic0)/3600., \
                                         iter))
f.close()

