import numpy as np
from astropy.io import ascii
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
p0 = np.load('chain.npy')[:, -1, :]


# - SAMPLE POSTERIOR

# initialize sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, \
                                args=[data, bins])

# emcee sampler; track time
iter = 100
tic0 = time.time()
sampler.run_mcmc(p0, iter)
toc = time.time()

# save the results in a binary file
np.save('contchain', sampler.chain)

# grab previous notes
notesy = ascii.read("notes.dat")
liter = notesy['col3'][-1]
ltime = notesy['col1'][-1]

# add a note
f = open("notes.dat", "a")
f.write("{0:f}   {1:f}   {2:f}\n".format(ltime+(toc-tic0)/3600., \
                                         (toc-tic0)/3600., \
                                         liter+iter))
f.close()

# loop to keep running!
for i in range(49):
    tic = time.time()
    sampler.run_mcmc(sampler.chain[:, -1, :], iter)
    toc = time.time()
    np.save('contchain', sampler.chain)
    f = open("notes.dat", "a")
    f.write("{0:f}   {1:f}   {2:f}\n".format(ltime+(toc-tic0)/3600., \
                                             (toc-tic)/3600., \
                                             liter+(2+i)*iter))
    f.close()

# concatenate chains
cchain = np.concatenate((np.load('chain.npy'), sampler.chain), 1)
print(np.shape(cchain))
np.save('combined_chain', cchain)
