import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt


# radial bins setup
nbins = 20
b = 0.05 + 0.05*np.arange(nbins)
a = np.roll(b, 1)
rin = 0.1/140.
a[0] = rin
cb = 0.5*(a+b)
bins = rin, b


Ic = 0.0824975
rc = 55./140.
SBtruth = Ic * (rc/cb)
SBtruth[cb > rc] = Ic * (cb[cb > rc]/rc)**(-4.)


ndim, nwalkers, nthreads = nbins+4, 80, 8


chain = np.load('chain.npy')
fchain = chain.reshape(-1, ndim)
trial  = np.arange(np.shape(chain)[1])/1000.

# plot chain progress for SB values
fig = plt.figure(1)
for idim in np.arange(nbins):
    for iw in np.arange(nwalkers):
        plt.subplot(5,4,idim+1)
        plt.plot(trial, chain[iw, :, idim+4], 'b')
        plt.plot(trial, SBtruth[idim]*np.ones_like(trial), 'r')
fig.savefig('chain_sbs.png')
fig.clf()

# plot chain progress for SB values
fig = plt.figure(1)
for idim in np.arange(4):
    for iw in np.arange(nwalkers):
        plt.subplot(4,1,idim+1)
        plt.plot(trial, chain[iw, :, idim], 'b')
        plt.plot(trial, np.zeros_like(trial), 'r')
fig.savefig('chain_geo.png')
fig.clf()




