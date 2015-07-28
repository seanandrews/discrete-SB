import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from astropy.io import fits
from deprojectVis import deprojectVis
from discreteModel import discreteModel

# I/O
filename = 'data/blind2_fo.combo.noisy.image.fits'

# radial bins setup
nbins = 20
b = 0.05 + 0.05*np.arange(nbins)
a = np.roll(b, 1)
rin = 0.1/140.
a[0] = rin
cb = 0.5*(a+b)
bins = rin, b


# - Fiducial guess for SB(r) based on synthesized image

# load synthesized image data
hdulist = fits.open(filename)
data_image = np.squeeze(hdulist[0].data)
hdr = hdulist[0].header

# define grid/coordinate system
RA_cen  = hdr['CRVAL1']
DEC_cen = hdr['CRVAL2']
dRA  = hdr['CDELT1']
dDEC = hdr['CDELT2']
nRA  = hdr['NAXIS1']
nDEC = hdr['NAXIS2']
RA   = RA_cen + dRA * (np.arange(nRA) - (hdr['CRPIX1']-1))
DEC  = DEC_cen + dDEC * (np.arange(nDEC)  - (hdr['CRPIX2']-1))
RAo, DECo = np.meshgrid(RA-RA_cen, DEC-DEC_cen)

# lots of important stuff happens...

# radial profile
radius = 3600.*np.sqrt(RAo**2 + DECo**2)

# average surface brightness in each bin
guess_sb = np.zeros_like(cb)
for i in range(len(cb)):
    guess_sb[i] = np.mean(data_image[(radius > a[i]) & (radius < b[i])])

# scale to units of Jy per square arcsec
omega_beam = np.pi*(3600.**2)*hdr['BMAJ']*hdr['BMIN']/(4.*np.log(2.))
guess_sb /= omega_beam


# truth
rtruth = np.logspace(np.log10(rin), 0.36, num=500)
Ic = 0.0824975
rc = 55./140.
SBtruth = Ic * (rc/rtruth)
SBtruth[rtruth > rc] = Ic * (rtruth[rtruth > rc]/rc)**(-4.)
#tru_sb = Ic * (rc/cb)
#tru_sb[cb > rc] = Ic * (cb[cb > rc]/rc)**(-4.)



# initialize walkers
ndim, nwalkers, nthreads = nbins, 80, 8
scl = 0.5*np.ones_like(guess_sb)
scl[cb < 3600.*hdr['BMAJ']] = 0.9
p0 = [guess_sb*(1.+scl*np.random.uniform(-1, 1, ndim)) for i in range(nwalkers)]

for i in range(nwalkers):
    mono = False
    indx = 0
    while (mono == False):
        ptrial = guess_sb*(1.+scl*np.random.uniform(-1, 1, ndim))
        mono = np.array_equal(np.sort(ptrial), ptrial[::-1])
        indx += 1
    p0[i][:] = ptrial


# plot initial ball of guesses for surface brightness profile
plt.axis([0.01, 1.5, 1e-4, 1e1])
plt.loglog(radius, data_image/omega_beam, '.y', alpha=0.01)
plt.loglog(rtruth, SBtruth, 'k-', cb, guess_sb, 'oc')
for i in range(nwalkers):
    plt.loglog(cb, p0[:][i], '-r', alpha=0.1)
plt.xlabel('radius [arcsec]')
plt.ylabel('surface brightness [Jy/arcsec**2]')
plt.savefig('blind2_fo.profile.png')
plt.close()



# load the "data" visibilities
data = np.load('data/blind2_fo.340GHz.vis.npz')
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
ftheta = incl, PA, offset, guess_sb
uvsamples = u, v
fmodelvis = discreteModel(ftheta, uvsamples, bins)
mindata = u, v, fmodelvis.real, fmodelvis.imag
fvis = deprojectVis(mindata, incl=incl, PA=PA, offset=offset)
frho, freal, fimag = fvis

# truth
ttheta = incl, PA, offset, SBtruth
bins_true = rin, rtruth
tmodelvis = discreteModel(ttheta, uvsamples, bins_true)
mindata = u, v, tmodelvis.real, tmodelvis.imag
tvis = deprojectVis(mindata, incl=incl, PA=PA, offset=offset)
trho, treal, timag = tvis

plt.axis([0, 2000., -0.025, 0.15])
plt.plot(drho, dreal, '.y', alpha=0.01)
# loop through initialized walkers
for i in range(nwalkers):
    guess_sb = p0[:][i]
    gtheta = incl, PA, offset, guess_sb
    gmodelvis = discreteModel(gtheta, uvsamples, bins)
    gindata = u, v, gmodelvis.real, gmodelvis.imag
    gvis = deprojectVis(gindata, incl=incl, PA=PA, offset=offset)
    grho, greal, gimag = gvis
    plt.plot(grho, greal, '.r', alpha=0.008)
plt.plot(frho, freal, '.c')
plt.plot(trho, treal, '.k')
plt.xlabel('deprojected baseline length [klambda]')
plt.ylabel('real visibility [Jy]')
plt.savefig('blind2_fo.visprof.png')


# save the initial ball of guesses
np.savez('p0', p0=p0)
