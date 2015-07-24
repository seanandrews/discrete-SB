import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filename = 'data/blind2_fo.combo.noisy.image.fits'

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

# bins averages
nbins = 20
b = 0.05 + 0.05*np.arange(nbins)
a = np.roll(b, 1)
rin = 0.01/140.
a[0] = rin
cb = 0.5*(a+b)
 
# averge surface brightness in each bin
avg_sb = np.zeros_like(cb)
bareas = np.zeros_like(cb)
for i in range(nbins):
    avg_sb[i] = np.mean(data_image[(radius > a[i]) & (radius < b[i])])
    bareas[i] = np.pi*(b[i]**2-a[i]**2)

# scale to units of Jy per square arcsec
omega_beam = np.pi*(3600.**2)*hdr['BMAJ']*hdr['BMIN']/(4.*np.log(2.))
avg_sb /= omega_beam

# the truth!
rtruth  = np.logspace(-2, 0.1, num=100)
rc = 0.39285714
Ic = 0.0824975
SBtruth = Ic * (rc/rtruth)
SBtruth[rtruth > rc] = Ic * (rtruth[rtruth > rc]/rc)**(-4.)

plt.axis([0.01, 1.5, 1e-4, 1e1])
plt.loglog(rtruth, SBtruth, '-k', cb, avg_sb, 'ob')
plt.savefig('profile.png') 
