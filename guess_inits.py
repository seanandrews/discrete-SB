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
plt.loglog(radius, data_image, '.r')

# bins averages
nbins = 40
b = np.linspace(0.1, 2.0, num=nbins)
a = np.roll(b, 1)
rin = 0.01/140.
a[0] = rin
cb = 0.5*(a+b)
 
# averge surface brightness in each bin
avg_sb = np.zeros_like(cb)
for i in range(nbins):
    avg_sb[i] = np.mean(data_image[(radius > a[i]) & (radius < b[i])])

plt.loglog(cb, avg_sb, 'ob')
plt.savefig('profile.png') 


#plt.figure()
#plt.imshow(data_image, origin='lower', cmap='bone_r')
#plt.show()
