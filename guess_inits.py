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

plt.figure()
plt.imshow(data_image, origin='lower', cmap='bone_r')
plt.show()
