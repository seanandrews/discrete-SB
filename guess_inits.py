import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from deprojectVis import deprojectVis
from discreteModel import discreteModel
import time
import sys
from lnprob import lnprob
import emcee


def guess_inits(filename, bins):

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

    # averge surface brightness in each bin
    rin, b = bins
    a = np.roll(b, 1)
    a[0] = rin
    cb = 0.5*(a+b)
    avg_sb = np.zeros_like(cb)
    for i in range(len(b)):
        avg_sb[i] = np.mean(data_image[(radius > a[i]) & (radius < b[i])])
        print(avg_sb[i])

    # scale to units of Jy per square arcsec
    avg_sb /= np.pi*(3600.**2)*hdr['BMAJ']*hdr['BMIN']/(4.*np.log(2.))

    # return guesses
    return avg_sb
