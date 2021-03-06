import numpy as np

def deprojectVis(data, incl=0., PA=0., offset=[0., 0.], wsc=1.):

    # - read in, parse data
    u, v, real, imag = data

    # - convert keywords into relevant units
    inclr = np.radians(incl)
    PAr = np.radians(PA)
    offr = 1e3*offset*np.pi/(180.*3600.)

    # - change to an appropriate coordinate system
    up = (u * np.cos(PAr) + v * np.sin(PAr)) * np.cos(inclr)
    vp = (-u * np.sin(PAr) + v * np.cos(PAr))
    rhop = np.sqrt(up**2 + vp**2)

    # - phase shifts
    realp = real*np.cos(-2.*np.pi*(offr[0]*u+offr[1]*v))
    imagp = imag*np.sin(-2.*np.pi*(offr[0]*u+offr[1]*v))

    # package for return
    output = rhop, realp, imagp

    return output
