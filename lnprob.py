import numpy as np
from discreteModel import discreteModel

def lnprob(p, data, bins):

    # unpack parameters
    incl = p[0]
    PA = p[1]
    offx = p[2]
    offy = p[3]
    sbs = p[4:]

    # priors
    # enforce positive surface brightnesses
    if (np.any(sbs) < 0.):
        return -np.inf

    # enforce monotonicity
    if (np.array_equal(np.sort(sbs), sbs[::-1]) == False):
        return -np.inf

    # geometry common sense
    if (incl < 0. or incl > 90. or PA < 0. or PA > 180.):
        return -np.inf


    # unpack data
    u, v, dreal, dimag, dwgt = data
    uvsamples = u, v


    # generate model visibilities
    pars = incl, PA, np.array([offx, offy]), sbs
    mvis = discreteModel(pars, uvsamples, bins)

    
    # compute a chi2 value
    chi2 = np.sum(dwgt*(dreal-mvis.real)**2 + dwgt*(dimag-mvis.imag)**2)


    # return a log-likelihood value
    return -0.5*chi2
