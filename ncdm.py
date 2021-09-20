# This file offers functions to estimate the relative surpression in the
# halo and subhalo mass functions of non cold dark matter models with
# respect to a CDM reference case.
#
# For detailed explanations, please see the publication Stücker et al (2021)
# Code by Jens Stuecker

import numpy as np
from scipy.optimize import root


#========================== Some Helper Functions ===========================#

def get_mscales(beta, mhm=1., mode="halo"):
    """The scales where the mass function is surpressed by 20%, 50% and 80%
    
    beta : The slope parameter describing the power-spectrum surpression
    mhm :  The half-mode mass (describing the scale of the cut-off). Use
          mhm=1. to get outputs in units of the half-mode mass.
    mode : can be 'halo' or 'satellite'
          
    returns: np.array with M20, M50, M80 as elements. Output masses are in the 
             same units as the input parameter mhm.
    """
    if mode=="halo":
        mu = np.array((0.2651, 1.638, 16.51))
        nu = np.array((0.3656, -0.0994, -0.9466))
    elif mode =="satellite":
        mu = np.array((0.1259, 1.134, 20.52))
        nu = np.array((0.4963, -0.0110, -1.1231))
    else:
        raise ValueError("Unknown mode %s, please use either mode='halo' "
                         "or mode='satellite'" % mode)
        
    return mhm*mu*beta**nu

def mass_function_abc(m, a, b, c):
    """The parametrized surpression function of the halo mass function"""
    return (1 + (a / m)**b )**c

def supp_mscale(a, b, c, frac=0.5):
    """The mass scale where the surpression reaches a given value, given a,b,c
    
    a, b, c : parameters (a >= 0, b >= 0, c <= 0)

    frac : desired surpression factor
    
    returns : the mass M where f(M)=frac, units are as the parameter "a"
    """
    if (a < 0) | (b < 0) | (c > 0): 
        # This is just here, to tell the fitting function when it goes out of bounds
        return np.nan
    
    return a / (frac**(1./c) - 1.)**(1./b)

def mscales_to_abc(m20, m50, m80):
    """Uniquely maps the parameters m20, m50, m80 to a, b, c"""
    def equations(par):
        a, b, c =  10**par[0], par[1], par[2]

        ms = supp_mscale(a, b, c, frac=np.array((0.2,0.5,0.8)))
        
        eq = np.log10(ms) - np.log10([m20, m50, m80])

        return eq
    
    def fitfunc(frac, a, b, c):
        return supp_mscale(a, b, -c, frac=frac)
    
    p0 = (np.log10(m50), 2., -1.)
    res =  root(equations, p0, method="lm")

    return 10**res.x[0], res.x[1], res.x[2]

#====================== Main Functions for use cases ========================#

def mass_function_beta_mhm(m, beta=2., mhm=1., mode="halo"):
    """Evaluates the mass function of a ncdm model
    
    m : the masses where to evaluate the mass function, same units as mhm
    beta : the slope of the primordial power spectrum cutoff
    mhm : the half mode mass -- describing the scale of the cutoff
    mode : can be "halo" or "satellite"
    
    returns : the relative suppression of the ncdm to the cdm mass function
    """
    
    m20,m50,m80 = get_mscales(beta=beta, mhm=mhm, mode=mode)
    a,b,c = mscales_to_abc(m20,m50,m80)
    
    return mass_function_abc(m, a,b,c)