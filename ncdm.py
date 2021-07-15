import numpy as np
from scipy.optimize import root

# For detailed explanations, please see the publication Stücker et al (2021)
# ToDo: - add some documentation
#       - add function to fit power spectrum surpressions
#       - add a notebook with examples

#========================== Some Helper Functions ===========================#

def get_mscales(beta, mhm=1., mode="halo"):
    if mode=="halo":
        mu = np.array((0.2651, 1.638, 16.51))
        nu = np.array((0.3656, -0.0994, -0.9466))
    elif mode =="satellite":
        mu = np.array((0.1259, 1.134, 20.52))
        nu = np.array((0.4963, -0.0110, -1.1231))
    else:
        raise ValueError("Unknown mode %s, please use either mode='halo' or mode='satellite'" % mode)
        
    return mhm*mu*beta**nu

def mass_function_abc(m, a, b, c):
    return (1 + (a / m)**b )**c

def supp_mscale(a, b, c, frac=0.5):
    return a / (frac**(1./c) - 1)**(1./b)

def mscales_to_abc(m20, m50, m80):
    def equations(par):
        a, b, c =  10**par[0], par[1], par[2]

        ms = supp_mscale(a, b, c, frac=np.array((0.2,0.5,0.8)))
        
        eq = np.log10(ms) - np.log10([m20, m50, m80])

        return eq

    p0 = (np.log10(m50), 2., -1.)
    res =  root(equations, p0, method="lm")

    return 10**res.x[0], res.x[1], res.x[2]

#====================== Main Functions for use cases ========================#

def mass_function_beta_mhm(m, beta=2., mhm=1., mode="halo"):
    m20,m50,m80 = get_mscales(beta=beta, mhm=mhm, mode=mode)
    a,b,c = mscales_to_abc(m20,m50,m80)
    
    return mass_function_abc(m, a,b,c)