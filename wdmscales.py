import numpy as np

def alpha_wdm(omega_m=0.31, omega_x=0.27, h=0.7, gx=1.5, mx=1., verbose=False, mode="schneider"):
    #alpha is in h/mpc
    if mode == "bode_music": # music accidentally uses omega_m instead of omega_x in this formula
        return 0.05 * (omega_m/0.4)**0.15 * (h/0.65)**1.3 * (mx/1.)**-1.15 * (1.5/gx)**0.29
    if mode == "bode_fixednu": # next to (A7) of the bode paper, this is what music was trying to use, I think
        return 0.05 * (omega_x/0.4)**0.15 * (h/0.65)**1.3 * (mx/1.)**-1.15 * (1.5/gx)**0.29
    elif mode == "bode_varynu": # This is the fitting formula in bode when allowing nu to vary
        return 0.048 * (omega_x/0.4)**0.15 * (h/0.65)**1.3 * (mx/1.)**-1.15 * (1.5/gx)**0.29
    elif (mode == "murgia") | (mode == "schneider"): # This is the formula used in the murgia paper arxiv:1704.07838, it originates from the schneider paper
        return 0.049 * (omega_x/0.25)**0.11 * (h/0.7)**1.22  * (mx/1.)**-1.11
    else:
        raise ValueError("Unknown mode %s" % mode)

def beta_gamma_map(beta, gamma, gammamap=-5., q=0.85, qb=0.5):
    betamap = beta * np.log((q**(0.5/gammamap) - 1. ) / (qb**(0.5/gammamap) - 1. ) )/ np.log((q**(0.5/gamma) - 1.)/(qb**(0.5/gamma) - 1.))
    return betamap, gammamap

def k_half_mode(alpha, beta, gamma):
    """finds the wavenumber where the transfer function is surpressed by 1/2
    """
    fraction = 0.5
    return 1./alpha * (fraction**(-1./gamma) - 1. )**(1./beta)

def half_mode_volume(alpha, beta, gamma):
    """The Lagrangian volume that is associated with the halfmode scale"""
    khm = k_half_mode(alpha, beta, gamma)
    lam = (2.*np.pi)/khm
    
    return 4.*np.pi/3. * (lam/2.)**3

def background_density(omega_m=0.31):
    """The mean matter density of the universe in (Msol/h)/(Mpc /h)**3"""
    parsec = 3.085677581491367e+16    # one parsec in SI units
    G = 6.6743e-11                    # Gravitational constant in SI units
    Msol = 1.98847e30                 # Solar mass in SI units
    
    H0_SI = 100e3 / (1e6*parsec)      # 100 km/s / Mpc in SI units
    rhocrit_SIh2 = 3.*H0_SI**2 / (8.*np.pi*G)
    
    rhocrit = rhocrit_SIh2 / Msol * (1e6*parsec)**3
    
    return omega_m * rhocrit

def half_mode_mass(alpha, beta, gamma, omgea_m=0.31):
    return half_mode_volume(alpha, beta, gamma) * background_density(omega_m=omgea_m)