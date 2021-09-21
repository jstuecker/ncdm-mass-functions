# This file offers some helper functions to handle the power spectra
# of a non cold dark matter universe and the length and mass scales
#
# For detailed explanations, please see the publication Stücker et al (2021)
# Code by Jens Stuecker

import numpy as np

def transfer_ncdm(k, alpha, beta, gamma=5.):
    """The non-cold dark amtter transfer function as in Murgia(2018) and Stucker (2021)"""
    return (1. + (alpha*k)**beta)**(-gamma)

def alpha_wdm(mx=1., mode="schneider", omega_x=0.27, h=0.7, gx=1.5, verbose=False):
    """Parameter of the power spectrum suppression for a given WDM thermal relic mass
    
    Uses either the fit from Schneider (2012) or Bode (2001)

    mx : thermal relic mass in eV
    mode : can be mode="schneider" (recommended) or "bode_varynu" or "bode_fixednu"
    omega_x : dark matter density parameter
    h : redued hubble parameter
    gx : degeneracy factor (only used by mode = "bode_varynu")
    omega_m : matter density parameter
    
    returns : the parameter alpha in units of Mpc/h
    """
    if mode == "bode_fixednu": # next to (A7) of the bode paper
        return 0.05 * (omega_x/0.4)**0.15 * (h/0.65)**1.3 * (mx/1.)**-1.15 * (1.5/gx)**0.29
    elif mode == "bode_varynu": # This is the fitting formula in bode when allowing nu to vary
        return 0.048 * (omega_x/0.4)**0.15 * (h/0.65)**1.3 * (mx/1.)**-1.15 * (1.5/gx)**0.29
    elif mode == "schneider": # arxiv:1112.0330 Schneider (2012)
        return 0.049 * (omega_x/0.25)**0.11 * (h/0.7)**1.22  * (mx/1.)**-1.11
    else:
        raise ValueError("Unknown mode %s" % mode)

def alpha_beta_gamma_3_to_2_par(alpha, beta, gamma, gammamap=-5., kfit=None):
    """Maps a 3 parameter transfer function to a two parameter one, fixing gamma
    
    alpha, beta, gamma : 3 parameter description
    gammamap : fixed gamma value (in Stucker (2021) it is assumed 5)
    kfit : (optional) a set of kvalues where to fit the approximation
    
    returns : (alphamap, betamap)  -- the two parameter approximaton
    """
    
    from scipy.optimize import curve_fit

    if kfit is None:
        kfit = np.logspace(np.log10(1e-4/alpha), np.log10(1e4/alpha), 300)
    Tfit = transfer_ncdm(kfit, alpha, beta, gamma)
    
    (alphamap, betamap), cov = curve_fit(transfer_ncdm, kfit, Tfit, p0=(alpha, beta), bounds=(0, np.inf))
    
    return alphamap, betamap, gammamap

def k_half_mode(alpha, beta, gamma):
    """finds the wavenumber where the transfer function is surpressed by 1/2"""
    fraction = 0.5
    return 1./alpha * (fraction**(-1./gamma) - 1. )**(1./beta)

def alpha_from_k_half_mode(khalf, beta, gamma):
    """Calculates alpha, given the half-mode wavenumber"""
    return 1./khalf * (2.**(1./gamma) - 1. )**(1./beta)

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

def half_mode_mass(alpha, beta, gamma, omega_m=0.31):
    """Calculates the half mode mass in units of Msol/h"""
    return half_mode_volume(alpha, beta, gamma) * background_density(omega_m=omega_m)