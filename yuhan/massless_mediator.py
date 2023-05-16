import sys, os
import numpy as np

import matplotlib.pyplot as plt
from scipy.special import erf

# Parameters
hbarc = 0.2     # eV um
rho_T = 2.0e3   # Sphere density, kg/m^3
mAMU = 1.66e-27 # Neutron mass

# R_um = 5          # Sphere radius, um
R_um = 0.05         # nanospheres

R = R_um / hbarc  # Radius in natural units, eV^-1
N_T = 0.5 * ( 4/3 * np.pi * (R_um*1e-6)**3) * rho_T/mAMU # Number of neutrons

# q_thr = 0.15e9    # momentum threshold, eV
q_thr = 200e3

# DM parameters
rhoDM = 0.3e9        # dark matter mass density, eV/cm^3

vmin = 5e-5      # minimum velocity to consider, natural units (c)
vesc = 1.815e-3  # galactic escape velocity
v0 = 7.34e-4     # v0 parameter from Zurek group paper
ve = 8.172e-4    # ve parameter from Zurek group paper

def f_halo(v):
    """
    DM velocity distribution in the Earth frame
    
    :param v: input velocity (array-like)
    :return: velocity distribtuion (array-like)
    """
    N0 = np.pi**1.5 * v0**3 * ( erf(vesc/v0) - 2/np.sqrt(np.pi) * (vesc/v0) * np.exp(-(vesc/v0)**2))
    
    # v < (vesc - ve)
    f1 = np.exp( - (v+ve)**2 / v0**2 ) * (np.exp(4*v*ve / v0**2) - 1)
    # (vesc - ve) < v < (vesc + ve)
    f2 = np.exp( - (v-ve)**2 / v0**2 ) - np.exp(- vesc**2 / v0**2)

    f = np.zeros_like(v)
    g1 = v < (vesc - ve)
    g2 = np.logical_and( vesc-ve < v, v < vesc + ve)
    
    f[g1] = f1[g1]
    f[g2] = f2[g2]

    return f * np.pi * v * v0**2 / (N0 * ve)

def f_halo_dan(v):
    """
    I think this is the 1-D standard halo model but need to double check.
    See Eq. (2) of https://link.aps.org/doi/10.1103/PhysRevD.42.3572
    """
    N0 = np.pi**1.5 * v0**3 * ( erf(vesc/v0) - 2/np.sqrt(np.pi) * (vesc/v0) * np.exp(-(vesc/v0)**2))
    return 4 * np.pi * v**2 * np.exp(-v**2 / v0**2) / N0

def dsig_domega_born(mx, mphi, alpha, q, point_charge):
    point_charge_sol = (4 * (mx**2) * (alpha**2) ) / ( (mphi**2 + q**2)**2 )
    if point_charge:
        return point_charge_sol
    else:
        form_factor = 3 * spherical_jn(n=1, z=q*R) / (q * R)
        return point_charge_sol * form_factor**2
    
def dsig_dq(dsigdomega, mx, alpha, q, vlist, q_thr):

    ss = np.empty(shape=(vlist.size, q.size))

    for i, v in enumerate(vlist):
        p = mx * v
        dsigdq = ( 2 * np.pi * q / (p**2) ) * dsigdomega

        # Cut off contribution below detection threshold
        dsigdq[q < q_thr] = 0

        Ecm = 0.5 * mx * v**2
        # Events outside of sphere only
        bmin = 5e-4 / hbarc # 5 um radius
        qmax = 2 * mx * v / np.sqrt(4 * Ecm**2 * bmin**2 / alpha**2 + 1)
        dsigdq[q > qmax] = 0

        # Account for vmin at a given q
        dsigdq[q > 2 * mx * v] = 0
        
        ss[i] = dsigdq
        
    return ss

def dR_dq(mx, mphi, alpha, q, vlist, q_thr):
    # Differential cross section
    dsigdomega = dsig_domega_born(mx, mphi, alpha, q, point_charge=True)
    dsigdq = dsig_dq(dsigdomega, mx, alpha, q, vlist, q_thr)
        
    int_vec = rhoDM / mx * vlist * f_halo_dan(vlist)
    
    drdq = np.empty_like(q)
    for i in range(q.size):
        drdq[i] = np.trapz( int_vec * dsigdq.T[i], x=vlist )
        
    conv_fac = hbarc**2 * 1e9 * 3e10 * 1e-8 * 3600  # natural units -> um^2/GeV, c [cm/s], um^2/cm^2, s/hr
    
    # Counts/hour/GeV
    return drdq * conv_fac

if __name__ == "__main__":
    print(f'Sphere radius: {R_um} um. Threshold = {q_thr:.3f} eV')
    npts = 100

    q = np.logspace(5, 10, 1000) # eV
    nvels = 2000
    vlist = np.linspace(vmin, vesc, nvels)

    # mx_ev = np.logspace(0, 12, npts) * 1e9
    # alpha_tot = np.logspace(-14, -6, npts) * N_T
    mx_ev = np.logspace(-4, 10, npts) * 1e9
    alpha_tot = np.logspace(-12, -4, npts) * N_T

    event_rate = np.empty( (mx_ev.size, alpha_tot.size) )

    for i, mx in enumerate(mx_ev):
        for j, alpha in enumerate(alpha_tot):
            print(f'Working on ( {mx/1e9:.3f} GeV, {alpha:.3f} )')
            
            drdq = dR_dq(mx, 0, alpha, q, vlist, q_thr)
            event_rate[i][j] = np.trapz(drdq, q/1e9)

    outdir = r"C:\Users\yuhan\work\microspheres\code\impulse\data\massless_mediator"
    if(not os.path.isdir(outdir)):
        os.mkdir(outdir)
    np.savez(outdir + '\event_rate_200keV_100nm.npz', mx=mx_ev, alpha=alpha_tot, rate=event_rate)