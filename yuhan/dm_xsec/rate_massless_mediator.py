import sys, os
import numpy as np

import matplotlib.pyplot as plt
from scipy.special import erf, spherical_jn

# Parameters
hbarc = 0.2     # eV um
rho_T = 2.0e3   # Sphere density, kg/m^3
mAMU = 1.66e-27 # Neutron mass

# DM parameters
rhoDM = 0.3e9        # dark matter mass density, eV/cm^3

vmin = 5e-5      # minimum velocity to consider, natural units (c)
vesc = 1.815e-3  # galactic escape velocity
v0 = 7.34e-4     # v0 parameter from Zurek group paper
ve = 8.172e-4    # ve parameter from Zurek group paper

def f_halo(v):
    """
    DM velocity distribution in the Earth frame from Zurek group paper
    
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
    """
    Differential cross section given by Born approximation

    :param mx : DM mass in natural units
    :param phi: mediator mass
    :param q  : momentum transfer (array_like)
    :param point_charge: if True give point charge solution
                         if False include a uniform sphere form factor

    :return: dsigma/domega in natural units (array_like)
    """
    point_charge_sol = (4 * (mx**2) * (alpha**2) ) / ( (mphi**2 + q**2)**2 )
    if point_charge:
        return point_charge_sol
    else:
        form_factor = 3 * spherical_jn(n=1, z=q*R) / (q * R)
        return point_charge_sol * form_factor**2
    
def dsig_dq(dsigdomega, mx, alpha, q, vlist, R):

    ss = np.empty(shape=(vlist.size, q.size))
    ss_out = np.empty(shape=(vlist.size, q.size))

    for i, v in enumerate(vlist):
        p = mx * v
        dsigdq = ( 2 * np.pi * q / (p**2) ) * dsigdomega

        # Cut off contribution below detection
        # Edit 2023/05/19: now done in later stages of analysis
        # dsigdq[q < q_thr] = 0

        # Cut off unphysical large-q scattering
        dsigdq[q > 2 * p] = 0
        ss[i] = dsigdq

        # Events outside of sphere only (approximate)
        # From exact sol of Rutherford scattering
        # b = (alpha / 2 Ecm) * cot(theta / 2)
        # q = 2p * sin(theta / 2)
        Ecm = 0.5 * mx * v**2
        bmin = R
        qmax = 2 * p / np.sqrt(4 * Ecm**2 * bmin**2 / alpha**2 + 1)
        dsigdq[q > qmax] = 0
        ss_out[i] = dsigdq
        
    return ss, ss_out

def dR_dq(mx, mphi, alpha, q, vlist, R):
    # Differential cross section
    dsigdomega = dsig_domega_born(mx, mphi, alpha, q, point_charge=True)
    dsigdq, dsigdq_out = dsig_dq(dsigdomega, mx, alpha, q, vlist, R)
        
    int_vec = rhoDM / mx * vlist * f_halo(vlist)
    
    drdq, drdq_out = np.empty_like(q), np.empty_like(q)
    for i in range(q.size):
        drdq[i] = np.trapz( int_vec * dsigdq.T[i], x=vlist )
        drdq_out[i] = np.trapz( int_vec * dsigdq_out.T[i], x=vlist )
        
    conv_fac = hbarc**2 * 1e9 * 3e10 * 1e-8 * 3600  # natural units -> um^2/GeV, c [cm/s], um^2/cm^2, s/hr
    
    # Counts/hour/GeV
    return drdq * conv_fac, drdq_out * conv_fac

def calc_event_rate(R_um, mx_gev, alpha_t):
    R = R_um / hbarc       # Sphere radius, eV^-1
    N_T = 0.5 * ( 4/3 * np.pi * (R_um*1e-6)**3 ) * rho_T/mAMU # Number of neutrons

    mx = mx_gev * 1e9      # DM mass, eV
    alpha = alpha_t * N_T  # Total coupling

    if R_um < 1:
        q = np.logspace(3, 10, 1000) # eV
    else:
        q = np.logspace(5, 10, 1000)

    nvels = 2000
    vlist = np.linspace(vmin, vesc, nvels)

    drdq, drdq_out = dR_dq(mx, 0, alpha, q, vlist, R)

    # GeV; Counts/hour/GeV
    return q/1e9, drdq, drdq_out

if __name__ == "__main__":
    npts = 20    # Number of pts in parameter space
    # outdir = r"C:\Users\yuhan\work\microspheres\code\impulse\data\massless_mediator"
    outdir = r"/home/yt388/palmer_scratch/data/massless_mediator"
    if(not os.path.isdir(outdir)):
        os.mkdir(outdir)

    # R_um = 7.5    # Sphere radius, um
    # mx_gev = np.logspace(0, 12, npts)    # DM mass in GeV
    # alpha_t = np.logspace(-14, -6, npts) # Single neutron coupling

    # R_um = 0.75
    # mx_gev = np.logspace(-2, 11, npts)
    # alpha_t = np.logspace(-14, -6, npts)

    # R_um = 0.075   # nanospheres; 75 nm
    # mx_gev = np.logspace(-4, 10, npts)
    # alpha_t = np.logspace(-12, -4, npts)

    R_um = 0.0075   # nanospheres; 75 nm
    mx_gev = np.logspace(-6, 8, npts)
    alpha_t = np.logspace(-12, -4, npts)
    
    if R_um < 0.5:
        sphere_type = 'nanosphere'
    else:
        sphere_type = 'microsphere'

    print(f'Sphere radius = {R_um:.3f} um')

    for i, mx in enumerate(mx_gev):
        for j, alpha in enumerate(alpha_t):
            print(f'Working on ( M_x = {mx:.3e} GeV, alpha_t = {alpha:.3e} )')
            qq, drdq, drdq_out = calc_event_rate(R_um, mx, alpha)

            np.savez(outdir + f'/drdq_{sphere_type}_{R_um:.2e}_{mx:.5e}_{alpha:.5e}.npz', mx_gev=mx, alpha_t=alpha, q=qq, drdq=drdq, drdq_out=drdq_out)
