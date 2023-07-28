import sys, os
import numpy as np

from rate_massless_mediator import *

def dR_dq_born(mx, mphi, alpha, q, vlist, R):
    dsigdomega = dsig_domega_born(mx, mphi, alpha, q, R, point_charge=False)

    ss = np.empty(shape=(vlist.size, q.size))
    for i, v in enumerate(vlist):
        p = mx * v
        dsigdq = ( 2 * np.pi * q / (p**2) ) * dsigdomega
        
        # Cut off unphysical large momentum transfer
        # Elastic scattering only
        dsigdq[q > 2 * p] = 0

        ss[i] = dsigdq
    
    int_vec = rhoDM / mx * vlist * f_halo(vlist)
    drdq = np.empty_like(q)
    for i in range(q.size):
        drdq[i] = np.trapz( int_vec * dsigdq.T[i], x=vlist )
        
    conv_fac = hbarc**2 * 1e9 * 3e10 * 1e-8 * 3600  # natural units -> um^2/GeV, c [cm/s], um^2/cm^2, s/hr

    # Counts/hour/GeV
    return drdq * conv_fac

def calc_event_rate_born(R_um, mphi, mx_gev, alpha_t):
    R = R_um / hbarc       # Sphere radius, eV^-1
    N_T = 0.5 * ( 4/3 * np.pi * (R_um*1e-6)**3 ) * rho_T / mAMU
    
    mx = mx_gev * 1e9      # DM mass, eV
    alpha = alpha_t * N_T 
    
    nq = 20000
    pmax = np.max([np.min([10 * vesc * mx, 10 * alpha / (R * vmin)]), 1e4])
    q_lin  = np.linspace(1, 2*pmax*1.1, nq)
    
    nvels = 2000
    vlist = np.linspace(vmin, vesc, nvels)
    
    drdq = dR_dq_born(mx, mphi, alpha, q_lin, vlist, R)
    
    # GeV, Counts/hour/GeV
    return q_lin/1e9, drdq

if __name__ == "__main__":
    npts = 20    # Number of pts in parameter space

    outdir = r"/home/yt388/palmer_scratch/data/born"
    if(not os.path.isdir(outdir)):
        os.mkdir(outdir)

    sphere_type = 'nanosphere'
    R_um = 0.0075   # nanospheres; 7.5 nm
    # mx_gev = np.logspace(-6, 1, npts)
    # alpha_t = np.logspace(-10, -4, npts)
    mx_gev = np.logspace(-7, 0, npts)
    alpha_t = np.logspace(-15, -8, npts)

    mphi = [1e4, 1e5]      # eV

    for m_phi in mphi:
        for i, mx in enumerate(mx_gev):
            for j, alpha in enumerate(alpha_t):
                qq, drdq = calc_event_rate_born(R_um, m_phi, mx, alpha)
                np.savez(outdir + f'/drdq_{sphere_type}_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{m_phi:.0e}.npz', mx_gev=mx, alpha_t=alpha, q=qq, drdq=drdq)

