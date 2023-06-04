import sys, os
import numpy as np

from scipy.optimize import minimize_scalar
from scipy.integrate import quad
import scipy.signal as sig
from scipy.special import erf
from scipy.interpolate import CubicSpline

from multiprocessing import Pool

## General Prameters
hbarc = 0.2      # eV um
rho_T = 2.0e3    # Sphere density, kg/m^3
mAMU  = 1.66e-27 # Neutron mass, kg

res   = 170e6    # detector resolution in eV
q_thr = 0.05e9   # momentum threshold, eV

## DM parameters
rhoDM = 0.3e9    # dark matter mass density, eV/cm^3
vmin  = 5e-5     # minimum velocity to consider, natural units (c)
vesc  = 1.815e-3 # galactic escape velocity
v0    = 7.34e-4  # v0 parameter from Zurek group paper
ve    = 8.172e-4 # ve parameter from Zurek group paper

## Functions
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
    I think this is the standard halo model but need to double check.
    See Eq. (2) of https://link.aps.org/doi/10.1103/PhysRevD.42.3572
    """
    N0 = np.pi**1.5 * v0**3 * ( erf(vesc/v0) - 2/np.sqrt(np.pi) * (vesc/v0) * np.exp(-(vesc/v0)**2))
    return 4 * np.pi * v**2 * np.exp(-v**2 / v0**2) / N0

def vtot(u, m_phi, R, alpha, point_charge=False):
    mR = m_phi * R
    
    u = np.asarray(u)
    ret = np.empty_like(u)
    
    if point_charge:
        if (mR > 0):
            ret = alpha * u * np.exp(-1 * m_phi / u)
        else:
            ret = alpha * u
        return ret
    
    # Divide the array of u=1/r into three cases
    neg     = (u <= 0)   # ill-defined
    outside = (u < 1/R)  # ouside sphere
    inside  = (u >= 1/R) # inside sphere
    
    ret[neg] = np.inf
    
    if(mR > 0):   # massive mediator
        ret[outside] = 3 * alpha/mR**3 * (mR * np.cosh(mR) - np.sinh(mR)) * np.exp(-m_phi/u[outside]) * u[outside]
        ret[inside] = 3 * alpha/mR**3 * (m_phi - u[inside]*(1 + mR)/(1+1./np.tanh(mR)) * (np.sinh(m_phi/u[inside])/np.sinh(mR)))
        
    else:         # massless mediator (alpha/r)
        ret[outside] = alpha * u[outside]
        ret[inside] = alpha/2 * (3/R - 1./(R**3 * u[inside]**2))

    return ret

def max_u_func(u, E, b, m_phi, R, alpha, point_charge):
    return np.log(np.abs(1 - (b*u)**2 - vtot(u, m_phi, R, alpha, point_charge) / E))

def max_u_numerical(E, b, m_phi, R, alpha, point_charge):    
    # Zero of the function closest to each `b`
    if not hasattr(b, '__iter__'):
        b = np.asarray([b])
        
    # Iteratively find the min value of the function
    max_u = np.empty_like(b)
    if point_charge:
        ulower, uupper = -8, 6
    else:
        ulower, uupper = -8, 8
    
    # Max u for each b
    for i, bb in enumerate(b):
        uu = np.logspace(ulower, uupper, 10000)
        count = 0
        converge = False
        
        while (not converge):
            ff = max_u_func(uu, E, bb, m_phi, R, alpha, point_charge)
            min_arg = np.argmin(ff)
            if ( ff[min_arg] < -15 ): # Func value smaller than exp(-15)
                converge = True
                max_u[i] = uu[min_arg]
            
            else:
                # Usually it converges very quickly (only 1 or 2 iterations are needed)
                if (count > 100):
                    print('Fail to converge after 100 iterations')
                    max_u[i] = uu[min_arg]
                    break

                elif (min_arg == 0 or min_arg == 9999):
                    print('Need to increase lower/upper limit')
                    max_u[i] = uu[min_arg]
                    break
                    
                else:
                    count += 1                
                    uu = np.linspace(uu[min_arg-1], uu[min_arg+1], 10000)
    return max_u

def integrand(rho, umax, E, b, m_phi, R, alpha, point_charge):
    rmin = 1 / umax
    
    r = rmin / (1 - rho * rho)
    first_term = (rmin*rmin / (rho*rho*E)) * (vtot(umax, m_phi, R, alpha, point_charge) - vtot(1/r, m_phi, R, alpha, point_charge))
    second_term = b * b * (2 - rho*rho)
    
    return 1 / np.sqrt(first_term + second_term)

def b_theta(M_X, m_phi, R, alpha, v, point_charge):
    p = M_X * v            # DM initial momentum (eV)
    E = 1./2 * M_X * v**2  # Initial kinetic energy of incoming particle

    # Make a list of impact parameters
    # Impact factor `b` (eV^-1)
    # Might need to adjust the range for different calculations
    nb = 2000
    if(m_phi > 0):
        b_um = np.logspace(-5, 5, nb)
    else:
        b_um = np.logspace(-5, 5, nb)
    b = b_um / hbarc
    
    umax     = max_u_numerical(E, b, m_phi, R, alpha, point_charge)
    integral = np.empty_like(b)
    for i, _b in enumerate(b):
        integral[i] = quad(integrand, 0, 1, args=(umax[i], E, _b, m_phi, R, alpha, point_charge))[0]
        
    theta = np.pi - 4 * b * integral
    
    return p, b, theta

def db_dq(q, b):
    """Calculate db/dq from unsorted q and b"""
    lnb = np.log(b)
    q_sorted, q_idx = np.unique(q, return_index=True)
    lnb_sorted, b_sorted = lnb[q_idx], b[q_idx]

    # CubicSpline interpolation makes sure the derivative exists
    # Do interpolation in log space and use
    # derivatives given by `scipy.interpolate.CubicSpline()`
    # db/dq = b * dlogb / dq
    lnb_cubic = CubicSpline(q_sorted, lnb_sorted)
    dbdq = b_sorted * lnb_cubic.derivative()( q_sorted )
    
    return q_sorted, b_sorted, dbdq

def dsig_dq(p, pmax, b, theta, q_lin):
    # Take care of nan in theta from numerical integration
    not_nan = np.logical_not(np.isnan(theta))
    b = b[not_nan]
    theta = theta[not_nan]

    # Most of the time there is a maximum point
    # in the theta-b plot
    # Split contribution above and below critical point
    # This avoids overestimating deriv around the peak
    bcidx = np.argmax(theta)
    bcrit = b[bcidx]
    
    ## now need the cross section above and below bcrit
    b1, t1 = b[:bcidx], theta[:bcidx]
    b2, t2 = b[bcidx:], theta[bcidx:]

    q1 = p * np.sqrt( 2*(1-np.cos(t1)) )
    q2 = p * np.sqrt( 2*(1-np.cos(t2)) )
    q  = p * np.sqrt( 2*(1-np.cos(theta)) )

    # Interpolate and calculate dsig/dq
    if(len(b1) > 1 ):
        q1_sorted, b1_sorted, db1 = np.abs( db_dq(q1, b1) )
        # Edit 20230604: linear interp below not used anymore
        # q1_sorted, q1_idx = np.unique(q1, return_index=True)
        # b1_cubic = CubicSpline(q1[q1_idx], b1[q1_idx])(q1[q1_idx])
        # db1 = np.abs(np.gradient(b1_cubic, q1[q1_idx]))
    
    q2_sorted, b2_sorted, db2 = np.abs( db_dq(q2, b2) )  
    # q2_sorted, q2_idx = np.unique(q2, return_index=True)
    # b2_cubic = CubicSpline(q2[q2_idx], b2[q2_idx])(q2[q2_idx])
    # db2 = np.abs(np.gradient(b2_cubic, q2[q2_idx]))
   
    # Linearly interpolate dsigdq and add two parts together 
    if (len(b1) > 1 ):
        dsigdq1 = np.interp(q_lin, q1_sorted, 2 * np.pi * b1_sorted * db1, right=0)
    else:
        dsigdq1 = np.zeros_like(q_lin)
    dsigdq2 = np.interp(q_lin, q2_sorted, 2 * np.pi * b2_sorted * db2, right=0)

    dsigdq_tot = dsigdq1 + dsigdq2
    
    # Edited 20230602: now done in later stages of analysis
    # dsigdq_tot[q_lin < q_thr] = 0  # Cut off at the momentum threshold
    
    return dsigdq_tot

def dR_dq(mx, q, dsdq, vlist):
    # Integrate over DM velocities to get dR/dq
    int_vec = rhoDM / mx * vlist * f_halo_dan(vlist)

    drdq = np.zeros_like(q)
    for i in range(q.size):
        drdq[i] = np.trapz( int_vec * dsdq.T[i], x=vlist )
    
    # natural units -> um^2/GeV, c [cm/s], um^2/cm^2, s/hr    
    conv_fac = hbarc**2 * 1e9 * 3e10 * 1e-8 * 3600
    
    # GeV; Counts/hour/GeV
    return q/1e9, drdq * conv_fac

def run_nugget_calc(R_um, M_X_in, alpha_n_in, m_phi):
    if R_um < 1:
        sphere_type = 'nanosphere'
    else:
        sphere_type = 'microsphere'

    R = R_um / hbarc  # Radius in natural units, eV^-1
    N_T = 0.5 * ( 4/3 * np.pi * (R_um*1e-6)**3) * rho_T/mAMU # Number of neutrons
    
    alpha_n = alpha_n_in  # Dimensionless single neutron-nugget coupling
    alpha = alpha_n * N_T # Total coupling
    mR = m_phi * R        # (= R/lambda), a useful length scale; now defined in `vtot()`

    # `m_phi` is already in eV
    M_X   = M_X_in * 1e9  # Dark matter nugget mass, eV (assumes mass in GeV given on command line)
    m_chi = 0.01 * 1e9    # eV
    N_chi = M_X / m_chi   # Number of dark matter particles in the nugget

    ## Start calculation
    # With fixed `M_X`, `alpha`, `m_phi` 
    # calculate b-theta for each DM velocity
    # and the corresponding cross section dsig/dq
    point_charge = False
    nvels = 2000      # Number of velocities to include in integration
    vlist = np.linspace(vmin, vesc, nvels)
    
    # Maximum momentum in the scattering
    # This would affect how we interpolate and calculate cross section
    # Make sure we are accurate down to ~ MeV for microspheres
    # and ~ keV for nanospheres
    if sphere_type == 'microsphere':
        pmax = np.max((vesc*M_X, 10e9))
    else:
        pmax = np.max((vesc*M_X, 10e6))

    nq     = 20000
    q_lin  = np.linspace(100, 2*pmax*1.1, nq)
    dsdq   = np.empty(shape=(nvels, nq))

    ## If not using pool
    #nb = 2000
    #bb, tt = np.empty(shape=(vlist.size, nb)), np.empty(shape=(vlist.size, nb))
    
    params = list(np.vstack( (np.full(nvels, M_X), np.full(nvels, m_phi), np.full(nvels, R),
                              np.full(nvels, alpha), vlist, np.full(nvels, point_charge) )).T)
    pool   = Pool(32)  # This is the number of CPU we want to allocate for each task
                       # i.e. #SBATCH --cpus-per-task=32
    b_theta_pooled = pool.starmap(b_theta, params)

    ## For debugging purposes
    # _transposed = list(zip(*b_theta_pooled))
    # bb, tt = _transposed[1], _transposed[2]

    for idx, v in enumerate(vlist):
        ## If not using pool
        #p, bb[idx], tt[idx] = b_theta(M_X, m_phi, alpha, v)
        
        # Use multiprocessing to accelerate calculation
        print(f'Idx: {idx}, Velocity: {v}')
        p         = b_theta_pooled[idx][0]
        b         = b_theta_pooled[idx][1]
        theta     = b_theta_pooled[idx][2]
        dsdq[idx] = dsig_dq(p, pmax, b, theta, q_lin)

    q_gev, drdq = dR_dq(M_X, q_lin, dsdq, vlist)

    # outdir = r"C:\Users\yuhan\work\microspheres\code\impulse\data\mphi_%.0e"%m_phi
    outdir = f'/home/yt388/microspheres/impulse/data/mphi_{m_phi:.0e}'
    if(not os.path.isdir(outdir)):
        os.mkdir(outdir)
    
    # For debugging purposes    
    # np.savez(outdir + "/b_theta_alpha_%.5e_MX_%.5e.npz"%(alpha_n, M_X/1e9), b=np.asarray(bb), theta=np.asarray(tt) , v=vlist)
    # np.savez(outdir + "/dsdqdv_alpha_%.5e_MX_%.5e.npz"%(alpha_n, M_X/1e9), q=q_lin, dsdqdv=dsdq, v=vlist)
    
    # GeV; Counts/hour/GeV
    np.savez(outdir + f'/drdq_{sphere_type}_{M_X_in:.5e}_{alpha_n:.5e}_{m_phi:.0e}.npz', mx_gev=M_X, alpha_t=alpha_n, q=q_gev, drdq=drdq)

if __name__ == "__main__":
    R_um       = float(sys.argv[1])  # Sphere radius in um
    M_X_in     = float(sys.argv[2])  # DM mass in GeV
    alpha_n_in = float(sys.argv[3])  # Single neutron coupling
    m_phi      = float(sys.argv[4])  # Mediator mass in eV
    
    print(f'Sphere radius = {R_um:.3f} um')
    print(f'Working on M_X = {M_X_in:.3e} GeV, alpha_t = {alpha_n_in:.3e}, m_phi = {m_phi:.0e} eV')
    
    run_nugget_calc(R_um, M_X_in, alpha_n_in, m_phi)
