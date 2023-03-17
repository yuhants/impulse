import sys, os
import numpy as np

from scipy.optimize import minimize_scalar, newton
from scipy.integrate import quad
import scipy.signal as sig
from scipy.special import erf
from scipy.interpolate import CubicSpline

from multiprocessing import Pool

## General Prameters
hbarc = 0.2     # eV um
rho_T = 2.0e3   # Sphere density, kg/m^3
mAMU = 1.66e-27 # Neutron mass

R_um = 5          # Sphere radius, um
R = R_um / hbarc  # Radius in natural units, eV^-1
N_T = 0.5 * ( 4/3 * np.pi * (R_um*1e-6)**3) * rho_T/mAMU # Number of neutrons
res = 170e6       # detector resolution in eV
q_thr = 0.05e9    # momentum threshold, eV

## DM parameters
# For DM velocity distribution
vmin = 5e-5      # minimum velocity to consider, natural units (c)
vesc = 1.815e-3  # galactic escape velocity
v0 = 7.34e-4     # v0 parameter from Zurek group paper
ve = 8.172e-4    # ve parameter from Zurek group paper

## Functions
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
    I think this is the standard halo model but need to double check.
    See Eq. (2) of https://link.aps.org/doi/10.1103/PhysRevD.42.3572
    """
    N0 = np.pi**1.5 * v0**3 * ( erf(vesc/v0) - 2/np.sqrt(np.pi) * (vesc/v0) * np.exp(-(vesc/v0)**2))
    return 4*np.pi*v**2 * np.exp(-v**2 / v0**2) / N0

def vtot_func(u, m_phi, alpha):
    """Exact Yukawa potential for a uniform density sphere
    
    Parameters
    ----------
    u : float
        Distance inverse 1/r in eV
    m_phi : float
        Mediator mass in eV
    alpha : float
        Dimensionless DM coupling to the sphere
        
    Returns
    -------
    float
        Potential V(u) in eV
    """
    mR = m_phi * R
    
    # Devide the array of u=1/r into three cases
    if (u < 1e-10) : return np.inf
    if (u < 1/R):  # outside
        return 3 * alpha/mR**3 * (mR * np.cosh(mR) - np.sinh(mR)) * np.exp(-m_phi/u) * u
    else:          # inside
        return 3 * alpha/mR**3 * (m_phi - u*(1 + mR)/(1+1./np.tanh(mR)) * (np.sinh(m_phi/u)/np.sinh(mR)))

def vtot(u, m_phi, alpha):
    """Exact Yukawa potential for a uniform density sphere (vectorized version)
    
    Parameters
    ----------
    u : float, array-like
        Distance inverse 1/r in eV
    m_phi : float
        Mediator mass in eV
    alpha : float
        Dimensionless DM coupling to the sphere
        
    Returns
    -------
    float, array-like
        Potential V(u) for each u in eV
    """
    mR = m_phi * R
    
    u = np.asarray(u)
    ret = np.empty_like(u)
    
    # Devide the array of u=1/r into three cases
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

def integrand(u, E, _b, m_phi, alpha):
    """
    Integrand for scattering angle calculation
    
    Parameters
    ----------
    u : float
    E : float
        CM energy in eV
    _b : float
        Impact parameter in eV^-1
        
    Returns
    -------
    float
        Max u value.
    
    """    
    sval = 1 - vtot_func(u, m_phi, alpha)/E - (_b*u)**2
    if sval < 1e-10:  # cut off bad values that should be zero
        sval= np.inf

    integ = _b / np.sqrt(sval)
    return integ

def max_u_func(u, b, E, m_phi, alpha):
    return 1 - (b*u)**2 - vtot(u, m_phi, alpha) / E

def max_u_numerical(E, b, m_phi, alpha):    
    """Numerical solution of maximum u during the scattering
    
    Parameters
    ----------
    E : float
        CM energy in eV
    b : float, array-like
        Impact parameter in eV^-1
        
    Returns
    -------
    1 * n array
        Max u value. n = dim(b)
    """
    max_u = newton(max_u_func, x0=b, args=(b,E, m_phi, alpha))
    return max_u

def b_theta(M_X, m_phi, alpha, v):
    p = M_X * v            # DM initial momentum (eV)
    E = 1./2 * M_X * v**2  # Initial kinetic energy of incoming particle
    
    # Make a list of impact parameters
    if(m_phi > 0):
        b_um = np.logspace(-3, 3, 2000)
    else:
        b_um = np.logspace(-3, 5, 2000)
    b = b_um / hbarc       # Impact factor (eV^-1)
    
    # Calculate scattering angle `theta` for each element in `b`
    max_u = max_u_numerical(E, b, m_phi, alpha)
    Psi = np.empty_like(b)
    for i, _b in enumerate(b):
        umax = max_u[i]
        _b = b[i]
        Psi[i] = quad(integrand, 0, umax, args=(E, _b, m_phi, alpha))[0]
    
    theta = np.pi - 2 * Psi
    
    good_pts = np.logical_not( np.logical_or(np.isnan(theta), np.isinf(theta)) )
    theta = theta[good_pts]
    b = b[good_pts]
    
    return p, b, theta

def dsig_dq(p, pmax, b, theta):
    bcidx = np.argmax(theta)
    bcrit = b[bcidx]

    ## now need the cross section above and below bcrit
    b1, t1 = b[:bcidx], theta[:bcidx]
    b2, t2 = b[bcidx:], theta[bcidx:]

    q1 = p * np.sqrt( 2*(1-np.cos(t1)) )
    q2 = p * np.sqrt( 2*(1-np.cos(t2)) )
    q = p * np.sqrt( 2*(1-np.cos(theta)) )

    q_lin = np.linspace(0, 2*pmax*1.1, 10000)
    if(len(b1) > 1 ):
        q1_sorted, q1_idx = np.unique(q1, return_index=True)
        b1_cubic = CubicSpline(q1[q1_idx], b1[q1_idx])(q1[q1_idx])
        db1 = np.abs(np.gradient(b1_cubic, q1[q1_idx]))
        
    q2_sorted, q2_idx = np.unique(q2, return_index=True)
    b2_cubic = CubicSpline(q2[q2_idx], b2[q2_idx])(q2[q2_idx])
    db2 = np.abs(np.gradient(b2_cubic, q2[q2_idx]))
    
    if (len(b1) > 1 ):
        dsigdq1 = np.interp(q_lin, q1[q1_idx], 2 * np.pi * b1[q1_idx] * db1, right=0)
    else:
        dsigdq1 = np.zeros_like(q_lin)
    dsigdq2 = np.interp(q_lin, q2[q2_idx], 2 * np.pi * b2[q2_idx] * db2, right=0)

    dsigdq_tot = dsigdq1 + dsigdq2
    dsigdq_tot[q_lin < q_thr] = 0  # Cut off at the momentum threshold
    
    return q_lin, dsigdq_tot

def run_nugget_calc(M_X_in, alpha_n_in, m_phi):

    M_X = M_X_in * 1e9   # Dark matter nugget mass, eV (assumes mass in GeV given on command line)
    m_chi = 0.01 * 1e9   # eV
    N_chi = M_X / m_chi  # Number of dark matter particles in the nugget

    rhoDM = 0.3e9        # dark matter mass density, eV/cm^3
    alpha_n = alpha_n_in # Dimensionless single neutron-nugget coupling
    alpha = alpha_n * N_T # Coupling of the entire sphere
    mR = m_phi * R        # (= R/lambda), a useful length scale; now defiend in `vtot()`

    ## Start calculation
    nvels = 2000      # Number of velocities to include in integration
    vlist = np.linspace(vmin, vesc, nvels)
    pmax = np.max((vesc * M_X, 10e9))

    #nb = 2000
    #bb, tt = np.empty(shape=(vlist.size, nb)), np.empty(shape=(vlist.size, nb))
    nq = 10000
    qq, ss = np.empty(shape=(vlist.size, nq)), np.empty(shape=(vlist.size, nq))

    params = list(np.vstack((np.full_like(vlist, M_X), np.full_like(vlist, m_phi), np.full_like(vlist, alpha), vlist)).T)
    pool = Pool(32)
    b_theta_pooled = pool.starmap(b_theta, params)

    _transposed = list(zip(*b_theta_pooled))
    bb, tt = _transposed[1], _transposed[2]

    for idx, v in enumerate(vlist):
        #print(idx)
        #p, bb[idx], tt[idx] = b_theta(M_X, m_phi, alpha, v)
        p = b_theta_pooled[idx][0]
        b = b_theta_pooled[idx][1]
        theta = b_theta_pooled[idx][2]
        qq[idx], ss[idx] = dsig_dq(p, pmax, b, theta)

    int_vec = rhoDM / M_X * vlist * f_halo(vlist)

    tot_xsec = np.zeros(nq)
    for i in range(10000):
        tot_xsec[i] = np.trapz( int_vec * ss.T[i], x=vlist )

    conv_fac = hbarc**2 * 1e9 * 3e10 * 1e-8 * 3600  # natural units -> um^2/GeV, c [cm/s], um^2/cm^2, s/hr

    #outdir = r"C:\Users\yuhan\work\microspheres\code\impulse\yuhan\data\mphi_%.0e"%m_phi
    outdir = "/home/yt388/microspheres/impulse/yuhan/data/mphi_%.0e"%m_phi
    if(not os.path.isdir(outdir)):
        os.mkdir(outdir)
    np.savez(outdir + "/pool_b_theta_alpha_%.5e_MX_%.5e.npz"%(alpha_n, M_X/1e9), b=np.asarray(bb), theta=np.asarray(tt) , v=vlist)
    np.savez(outdir + "/pool_dsdqdv_alpha_%.5e_MX_%.5e.npz"%(alpha_n, M_X/1e9), qq=qq/1e9, dsdqdv = ss, v=vlist)
    np.savez(outdir + "/pool_differential_rate_alpha_%.5e_MX_%.5e.npz"%(alpha_n, M_X/1e9), q=qq/1e9, dsigdq = tot_xsec*conv_fac)

if __name__ == "__main__":
    M_X_in = float(sys.argv[1])      # DM mass in GeV
    alpha_n_in = float(sys.argv[2])  # Dimensionless coupling
    m_phi = float(sys.argv[3])       # Mediator mass in eV

    print(M_X_in, alpha_n_in, m_phi)
    run_nugget_calc(M_X_in, alpha_n_in, m_phi)
