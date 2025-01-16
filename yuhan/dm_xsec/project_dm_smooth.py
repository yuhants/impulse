"""
Project the calculate DM scattering rate to the
z axis and add a Gaussian noise
"""

import numpy as np
import h5py
import os
from scipy.signal import savgol_filter

def get_random_q_samples(qq, drdq, rr):
    norm_factor = np.trapz(drdq, qq)

    f_drdq_norm = drdq / norm_factor       # PDF of q
    Fc_drdq_norm = np.cumsum(f_drdq_norm)  # CDF of q

    qq_sampled = np.interp(rr, Fc_drdq_norm, qq, left=0, right=0)
    return qq_sampled, norm_factor

R_um       = 0.083
mphi       = 1  # eV

mx_list    = np.logspace(-2, 5, 40)
alpha_list = np.logspace(-10, -4, 40)

sigma_gaus = 200  # keV/c

n_mc = int(1e8)
rand_seed = 22040403
rng = np.random.default_rng(rand_seed)

rr = rng.uniform(0, 1, n_mc)
phiphi = rng.uniform(0, np.pi, n_mc)
noise_gaussian = rng.normal(0, sigma_gaus, n_mc)

# data_dir = f'/Users/yuhan/work/impulse/yuhan/data/mphi_1e+00'
data_dir = '/home/yt388/palmer_scratch/data/mphi_1e+00'

# Read in the calculated rate
for i, mx in enumerate(mx_list):
    for j, alpha in enumerate(alpha_list):
        outfile_name = f'drdqz_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.h5py'
        
        if os.path.isfile(os.path.join(data_dir, outfile_name)):
            print(f'Skipping {outfile_name}')
            continue

        file = f'{data_dir}/drdq_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
        drdq_npz = np.load(file)

        qq = drdq_npz['q_kev']
        drdq = drdq_npz['drdq_hz_kev']

        # If we have zero sensitivity
        if np.sum(drdq) == 0:
            bins = np.arange(0, 10000, 50)
            bc = 0.5 * (2 * bins + 50)
            hh, hhz, hhzn = [np.zeros_like(bc) for i in range(3)]
            norm = 0

        else:
            # Smooth with a polynomial filter in the log space
            _qq, _drdq_smoothed = qq, np.nan_to_num( np.exp(savgol_filter(np.log(drdq), 50, 1)), nan=0 )
            if np.sum(_drdq_smoothed) == 0:
                _drdq_smoothed = drdq

            qq_sampled, norm = get_random_q_samples(_qq, _drdq_smoothed, rr)
            qmax = max(50 * ((np.max(qq) // 50) + 2), 10000)

            hh, be   = np.histogram(qq_sampled, bins=np.arange(0, qmax, 50), density=True)
            hhz, be  = np.histogram(qq_sampled*np.abs(np.cos(phiphi)), bins=np.arange(0, qmax, 50), density=True)
            hhzn, be = np.histogram(qq_sampled*np.abs(np.cos(phiphi)) + noise_gaussian, bins=np.arange(0, qmax, 50), density=True)
            bc = 0.5 * (be[1:] + be[:-1])

        with h5py.File(os.path.join(data_dir, outfile_name), 'w') as fout:
            print(f'Writing file {os.path.join(data_dir, outfile_name)}')

            g = fout.create_group('dm_rate_sampled')
            
            g3 = g.create_dataset('bin_center_kev',     data=bc,        dtype=np.float64)
            g0 = g.create_dataset('rate_binned_iso',    data=hh*norm,   dtype=np.float64)
            g1 = g.create_dataset('rate_binned_z',      data=hhz*norm,  dtype=np.float64)
            g2 = g.create_dataset('rate_binned_z_gaus', data=hhzn*norm, dtype=np.float64)

            fout.close()
