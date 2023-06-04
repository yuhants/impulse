import os
import numpy as np

R_um       = 0.075 
mx_list    = np.logspace(-4, 10, 10)
alpha_list = np.logspace(-12, -4, 10)
mphi_list  = [1e-2,]

job_file = open("job_list.txt", "wt")

for mx in mx_list:
    for alpha in alpha_list:
        for mphi in mphi_list:

            outdir = f'/home/yt388/microspheres/impulse/data/mphi_{mphi:.0e}'
            outfile = outdir + f'/drdq_nanosphere_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
            if( os.path.isfile(outfile) ):
                print("Skipping: ", outfile)
                continue

            job_str = f'module load miniconda; source activate microsphere; python rate_massive_mediator.py {R_um} {mx} {alpha} {mphi}\n'
            job_file.write( job_str )

job_file.close()
