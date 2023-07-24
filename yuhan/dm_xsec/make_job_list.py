import os
import numpy as np

R_um       = 0.0075 
mx_list    = np.logspace(-6, 1, 40)
alpha_list = np.logspace(-10, -4, 40)
mphi_list  = [1000, 100, 10]

job_file = open("job_list_15nm.txt", "wt")

for mx in mx_list:
    for alpha in alpha_list:
        for mphi in mphi_list:

            outdir = f'/home/yt388/palmer_scratch/data/mphi_{mphi:.0e}'
            outfile = outdir + f'/drdq_nanosphere_{R_um:.2e}_{mx:.5e}_{alpha:.5e}_{mphi:.0e}.npz'
            if( os.path.isfile(outfile) ):
                 print("Skipping: ", outfile)
                 continue

            job_str = f'module load miniconda; conda activate microsphere; python rate_massive_mediator.py {R_um} {mx} {alpha} {mphi}\n'
            job_file.write( job_str )

job_file.close()
