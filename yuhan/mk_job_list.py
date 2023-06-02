import os
import numpy as np

#mass_list = np.hstack((np.logspace(1, 4, 48), np.logspace(5,9,5))) #np.logspace(1,5,21)
mass_list =  np.logspace(np.log10(5e4), 7, 6) ##np.hstack((np.logspace(1, 4, 18), np.logspace(5,9,4))) #np.logspace(1,5,21)
alpha_list = np.logspace(-11,-4,8)
phi_list = [1e-2,]

job_file = open("job_list.txt", "wt")

job_str = "module load miniconda; source activate py37_dev; /gpfs/loomis/project/david_moore/dcm42/conda_envs/py37_dev/bin/python nugget_rate_v3.py %e %e %e\n"

for m in mass_list:
    for a in alpha_list:
        for p in phi_list:

            outfile = "data/mphi_%.0e/differential_rate_alpha_%.5e_MX_%.5e.npz"%(p,a,m)
            if( os.path.isfile(outfile) ):
                print("Skipping: ", outfile)
                continue
            job_file.write( job_str%(m,a,p) )

job_file.close()
