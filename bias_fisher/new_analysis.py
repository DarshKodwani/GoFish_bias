import numpy as np
import matplotlib.pyplot as plt

def read_fisher(prefix) :
    data=np.load(prefix+"/fisher_raw.npz")
    
    fish_package={'names' :data['names'],
                  'values':data['values'],
                  'labels':data['labels'],
                  'fisher':data['fisher_tot']}
    return fish_package

#Without marginalising over the bias parameters


print read_fisher("lens_SPD_fsky005_bias")['names']

def compute_errors_percentage(prefix1,prefix2) :
    pkg1 = read_fisher(prefix1)
    cov1 = np.linalg.inv(pkg1['fisher'])
    errors1 = []
    
    pkg2 = read_fisher(prefix2)
    cov2 = np.linalg.inv(pkg2['fisher'])
    errors2 = []

    dif_error = []
    for i, nm in enumerate(pkg1['names']) :  
        dif_error.append( ( abs((np.sqrt(cov1[i,i]) - np.sqrt(cov2[i,i]))/np.sqrt(cov2[i,i]) )) )
    return dif_error

def compute_errors_percentage_unmarg(prefix1,prefix2,num_pname) :
    pkg1 = read_fisher(prefix1)
    cov1 = np.linalg.inv(pkg1['fisher'][:num_pname,:num_pname]) #NOTE: this is hard coded to the fact that we have 6 cosmological parameters!!!
    errors1 = []
    
    pkg2 = read_fisher(prefix2)
    cov2 = np.linalg.inv(pkg2['fisher'][:num_pname,:num_pname]) #NOTE: this is hard coded to the fact that we have 6 cosmological parameters!!!
    errors2 = []

    dif_error = []
    pname = []
    for i in np.arange(num_pname) :  
        pname.append(pkg1['names'][i])
        dif_error.append( ( abs((np.sqrt(cov1[i,i]) - np.sqrt(cov2[i,i]) )/np.sqrt(cov2[i,i]) )) )
    return dif_error, pname

num_pname = 6
print compute_errors_percentage("lenscluster_GPD_fsky1_bias", "lenscluster_SPD_fsky1_bias")
print compute_errors_percentage_unmarg("lenscluster_GPD_fsky1_bias", "lenscluster_SPD_fsky1_bias",num_pname)
