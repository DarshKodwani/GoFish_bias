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

def compute_errors_percentage_unmarg(prefix1,prefix2) :
    pkg1 = read_fisher(prefix1)
    cov1 = np.linalg.inv(pkg1['fisher'][:5,:5]) #NOTE: this is hard coded to the fact that we have 6 cosmological parameters!!!
    errors1 = []
    
    pkg2 = read_fisher(prefix2)
    cov2 = np.linalg.inv(pkg2['fisher'][:5,:5]) #NOTE: this is hard coded to the fact that we have 6 cosmological parameters!!!
    errors2 = []

    dif_error = []
    for i in np.arange(5) :  
        dif_error.append( ( abs((np.sqrt(cov1[i,i]) - np.sqrt(cov2[i,i]) )/np.sqrt(cov2[i,i]) )) )
    return dif_error


print compute_errors_percentage("lens_GPI_fsky1", "lens_SPI_fsky1")
print compute_errors_percentage_unmarg("lens_GPI_fsky1", "lens_SPI_fsky1")
