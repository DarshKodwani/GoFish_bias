import numpy as np
import matplotlib.pyplot as plt

def read_fisher(prefix) :
    data=np.load(prefix+"/fisher_raw.npz")
    
    fish_package={'names' :data['names'],
                  'values':data['values'],
                  'labels':data['labels'],
                  'fisher':data['fisher_tot']}
    return fish_package

def print_sigmas(prefix) :
    pkg=read_fisher(prefix)
    cov=np.linalg.inv(pkg['fisher'])
    for i,nm in enumerate(pkg['names']) :
        print nm+" = %lE"%(pkg['values'][i])+" +- %lE"%(np.sqrt(cov[i,i]))

#print print_sigmas("Fisher_lens_cluster/Fisher_lenscluster_full_GPD_fsky1")
print "Galaxy shear and clustering for 7 and 9 redhshift bins and f_sky = 1"
print "---------GPD----------\n ", print_sigmas("Fisher_lenscluster_full_GPD_fsky1")
print "---------GPI----------\n ", print_sigmas("Fisher_lenscluster_full_GPI_fsky1")
print "---------SPD----------\n ", print_sigmas("Fisher_lenscluster_full_SPD_fsky1")
print "---------SPI----------\n ", print_sigmas("Fisher_lenscluster_full_SPI_fsky1")


