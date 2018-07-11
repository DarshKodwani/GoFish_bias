import numpy as np
import matplotlib.pyplot as plt

full_fish = np.load("fisher_raw.npz")

#print type(full_fish['fisher_tot'])
print full_fish['names']
#print full_fish.keys()

t1= np.delete(full_fish['fisher_tot'],np.s_[6:],axis=0)
cosmo_only= np.delete(t1,np.s_[6:],axis=1)
#print np.shape(cosmo_only)
#print cosmo_only
cov_cosmo_only = np.linalg.inv(cosmo_only)
sigma_abs = np.sqrt(abs(cov_cosmo_only))
#print cov_cosmo_only
print sigma_abs
