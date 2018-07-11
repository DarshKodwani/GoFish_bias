import numpy as np 

data = np.load("fisher_raw.npz")

print data.keys()

pdcm_bias_v = data['pdcm_bias_v']
fish_PI = data['fisher_PI']
names = data['names']
ifish_PI = np.linalg.inv(fish_PI)

bias_v = np.sum(np.sum(ifish_PI[None,:,:]*pdcm_bias_v[:,:,:], axis = 2),axis =1)
bias = (1/2.)*np.sum(ifish_PI[:,:]*bias_v[None,:], axis = 1)

print bias
print names
