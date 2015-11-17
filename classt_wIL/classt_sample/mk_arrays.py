import numpy as np
import matplotlib.pyplot as plt

zmax=0.9
nz=99
zarr=zmax*np.arange(nz)/float(nz-1)
earr=np.zeros(nz)
sarr=1.*np.ones(nz)
barr=1.5*np.ones(nz)
narr=(zarr/0.3)**2*np.exp(-(zarr/0.3)**1.5)
aIarr=5.76*np.ones(nz)
frarr=0.3/(1+(zarr/0.5)**5)

plt.plot(zarr,narr); plt.show()
plt.plot(zarr,frarr); plt.show()

np.savetxt("nz_test.txt",np.transpose([zarr,narr]))
np.savetxt("bias_test.txt",np.transpose([zarr,barr]))
np.savetxt("s_bias_test.txt",np.transpose([zarr,sarr]))
np.savetxt("e_bias_test.txt",np.transpose([zarr,earr]))
np.savetxt("a_bias_test.txt",np.transpose([zarr,aIarr]))
np.savetxt("fred_test.txt",np.transpose([zarr,frarr]))