import numpy as np
import matplotlib.pyplot as plt
import fishplot as fsh

fish_GPI=np.load("/users/dkodwani/Software/CCL/GoFish_SSC2/Fisher_lens_cluster/Fisher_lenscluster_full_GPI_fsky1/fisher_raw.npz")
fish_GPD=np.load("/users/dkodwani/Software/CCL/GoFish_SSC2/Fisher_lens_cluster/Fisher_lenscluster_full_GPD_fsky1/fisher_raw.npz")
fish_SPI=np.load("/users/dkodwani/Software/CCL/GoFish_SSC2/Fisher_lens_cluster/Fisher_lenscluster_full_SPI_fsky1/fisher_raw.npz")
fish_SPD=np.load("/users/dkodwani/Software/CCL/GoFish_SSC2/Fisher_lens_cluster/Fisher_lenscluster_full_SPD_fsky1/fisher_raw.npz")

fish_GPI['fisher_tot']
fish_GPI['names']
fish_GPI['values']
fish_GPI['labels']

fish_GPD['fisher_tot']
fish_GPD['names']
fish_GPD['values']
fish_GPD['labels']

fish_SPI['fisher_tot']
fish_SPI['names']
fish_SPI['values']
fish_SPI['labels']

fish_SPD['fisher_tot']
fish_SPD['names']
fish_SPD['values']
fish_SPD['labels']

print fish_GPI.keys()
quit()

pars=[]
for n,v,l in zip(fish_GPI['names'],fish_GPI['values'],fish_GPI['labels']) :
    pars.append(fsh.ParamFisher(0,-1,n,l,True,True,0))
pars=np.array(pars)
#fsh.plot_fisher_all(pars,[dfy1['fisher_tot'],dfy10['fisher_tot']],['none','none'],[2,2],['solid','solid'],['r','g'],['Y1','Y10'],3.,'none')
fsh.plot_fisher_all(pars,[fish_GPI['fisher_cls'],fish_GPD['fisher_cls']],
                    [{'col':'#F7B825','ls':'solid','lw':2,'alpha':1.0},{'col':'#0066CC','ls':'solid','lw':2,'alpha':1.0}],
                    ['GPI','GPD'],2.,'fisher_covariance.png',do_1D=False)
plt.show()
#F7B825
#0066CC
