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


print read_fisher("lens_SPI_fsky005")['names']

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
    return np.array(dif_error)

def compute_errors_percentage_unmarg(prefix1,prefix2,num_pname) :
    pkg1 = read_fisher(prefix1)
    cov1 = np.linalg.inv(pkg1['fisher'][:num_pname,:num_pname]) #NOTE: this is hard coded to the fact that we have 4 cosmological parameters!!!
    errors1 = []
    
    pkg2 = read_fisher(prefix2)
    cov2 = np.linalg.inv(pkg2['fisher'][:num_pname,:num_pname]) #NOTE: this is hard coded to the fact that we have 6 cosmological parameters!!!
    errors2 = []

    dif_error = []
    pname = []
    for i in np.arange(num_pname) :  
        pname.append(pkg1['names'][i])
        dif_error.append( ( abs((np.sqrt(cov1[i,i]) - np.sqrt(cov2[i,i]) )/np.sqrt(cov2[i,i]) )) )
    return np.array(dif_error), np.array(pname)

num_pname = 4

##### GPD ERRORS #####
GPD_errors = {}
GPD_errors["c1"] =  compute_errors_percentage("cluster_GPI_fsky1", "cluster_GPD_fsky1")
GPD_errors["c04"] =  compute_errors_percentage("cluster_GPI_fsky04", "cluster_GPD_fsky04")
GPD_errors["c005"] =  compute_errors_percentage("cluster_GPI_fsky005", "cluster_GPD_fsky005")

GPD_errors["l1"] = compute_errors_percentage("lens_GPI_fsky1", "lens_GPD_fsky1")
GPD_errors["l04"] = compute_errors_percentage("lens_GPI_fsky04", "lens_GPD_fsky04")
GPD_errors["l005"] = compute_errors_percentage("lens_GPI_fsky005", "lens_GPD_fsky005")

GPD_errors["lc1"] = compute_errors_percentage("lenscluster_GPI_fsky1", "lenscluster_GPD_fsky1")
GPD_errors["lc04"] =  compute_errors_percentage("lenscluster_GPI_fsky04", "lenscluster_GPD_fsky04")
GPD_errors["lc005"]= compute_errors_percentage("lenscluster_GPI_fsky005", "lenscluster_GPD_fsky005")


print "GPD ERRORS! \n", GPD_errors['c04']

##### SPI ERRORS  #####                                                                                                                                                                                    
SPI_errors = {}
SPI_errors["c1"] =  compute_errors_percentage("cluster_GPI_fsky1", "cluster_SPI_fsky1")
SPI_errors["c04"] =  compute_errors_percentage("cluster_GPI_fsky04", "cluster_SPI_fsky04")
SPI_errors["c005"] =  compute_errors_percentage("cluster_GPI_fsky005", "cluster_SPI_fsky005")

SPI_errors["l1"] = compute_errors_percentage("lens_GPI_fsky1", "lens_SPI_fsky1")
SPI_errors["l04"] = compute_errors_percentage("lens_GPI_fsky04", "lens_SPI_fsky04")
SPI_errors["l005"] = compute_errors_percentage("lens_GPI_fsky005", "lens_SPI_fsky005")

SPI_errors["lc1"] = compute_errors_percentage("lenscluster_GPI_fsky1", "lenscluster_SPI_fsky1")
SPI_errors["lc04"] =  compute_errors_percentage("lenscluster_GPI_fsky04", "lenscluster_SPI_fsky04")
SPI_errors["lc005"]= compute_errors_percentage("lenscluster_GPI_fsky005", "lenscluster_SPI_fsky005")


print "SPI ERROS! \n", SPI_errors['c04']  

##### SPD ERRORS #####
SPD_errors = {}
SPD_errors["c1"] =  compute_errors_percentage("cluster_GPI_fsky1", "cluster_SPD_fsky1")
SPD_errors["c04"] =  compute_errors_percentage("cluster_GPI_fsky04", "cluster_SPD_fsky04")
SPD_errors["c005"] =  compute_errors_percentage("cluster_GPI_fsky005", "cluster_SPD_fsky005")

SPD_errors["l1"] = compute_errors_percentage("lens_GPI_fsky1", "lens_SPD_fsky1")
SPD_errors["l04"] = compute_errors_percentage("lens_GPI_fsky04", "lens_SPD_fsky04")
SPD_errors["l005"] = compute_errors_percentage("lens_GPI_fsky005", "lens_SPD_fsky005")

SPD_errors["lc1"] = compute_errors_percentage("lenscluster_GPI_fsky1", "lenscluster_SPD_fsky1")
SPD_errors["lc04"] =  compute_errors_percentage("lenscluster_GPI_fsky04", "lenscluster_SPD_fsky04")
SPD_errors["lc005"]= compute_errors_percentage("lenscluster_GPI_fsky005", "lenscluster_SPD_fsky005")


print "SPD ERROS \n", SPD_errors['c04']
