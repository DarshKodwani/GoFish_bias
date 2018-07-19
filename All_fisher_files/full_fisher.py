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
 #   errors = []
    for i,nm in enumerate(pkg['names']) :
        print nm+" = %lE"%(pkg['values'][i])+" +- %lE"%(np.sqrt(cov[i,i]))
 #       errors.append(np.sqrt(cov[i,i]))
 #   return np.array(errors) #The shape of this object is [om,s8,ob,ns,wa,w0,bias...]. 

#print print_sigmas("lens_GPI_fsky1")
#quit()

print "Galaxy shear Fisher"

print "---------GPI fsky=1----------\n ", print_sigmas("lens_GPI_fsky1")
print "---------GPI fsky=0.4----------\n ", print_sigmas("lens_GPI_fsky04")
print "---------GPI fsky=0.05----------\n ", print_sigmas("lens_GPI_fsky005")

print "---------GPD fsky=1----------\n ", print_sigmas("lens_GPD_fsky1")
print "---------GPD fsky=0.4----------\n ", print_sigmas("lens_GPD_fsky04")
print "---------GPD fsky=0.05----------\n ", print_sigmas("lens_GPD_fsky005")

print "---------SPI fsky=1----------\n ", print_sigmas("lens_SPI_fsky1")
print "---------SPI fsky=0.4----------\n ", print_sigmas("lens_SPI_fsky04")
print "---------SPI fsky=0.05----------\n ", print_sigmas("lens_SPI_fsky005")

print "---------SPD fsky=1----------\n ", print_sigmas("lens_SPD_fsky1")
print "---------SPD fsky=0.4----------\n ", print_sigmas("lens_SPD_fsky04")
print "---------SPD fsky=0.05----------\n ", print_sigmas("lens_SPD_fsky005")

print "Galaxy clustering  Fisher"

print "---------GPI fsky=1----------\n ", print_sigmas("cluster_GPI_fsky1")
print "---------GPI fsky=0.4----------\n ", print_sigmas("cluster_GPI_fsky04")
print "---------GPI fsky=0.05----------\n ", print_sigmas("cluster_GPI_fsky005")

print "---------GPD fsky=1----------\n ", print_sigmas("cluster_GPD_fsky1")
print "---------GPD fsky=0.4----------\n ", print_sigmas("cluster_GPD_fsky04")
print "---------GPD fsky=0.05----------\n ", print_sigmas("cluster_GPD_fsky005")

print "---------SPI fsky=1----------\n ", print_sigmas("cluster_SPI_fsky1")
print "---------SPI fsky=0.4----------\n ", print_sigmas("cluster_SPI_fsky04")
print "---------SPI fsky=0.05----------\n ", print_sigmas("cluster_SPI_fsky005")

print "---------SPD fsky=1----------\n ", print_sigmas("cluster_SPD_fsky1")
print "---------SPD fsky=0.4----------\n ", print_sigmas("cluster_SPD_fsky04")
print "---------SPD fsky=0.05----------\n ", print_sigmas("cluster_SPD_fsky005")

print "Galaxy clusteirng and shear Fisher"

print "---------GPI fsky=1----------\n ", print_sigmas("lenscluster_GPI_fsky1")
print "---------GPI fsky=0.4----------\n ", print_sigmas("lenscluster_GPI_fsky04")
print "---------GPI fsky=0.05----------\n ", print_sigmas("lenscluster_GPI_fsky005")

print "---------GPD fsky=1----------\n ", print_sigmas("lenscluster_GPD_fsky1")
print "---------GPD fsky=0.4----------\n ", print_sigmas("lenscluster_GPD_fsky04")
print "---------GPD fsky=0.05----------\n ", print_sigmas("lenscluster_GPD_fsky005")

print "---------SPI fsky=1----------\n ", print_sigmas("lenscluster_SPI_fsky1")
print "---------SPI fsky=0.4----------\n ", print_sigmas("lenscluster_SPI_fsky04")
print "---------SPI fsky=0.05----------\n ", print_sigmas("lenscluster_SPI_fsky005")

print "---------SPD fsky=1----------\n ", print_sigmas("lenscluster_SPD_fsky1")
print "---------SPD fsky=0.4----------\n ", print_sigmas("lenscluster_SPD_fsky04")
print "---------SPD fsky=0.05----------\n ", print_sigmas("lenscluster_SPD_fsky005")
