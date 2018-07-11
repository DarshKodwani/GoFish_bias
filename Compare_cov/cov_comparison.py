from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib import pyplot as plt

cov_G = np.load('Gauss_cov_v1.npz') ; cov_G = cov_G["arr_0"]
cov_S = np.load('Ssc_cov_v1.npz') ; cov_S = cov_S["arr_0"]
cov_E = np.load('/users/dkodwani/Software/CCL/GoFish_SSC2/4darsh/curves_SRD_ss/icov_y1.npz') ; cov_E = np.linalg.inv(cov_E["icov"])
cov_GS=cov_G+cov_S

#Write the summary statistics of the covariances to an external file 

#G_stats = np.array((np.mean(cov_G), np.max(cov_G), np.min(cov_G)))
#S_stats = np.array((np.mean(cov_S), np.max(cov_S), np.min(cov_S)))
#GS_stats = np.array((np.mean(cov_GS), np.max(cov_GS), np.min(cov_GS)))
#E_stats = np.array((np.mean(cov_E), np.max(cov_E), np.min(cov_E)))
#full_stats = np.vstack((G_stats,S_stats,GS_stats,E_stats))
#np.savetxt('cov_statistics_dz0005.txt', full_stats)

#Comapre the covariances of various Nell

cov_G_larrdef = np.load('Gauss_cov_larrdef.npz') ; cov_G_larrdef = cov_G_larrdef["arr_0"]
cov_G_larrdouble = np.load('Gauss_cov_larrdouble.npz') ; cov_G_larrdouble = cov_G_larrdouble["arr_0"]
cov_G_larrquad = np.load('Gauss_cov_larrquad.npz') ; cov_G_larrquad = cov_G_larrquad["arr_0"]
cov_S_larrdef = np.load('Ssc_cov_larrdef.npz') ; cov_S_larrdef = cov_S_larrdef["arr_0"]
cov_S_larrdouble = np.load('Ssc_cov_larrdouble.npz') ; cov_S_larrdouble = cov_S_larrdouble["arr_0"]
cov_S_larrquad = np.load('Ssc_cov_larrquad.npz') ; cov_S_larrquad = cov_S_larrquad["arr_0"]
cov_GS_larrdef=cov_G_larrdef+cov_S_larrdef
cov_GS_larrdouble=cov_G_larrdouble+cov_S_larrdouble
cov_GS_larrquad=cov_G_larrquad+cov_S_larrquad

print "SHAPE SSC DEF, DOUBLE,QUAD", np.shape(cov_S_larrdef), np.shape(cov_S_larrdouble), np.shape(cov_S_larrquad)
Ccov_S_larrdouble = (1./4.)*(np.sum(np.sum(cov_S_larrdouble.reshape(225,2,225,2),axis=3),axis=1))
Ccov_S_larrquad = (1/16.)*(np.sum(np.sum(cov_S_larrquad.reshape(225,4,225,4),axis=3),axis=1))
eldef=np.array((np.mean(cov_S_larrdef),np.min(cov_S_larrdef),np.max(cov_S_larrdef)))
eldouble=np.array((np.mean(Ccov_S_larrdouble), np.min(Ccov_S_larrdouble), np.max(Ccov_S_larrdouble)))
elquad=np.array((np.mean(Ccov_S_larrquad), np.min(Ccov_S_larrquad), np.max(Ccov_S_larrquad)))
eltot=np.vstack((eldef, eldouble, elquad))
np.savetxt('ell_checks.txt', eltot)
print "RESHAPED", np.shape(Ccov_S_larrdouble),np.shape(Ccov_S_larrquad)

#Plots of covariance

#fig, (ax, ax2, cax) = plt.subplots(ncols=3,figsize=(7,3), 
#                  gridspec_kw={"width_ratios":[1,1, 0.05]})
#fig.subplots_adjust(wspace=0.3)
#im1 = plt.imshow(np.log10(np.fabs(cov_G)))
#im2 = plt.imshow(np.log10(np.fabs(cov_G)))
#ax.set_ylabel("y label")

#ip = InsetPosition(ax2, [1.05,0,0.05,1]) 
#cax.set_axes_locator(ip)

#fig.colorbar(im, cax=cax, ax=[ax,ax2])

#plt.show()
#quit()
#fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(8.5, 5))
fig = plt.figure(figsize=(10, 3.5))

plt.subplot(1, 3, 1)
plt.title("Gaussian")
plt.imshow(np.log10(np.fabs(cov_G)))
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("SSC")
plt.imshow(np.log10(np.fabs(cov_S)))
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("SSC+Gaussian")
plt.imshow(np.log10(np.fabs(cov_S + cov_G)))

#cax1 = divider.append_axes("right", size="5%", pad=0.05)
#plt.colorbar(extend='both')
#cax = plt.axes()
plt.colorbar()
plt.savefig("Lens_cov.pdf")
plt.show()
