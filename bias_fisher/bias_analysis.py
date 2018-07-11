import numpy as np 
import matplotlib.pyplot as plt

######## Loading data ##########

cluster_GPD_fsky1 = np.load("cluster_GPD_fsky1_bias/fisher_raw.npz")
cluster_GPD_fsky04 = np.load("cluster_GPD_fsky04_bias/fisher_raw.npz")
cluster_GPD_fsky005 = np.load("cluster_GPD_fsky005_bias/fisher_raw.npz")
cluster_SPD_fsky1 = np.load("cluster_SPD_fsky1_bias/fisher_raw.npz")
cluster_SPD_fsky04 = np.load("cluster_SPD_fsky04_bias/fisher_raw.npz")
cluster_SPD_fsky005 = np.load("cluster_SPD_fsky005_bias/fisher_raw.npz")

lens_GPD_fsky1 = np.load("lens_GPD_fsky1_bias/fisher_raw.npz")
lens_GPD_fsky04 = np.load("lens_GPD_fsky04_bias/fisher_raw.npz")
lens_GPD_fsky005 = np.load("lens_GPD_fsky005_bias/fisher_raw.npz")
lens_SPD_fsky1 = np.load("lens_SPD_fsky1_bias/fisher_raw.npz")
lens_SPD_fsky04 = np.load("lens_SPD_fsky04_bias/fisher_raw.npz")
lens_SPD_fsky005 = np.load("lens_SPD_fsky005_bias/fisher_raw.npz")

lenscluster_GPD_fsky1 = np.load("lenscluster_GPD_fsky1_bias/fisher_raw.npz")
lenscluster_GPD_fsky04 = np.load("lenscluster_GPD_fsky04_bias/fisher_raw.npz")
lenscluster_GPD_fsky005 = np.load("lenscluster_GPD_fsky005_bias/fisher_raw.npz")
lenscluster_SPD_fsky1 = np.load("lenscluster_SPD_fsky1_bias/fisher_raw.npz")
lenscluster_SPD_fsky04 = np.load("lenscluster_SPD_fsky04_bias/fisher_raw.npz")
lenscluster_SPD_fsky005 = np.load("lenscluster_SPD_fsky005_bias/fisher_raw.npz")

######## Extracting stuff #########

#Clusters

ifish_PI_CGPD_f1 = np.linalg.inv(cluster_GPD_fsky1["fisher_PI"])
ifish_PI_CGPD_f04 = np.linalg.inv(cluster_GPD_fsky04["fisher_PI"])
ifish_PI_CGPD_f005 = np.linalg.inv(cluster_GPD_fsky005["fisher_PI"])
ifish_PI_CSPD_f1 = np.linalg.inv(cluster_SPD_fsky1["fisher_PI"])
ifish_PI_CSPD_f04 = np.linalg.inv(cluster_SPD_fsky04["fisher_PI"])
ifish_PI_CSPD_f005 = np.linalg.inv(cluster_SPD_fsky005["fisher_PI"])

bv_PI_CGPD_f1 = cluster_GPD_fsky1["pdcm_bias_v"]
bv_PI_CGPD_f04 = cluster_GPD_fsky04["pdcm_bias_v"]
bv_PI_CGPD_f005 = cluster_GPD_fsky005["pdcm_bias_v"]
bv_PI_CSPD_f1 = cluster_SPD_fsky1["pdcm_bias_v"]
bv_PI_CSPD_f04 = cluster_SPD_fsky04["pdcm_bias_v"]
bv_PI_CSPD_f005 = cluster_SPD_fsky005["pdcm_bias_v"]

ifish_PD_CGPD_f1 = np.linalg.inv(cluster_GPD_fsky1["fisher_PD"])
ifish_PD_CGPD_f04 = np.linalg.inv(cluster_GPD_fsky04["fisher_PD"])
ifish_PD_CGPD_f005 = np.linalg.inv(cluster_GPD_fsky005["fisher_PD"])
ifish_PD_CSPD_f1 = np.linalg.inv(cluster_SPD_fsky1["fisher_PD"])
ifish_PD_CSPD_f04 = np.linalg.inv(cluster_SPD_fsky04["fisher_PD"])
ifish_PD_CSPD_f005 = np.linalg.inv(cluster_SPD_fsky005["fisher_PD"])

#Lensing

ifish_PI_LGPD_f1 = np.linalg.inv(lens_GPD_fsky1["fisher_PI"])
ifish_PI_LGPD_f04 = np.linalg.inv(lens_GPD_fsky04["fisher_PI"])
ifish_PI_LGPD_f005 = np.linalg.inv(lens_GPD_fsky005["fisher_PI"])
ifish_PI_LSPD_f1 = np.linalg.inv(lens_SPD_fsky1["fisher_PI"])
ifish_PI_LSPD_f04 = np.linalg.inv(lens_SPD_fsky04["fisher_PI"])
ifish_PI_LSPD_f005 = np.linalg.inv(lens_SPD_fsky005["fisher_PI"])

bv_PI_LGPD_f1 = lens_GPD_fsky1["pdcm_bias_v"]
bv_PI_LGPD_f04 = lens_GPD_fsky04["pdcm_bias_v"]
bv_PI_LGPD_f005 = lens_GPD_fsky005["pdcm_bias_v"]
bv_PI_LSPD_f1 = lens_SPD_fsky1["pdcm_bias_v"]
bv_PI_LSPD_f04 = lens_SPD_fsky04["pdcm_bias_v"]
bv_PI_LSPD_f005 = lens_SPD_fsky005["pdcm_bias_v"]

ifish_PD_LGPD_f1 = np.linalg.inv(lens_GPD_fsky1["fisher_PD"])
ifish_PD_LGPD_f04 = np.linalg.inv(lens_GPD_fsky04["fisher_PD"])
ifish_PD_LGPD_f005 = np.linalg.inv(lens_GPD_fsky005["fisher_PD"])
ifish_PD_LSPD_f1 = np.linalg.inv(lens_SPD_fsky1["fisher_PD"])
ifish_PD_LSPD_f04 = np.linalg.inv(lens_SPD_fsky04["fisher_PD"])
ifish_PD_LSPD_f005 = np.linalg.inv(lens_SPD_fsky005["fisher_PD"])

#Lensing + clusters

ifish_PI_LCGPD_f1 = np.linalg.inv(lenscluster_GPD_fsky1["fisher_PI"])
ifish_PI_LCGPD_f04 = np.linalg.inv(lenscluster_GPD_fsky04["fisher_PI"])
ifish_PI_LCGPD_f005 = np.linalg.inv(lenscluster_GPD_fsky005["fisher_PI"])
ifish_PI_LCSPD_f1 = np.linalg.inv(lenscluster_SPD_fsky1["fisher_PI"])
ifish_PI_LCSPD_f04 = np.linalg.inv(lenscluster_SPD_fsky04["fisher_PI"])
ifish_PI_LCSPD_f005 = np.linalg.inv(lenscluster_SPD_fsky005["fisher_PI"])

bv_PI_LCGPD_f1 = lenscluster_GPD_fsky1["pdcm_bias_v"]
bv_PI_LCGPD_f04 = lenscluster_GPD_fsky04["pdcm_bias_v"]
bv_PI_LCGPD_f005 = lenscluster_GPD_fsky005["pdcm_bias_v"]
bv_PI_LCSPD_f1 = lenscluster_SPD_fsky1["pdcm_bias_v"]
bv_PI_LCSPD_f04 = lenscluster_SPD_fsky04["pdcm_bias_v"]
bv_PI_LCSPD_f005 = lenscluster_SPD_fsky005["pdcm_bias_v"]

ifish_PD_LCGPD_f1 = np.linalg.inv(lenscluster_GPD_fsky1["fisher_PD"])
ifish_PD_LCGPD_f04 = np.linalg.inv(lenscluster_GPD_fsky04["fisher_PD"])
ifish_PD_LCGPD_f005 = np.linalg.inv(lenscluster_GPD_fsky005["fisher_PD"])
ifish_PD_LCSPD_f1 = np.linalg.inv(lenscluster_SPD_fsky1["fisher_PD"])
ifish_PD_LCSPD_f04 = np.linalg.inv(lenscluster_SPD_fsky04["fisher_PD"])
ifish_PD_LCSPD_f005 = np.linalg.inv(lenscluster_SPD_fsky005["fisher_PD"])

##### Computing the bias #####

#Clusters
bias_v_CGD_f1 = np.sum(np.sum(ifish_PI_CGPD_f1[None,:,:]*bv_PI_CGPD_f1[:,:,:], axis = 2),axis =1)
bias_CGD_f1 = (1/2.)*np.sum(ifish_PI_CGPD_f1[:,:]*bias_v_CGD_f1[None,:], axis = 1)
bias_v_CGD_f04 = np.sum(np.sum(ifish_PI_CGPD_f04[None,:,:]*bv_PI_CGPD_f04[:,:,:], axis = 2),axis =1)
bias_CGD_f04 = (1/2.)*np.sum(ifish_PI_CGPD_f04[:,:]*bias_v_CGD_f04[None,:], axis = 1)
bias_v_CGD_f005 = np.sum(np.sum(ifish_PI_CGPD_f005[None,:,:]*bv_PI_CGPD_f005[:,:,:], axis = 2),axis =1)
bias_CGD_f005 = (1/2.)*np.sum(ifish_PI_CGPD_f005[:,:]*bias_v_CGD_f005[None,:], axis = 1)

bias_v_CSD_f1 = np.sum(np.sum(ifish_PI_CSPD_f1[None,:,:]*bv_PI_CSPD_f1[:,:,:], axis = 2),axis =1)
bias_CSD_f1 = (1/2.)*np.sum(ifish_PI_CSPD_f1[:,:]*bias_v_CSD_f1[None,:], axis = 1)
bias_v_CSD_f04 = np.sum(np.sum(ifish_PI_CSPD_f04[None,:,:]*bv_PI_CSPD_f04[:,:,:], axis = 2),axis =1)
bias_CSD_f04 = (1/2.)*np.sum(ifish_PI_CSPD_f04[:,:]*bias_v_CSD_f04[None,:], axis = 1)
bias_v_CSD_f005 = np.sum(np.sum(ifish_PI_CSPD_f005[None,:,:]*bv_PI_CSPD_f005[:,:,:], axis = 2),axis =1)
bias_CSD_f005 = (1/2.)*np.sum(ifish_PI_CSPD_f005[:,:]*bias_v_CSD_f005[None,:], axis = 1)

#Lensing
bias_v_LGD_f1 = np.sum(np.sum(ifish_PI_LGPD_f1[None,:,:]*bv_PI_LGPD_f1[:,:,:], axis = 2),axis =1)
bias_LGD_f1 = (1/2.)*np.sum(ifish_PI_LGPD_f1[:,:]*bias_v_LGD_f1[None,:], axis = 1)
bias_v_LGD_f04 = np.sum(np.sum(ifish_PI_LGPD_f04[None,:,:]*bv_PI_LGPD_f04[:,:,:], axis = 2),axis =1)
bias_LGD_f04 = (1/2.)*np.sum(ifish_PI_LGPD_f04[:,:]*bias_v_LGD_f04[None,:], axis = 1)
bias_v_LGD_f005 = np.sum(np.sum(ifish_PI_LGPD_f005[None,:,:]*bv_PI_LGPD_f005[:,:,:], axis = 2),axis =1)
bias_LGD_f005 = (1/2.)*np.sum(ifish_PI_LGPD_f005[:,:]*bias_v_LGD_f005[None,:], axis = 1)

bias_v_LSD_f1 = np.sum(np.sum(ifish_PI_LSPD_f1[None,:,:]*bv_PI_LSPD_f1[:,:,:], axis = 2),axis =1)
bias_LSD_f1 = (1/2.)*np.sum(ifish_PI_LSPD_f1[:,:]*bias_v_LSD_f1[None,:], axis = 1)
bias_v_LSD_f04 = np.sum(np.sum(ifish_PI_LSPD_f04[None,:,:]*bv_PI_LSPD_f04[:,:,:], axis = 2),axis =1)
bias_LSD_f04 = (1/2.)*np.sum(ifish_PI_LSPD_f04[:,:]*bias_v_LSD_f04[None,:], axis = 1)
bias_v_LSD_f005 = np.sum(np.sum(ifish_PI_LSPD_f005[None,:,:]*bv_PI_LSPD_f005[:,:,:], axis = 2),axis =1)
bias_LSD_f005 = (1/2.)*np.sum(ifish_PI_LSPD_f005[:,:]*bias_v_LSD_f005[None,:], axis = 1)

#Lensing+clustering
bias_v_LCGD_f1 = np.sum(np.sum(ifish_PI_LCGPD_f1[None,:,:]*bv_PI_LCGPD_f1[:,:,:], axis = 2),axis =1)
bias_LCGD_f1 = (1/2.)*np.sum(ifish_PI_LCGPD_f1[:,:]*bias_v_LCGD_f1[None,:], axis = 1)
bias_v_LCGD_f04 = np.sum(np.sum(ifish_PI_LCGPD_f04[None,:,:]*bv_PI_LCGPD_f04[:,:,:], axis = 2),axis =1)
bias_LCGD_f04 = (1/2.)*np.sum(ifish_PI_LCGPD_f04[:,:]*bias_v_LCGD_f04[None,:], axis = 1)
bias_v_LCGD_f005 = np.sum(np.sum(ifish_PI_LCGPD_f005[None,:,:]*bv_PI_LCGPD_f005[:,:,:], axis = 2),axis =1)
bias_LCGD_f005 = (1/2.)*np.sum(ifish_PI_LCGPD_f005[:,:]*bias_v_LCGD_f005[None,:], axis = 1)

bias_v_LCSD_f1 = np.sum(np.sum(ifish_PI_LCSPD_f1[None,:,:]*bv_PI_LCSPD_f1[:,:,:], axis = 2),axis =1)
bias_LCSD_f1 = (1/2.)*np.sum(ifish_PI_LCSPD_f1[:,:]*bias_v_LCSD_f1[None,:], axis = 1)
bias_v_LCSD_f04 = np.sum(np.sum(ifish_PI_LCSPD_f04[None,:,:]*bv_PI_LCSPD_f04[:,:,:], axis = 2),axis =1)
bias_LCSD_f04 = (1/2.)*np.sum(ifish_PI_LCSPD_f04[:,:]*bias_v_LCSD_f04[None,:], axis = 1)
bias_v_LCSD_f005 = np.sum(np.sum(ifish_PI_LCSPD_f005[None,:,:]*bv_PI_LCSPD_f005[:,:,:], axis = 2),axis =1)
bias_LCSD_f005 = (1/2.)*np.sum(ifish_PI_LCSPD_f005[:,:]*bias_v_LCSD_f005[None,:], axis = 1)

#Plots

#Omega_m plot

""" Creating data vectors for plots """

fskys = [1, 0.4, 0.05]
omega_m_GCbias = np.array([bias_CGD_f1[0], bias_CGD_f04[0], bias_CGD_f005[0]])
omega_m_SCbias = np.array([bias_CSD_f1[0], bias_CSD_f04[0], bias_CSD_f005[0]])
omega_m_GLbias = np.array([bias_LGD_f1[0], bias_LGD_f04[0], bias_LGD_f005[0]])
omega_m_SLbias = np.array([bias_LSD_f1[0], bias_LSD_f04[0], bias_LSD_f005[0]])
omega_m_GLCbias = np.array([bias_LCGD_f1[0], bias_LCGD_f04[0], bias_LCGD_f005[0]])
omega_m_SLCbias = np.array([bias_LCSD_f1[0], bias_LCSD_f04[0], bias_LCSD_f005[0]])

omega_m_GCerror = np.array([np.sqrt(ifish_PD_CGPD_f1[0,0]), np.sqrt(ifish_PD_CGPD_f04[0,0]), np.sqrt(ifish_PD_CGPD_f005[0,0])])
omega_m_SCerror = np.array([np.sqrt(ifish_PD_CSPD_f1[0,0]), np.sqrt(ifish_PD_CSPD_f04[0,0]), np.sqrt(ifish_PD_CSPD_f005[0,0])])
omega_m_GLerror = np.array([np.sqrt(ifish_PD_LGPD_f1[0,0]), np.sqrt(ifish_PD_LGPD_f04[0,0]), np.sqrt(ifish_PD_LGPD_f005[0,0])])
omega_m_SLerror = np.array([np.sqrt(ifish_PD_LSPD_f1[0,0]), np.sqrt(ifish_PD_LSPD_f04[0,0]), np.sqrt(ifish_PD_LSPD_f005[0,0])])
omega_m_GLCerror = np.array([np.sqrt(ifish_PD_LCGPD_f1[0,0]), np.sqrt(ifish_PD_LCGPD_f04[0,0]), np.sqrt(ifish_PD_LCGPD_f005[0,0])])
omega_m_SLCerror = np.array([np.sqrt(ifish_PD_LCSPD_f1[0,0]), np.sqrt(ifish_PD_LCSPD_f04[0,0]), np.sqrt(ifish_PD_LCSPD_f005[0,0])])

fig1 = plt.figure(figsize=(11.0, 11.0))
ax1 = plt.gca()
ax1.plot(fskys ,abs(omega_m_GCbias/omega_m_GCerror), 'ro', label="Gaussian cluster only bias")
ax1.plot(fskys ,abs(omega_m_GLbias/omega_m_GLerror), 'bo', label = "Gaussian lensing only bias")
ax1.plot(fskys ,abs(omega_m_GLCbias/omega_m_GLCerror), 'go', label = "Gaussian lensing+clustering bias")

ax1.plot(fskys[1:] ,abs(omega_m_SCbias[1:]/omega_m_SCerror[1:]), 'r+', label="Gaussian+SSC cluster only bias")
ax1.plot(fskys[1:] ,abs(omega_m_SLbias[1:]/omega_m_SLerror[1:]), 'b+', label = "Gaussian+SSC lensing only bias")
ax1.plot(fskys[1:] ,abs(omega_m_SLCbias[1:]/omega_m_SLCerror[1:]), 'g+', label = "Gaussian+SSC lensing+clustering bias")
ax1.set_yscale('log')

plt.ylabel(r'Bias of $\Omega_M$ normalised to the 1 sigma error bar for PDCM')
plt.legend()
plt.savefig("om_bias.pdf")

#Sigma_8 plot

""" Creating data vectors for plots """

sigma_8_GCbias = np.array([bias_CGD_f1[1], bias_CGD_f04[1], bias_CGD_f005[1]])
sigma_8_SCbias = np.array([bias_CSD_f1[1], bias_CSD_f04[1], bias_CSD_f005[1]])
sigma_8_GLbias = np.array([bias_LGD_f1[1], bias_LGD_f04[1], bias_LGD_f005[1]])
sigma_8_SLbias = np.array([bias_LSD_f1[1], bias_LSD_f04[1], bias_LSD_f005[1]])
sigma_8_GLCbias = np.array([bias_LCGD_f1[1], bias_LCGD_f04[1], bias_LCGD_f005[1]])
sigma_8_SLCbias = np.array([bias_LCSD_f1[1], bias_LCSD_f04[1], bias_LCSD_f005[1]])

sigma_8_GCerror = np.array([np.sqrt(ifish_PD_CGPD_f1[1,1]), np.sqrt(ifish_PD_CGPD_f04[1,1]), np.sqrt(ifish_PD_CGPD_f005[1,1])])
sigma_8_SCerror = np.array([np.sqrt(ifish_PD_CSPD_f1[1,1]), np.sqrt(ifish_PD_CSPD_f04[1,1]), np.sqrt(ifish_PD_CSPD_f005[1,1])])
sigma_8_GLerror = np.array([np.sqrt(ifish_PD_LGPD_f1[1,1]), np.sqrt(ifish_PD_LGPD_f04[1,1]), np.sqrt(ifish_PD_LGPD_f005[1,1])])
sigma_8_SLerror = np.array([np.sqrt(ifish_PD_LSPD_f1[1,1]), np.sqrt(ifish_PD_LSPD_f04[1,1]), np.sqrt(ifish_PD_LSPD_f005[1,1])])
sigma_8_GLCerror = np.array([np.sqrt(ifish_PD_LCGPD_f1[1,1]), np.sqrt(ifish_PD_LCGPD_f04[1,1]), np.sqrt(ifish_PD_LCGPD_f005[1,1])])
sigma_8_SLCerror = np.array([np.sqrt(ifish_PD_LCSPD_f1[1,1]), np.sqrt(ifish_PD_LCSPD_f04[1,1]), np.sqrt(ifish_PD_LCSPD_f005[1,1])])


fig2 = plt.figure(figsize=(11.0,11.0))
ax2 = plt.gca()
ax2.plot(fskys ,abs(sigma_8_GCbias/sigma_8_GCerror), 'ro', label="Gaussian cluster only bias")
ax2.plot(fskys ,abs(sigma_8_GLbias/sigma_8_GLerror), 'bo', label="Gaussian lensing only bias")
ax2.plot(fskys ,abs(sigma_8_GLCbias/sigma_8_GCerror), 'go', label="Gaussian+lensing cluster only bias")

ax2.plot(fskys[1:] ,abs(sigma_8_SCbias[1:]/sigma_8_SCerror[1:]), 'r+', label="Gaussian+SSC cluster only bias")
ax2.plot(fskys[1:] ,abs(sigma_8_SLbias[1:]/sigma_8_SLerror[1:]), 'b+', label = "Gaussian+SSC lensing only bias")
ax2.plot(fskys[1:] ,abs(sigma_8_SLCbias[1:]/sigma_8_SLCerror[1:]), 'g+', label = "Gaussian+SSC lensing+clustering bias")
ax2.set_yscale('log')

plt.ylabel(r'Bias of $S_8$ normalised to the 1 sigma error bar for PDCM')
plt.legend()
plt.savefig("s8_bias.pdf")

#w_a plot

""" Creating data vectors for plots """

w_a_GCbias = np.array([bias_CGD_f1[4], bias_CGD_f04[4], bias_CGD_f005[4]])
w_a_SCbias = np.array([bias_CSD_f1[4], bias_CSD_f04[4], bias_CSD_f005[4]])
w_a_GLbias = np.array([bias_LGD_f1[4], bias_LGD_f04[4], bias_LGD_f005[4]])
w_a_SLbias = np.array([bias_LSD_f1[4], bias_LSD_f04[4], bias_LSD_f005[4]])
w_a_GLCbias = np.array([bias_LCGD_f1[4], bias_LCGD_f04[4], bias_LCGD_f005[4]])
w_a_SLCbias = np.array([bias_LCSD_f1[4], bias_LCSD_f04[4], bias_LCSD_f005[4]])

w_a_GCerror = np.array([np.sqrt(ifish_PD_CGPD_f1[4,4]), np.sqrt(ifish_PD_CGPD_f04[4,4]), np.sqrt(ifish_PD_CGPD_f005[4,4])])
w_a_SCerror = np.array([np.sqrt(ifish_PD_CSPD_f1[4,4]), np.sqrt(ifish_PD_CSPD_f04[4,4]), np.sqrt(ifish_PD_CSPD_f005[4,4])])
w_a_GLerror = np.array([np.sqrt(ifish_PD_LGPD_f1[4,4]), np.sqrt(ifish_PD_LGPD_f04[4,4]), np.sqrt(ifish_PD_LGPD_f005[4,4])])
w_a_SLerror = np.array([np.sqrt(ifish_PD_LSPD_f1[4,4]), np.sqrt(ifish_PD_LSPD_f04[4,4]), np.sqrt(ifish_PD_LSPD_f005[4,4])])
w_a_GLCerror = np.array([np.sqrt(ifish_PD_LCGPD_f1[4,4]), np.sqrt(ifish_PD_LCGPD_f04[4,4]), np.sqrt(ifish_PD_LCGPD_f005[4,4])])
w_a_SLCerror = np.array([np.sqrt(ifish_PD_LCSPD_f1[4,4]), np.sqrt(ifish_PD_LCSPD_f04[4,4]), np.sqrt(ifish_PD_LCSPD_f005[4,4])])


fig3 = plt.figure(figsize=(11.0, 11.0))
ax3 = plt.gca()
ax3.plot(fskys ,abs(w_a_GCbias/w_a_GCerror), 'ro', label="Gaussian cluster only bias")
ax3.plot(fskys ,abs(w_a_GLbias/w_a_GLerror), 'bo', label="Gaussian lensing only bias")
ax3.plot(fskys ,abs(w_a_GLCbias/w_a_GCerror), 'go', label="Gaussian+lensing cluster only bias")

ax3.plot(fskys[1:] ,abs(w_a_SCbias[1:]/w_a_SCerror[1:]), 'r+', label="Gaussian+SSC cluster only bias")
ax3.plot(fskys[1:] ,abs(w_a_SLbias[1:]/w_a_SLerror[1:]), 'b+', label = "Gaussian+SSC lensing only bias")
ax3.plot(fskys[1:] ,abs(w_a_SLCbias[1:]/w_a_SLCerror[1:]), 'g+', label = "Gaussian+SSC lensing+clustering bias")
ax3.set_yscale('log')

plt.ylabel(r'Bias of $w_a$ normalised to the 1 sigma error bar for PDCM')
plt.legend()
plt.savefig("wa_bias.pdf")

#w_0 plot

""" Creating data vectors for plots """

w_0_GCbias = np.array([bias_CGD_f1[5], bias_CGD_f04[5], bias_CGD_f005[5]])
w_0_SCbias = np.array([bias_CSD_f1[5], bias_CSD_f04[5], bias_CSD_f005[5]])
w_0_GLbias = np.array([bias_LGD_f1[5], bias_LGD_f04[5], bias_LGD_f005[5]])
w_0_SLbias = np.array([bias_LSD_f1[5], bias_LSD_f04[5], bias_LSD_f005[5]])
w_0_GLCbias = np.array([bias_LCGD_f1[5], bias_LCGD_f04[5], bias_LCGD_f005[5]])
w_0_SLCbias = np.array([bias_LCSD_f1[5], bias_LCSD_f04[5], bias_LCSD_f005[5]])

w_0_GCerror = np.array([np.sqrt(ifish_PD_CGPD_f1[5,5]), np.sqrt(ifish_PD_CGPD_f04[5,5]), np.sqrt(ifish_PD_CGPD_f005[5,5])])
w_0_SCerror = np.array([np.sqrt(ifish_PD_CSPD_f1[5,5]), np.sqrt(ifish_PD_CSPD_f04[5,5]), np.sqrt(ifish_PD_CSPD_f005[5,5])])
w_0_GLerror = np.array([np.sqrt(ifish_PD_LGPD_f1[5,5]), np.sqrt(ifish_PD_LGPD_f04[5,5]), np.sqrt(ifish_PD_LGPD_f005[5,5])])
w_0_SLerror = np.array([np.sqrt(ifish_PD_LSPD_f1[5,5]), np.sqrt(ifish_PD_LSPD_f04[5,5]), np.sqrt(ifish_PD_LSPD_f005[5,5])])
w_0_GLCerror = np.array([np.sqrt(ifish_PD_LCGPD_f1[5,5]), np.sqrt(ifish_PD_LCGPD_f04[5,5]), np.sqrt(ifish_PD_LCGPD_f005[5,5])])
w_0_SLCerror = np.array([np.sqrt(ifish_PD_LCSPD_f1[5,5]), np.sqrt(ifish_PD_LCSPD_f04[5,5]), np.sqrt(ifish_PD_LCSPD_f005[5,5])])


fig4 = plt.figure(figsize=(11.0, 11.0))
ax4 = plt.gca()
ax4.plot(fskys ,abs(w_0_GCbias/w_0_GCerror), 'ro', label="Gaussian cluster only bias")
ax4.plot(fskys ,abs(w_0_GLbias/w_0_GLerror), 'bo', label="Gaussian lensing only bias")
ax4.plot(fskys ,abs(w_0_GLCbias/w_0_GCerror), 'go', label="Gaussian+lensing cluster only bias")

ax4.plot(fskys[1:] ,abs(w_0_SCbias[1:]/w_0_SCerror[1:]), 'r+', label="Gaussian+SSC cluster only bias")
ax4.plot(fskys[1:] ,abs(w_0_SLbias[1:]/w_0_SLerror[1:]), 'b+', label = "Gaussian+SSC lensing only bias")
ax4.plot(fskys[1:] ,abs(w_0_SLCbias[1:]/w_0_SLCerror[1:]), 'g+', label = "Gaussian+SSC lensing+clustering bias")
ax4.set_yscale('log')

plt.ylabel(r'Bias of $w_0$ normalised to the 1 sigma error bar for PDCM')
plt.legend()
plt.savefig("w0_bias.pdf")
plt.show()




print cluster_GPD_fsky1.keys()
print cluster_GPD_fsky1["labels"]

