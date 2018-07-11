import numpy as np

def write_params(year) :
    f=open("param_srd_y%d.ini"%year,"w")
    stout ="#Cosmological parameters\n"
    stout+="[om]\n"
    stout+="#Fiducial value\n"
    stout+="x= 0.3156\n"
    stout+="#Increment used for numerical derivatives\n"
    stout+="dx= 0.002\n"
    stout+="#Set to 'no' if this parameter should be keep fixed\n"
    stout+="is_free= yes\n"
    stout+="onesided=0\n"
    stout+="\n"
    stout+="[ob]\n"
    stout+="x= 0.0492\n"
    stout+="dx= 0.0002\n"
    stout+="is_free= yes\n"
    stout+="onesided=0\n"
    stout+="\n"
    stout+="[hh]\n"
    stout+="x= 0.6727\n"
    stout+="dx= 0.01\n"
    stout+="is_free= yes\n"
    stout+="onesided=0\n"
    stout+="\n"
    stout+="[w0]\n"
    stout+="x= -1.0\n"
    stout+="dx= 0.01\n"
    stout+="is_free= yes\n"
    stout+="onesided=0\n"
    stout+="\n"
    stout+="[wa]\n"
    stout+="x=0.0\n"
    stout+="dx=0.05\n"
    stout+="is_free = yes\n"
    stout+="onesided=0\n"
    stout+="\n"
    stout+="[s8]\n"
    stout+="x= 0.831\n"
    stout+="dx= 0.01\n"
    stout+="is_free= yes\n"
    stout+="onesided=0\n"
    stout+="\n"
    stout+="[ns]\n"
    stout+="x= 0.9645\n"
    stout+="dx= 0.005\n"
    stout+="is_free= yes\n"
    stout+="onesided=0\n"
    stout+="\n"
    stout+="[tau]\n"
    stout+="x=0.06\n"
    stout+="dx=0.01\n"
    stout+="is_free = no\n"
    stout+="onesided = 0\n"
    stout+="\n"
    stout+="[rt]\n"
    stout+="x=0.00\n"
    stout+="dx=0.005\n"
    stout+="is_free = no\n"
    stout+="onesided = 1\n"
    stout+="\n"
    stout+="[mnu]\n"
    stout+="x=0\n"
    stout+="dx=10.\n"
    stout+="is_free = no\n"
    stout+="onesided = 0\n"
    stout+="\n"
    stout+="[pan]\n"
    stout+="x=0.\n"
    stout+="dx=0.02\n"
    stout+="is_free = no\n"
    stout+="onesided = 1\n"
    stout+="\n"
    stout+="[Tracer 1]\n"
    stout+="tracer_name= LSST_clustering\n"
    stout+="tracer_type= gal_clustering\n"
    stout+="#File describing the redshift bins for this tracer\n"
    stout+="bins_file= curves_SRD/bins_clustering_y%d.txt\n"%year
    stout+="#Overall true-redshift distribution\n"
    stout+="nz_file= curves_SRD/nz_clustering_y%d.txt\n"%year
    stout+="#Clustering bias\n"
    stout+="bias_file= curves_SRD/bz_clustering_y%d.txt\n"%year
    stout+="#Magnification bias\n"
    stout+="sbias_file= nothing\n"
    stout+="#Evolution bias\n"
    stout+="ebias_file= nothing\n"
    stout+="lmin= 20\n"
    stout+="lmax= 5400\n"
    stout+="use_tracer= yes\n"
    stout+="\n"
    stout+="[Tracer 2]\n"
    stout+="tracer_name= LSST_shear\n"
    stout+="tracer_type= gal_shear\n"
    stout+="bins_file= curves_SRD/bins_shear_y%d.txt\n"%year
    stout+="nz_file= curves_SRD/nz_shear_y%d.txt\n"%year
    stout+="#Alignment bias (only needed if including IA)\n"
    stout+="abias_file= nothing\n"
    stout+="#Red (aligned) fraction (only needed if including IA)\n"
    stout+="rfrac_file= nothing\n"
    stout+="#Intrinsic ellipticity dispertion\n"
    stout+="sigma_gamma= 0.26\n"
    stout+="lmin= 20\n"
    stout+="lmax= 5400\n"
    stout+="use_tracer= yes\n"
    stout+="\n"
    stout+="[CLASS parameters]\n"
    stout+="#File with bandpowers. Two columns corresponding to the l_min and l_max(inclusive) of each bandpower\n"
    stout+="bandpowers_file= curves_SRD/larr.txt\n"
    stout+="#Maximum multipole for which the CMB Cls will be computed\n"
    stout+="#(only relevant if CMB primary or CMB lensing are included)\n"
    stout+="lmax_cmb= 5000\n"
    stout+="#Maximum multipole for which the clustering and lensing Cls will be computed\n"
    stout+="#(only relevant if galaxy clustering or shear are included)\n"
    stout+="lmax_lss= 5400\n"
    stout+="#Minimum multipole from which Limber will be used\n"
    stout+="lmin_limber= 0\n"
    stout+="#Include intrinsic alignments in the shear power spectra?\n"
    stout+="include_alignment= no\n"
    stout+="#Include RSDs in the clustering power spectra?\n"
    stout+="include_rsd= no\n"
    stout+="#Include lensing magnification in the clustering power spectra?\n"
    stout+="include_magnification= no\n"
    stout+="#Include relativistic effects in the clustering power spectra?\n"
    stout+="include_gr_vel= no\n"
    stout+="include_gr_pot= no\n"
    stout+="#Command used to execute classt (it should take the CLASS param file as an argument)\n"

    stout+="exec_path= addqueue -q cmb -s -n 1x12 -m 1.8 ./class_mod\n"
    #stout+="exec_path= ./class_mod\n"
    stout+="#Use non-linear matter transfer function (HALOFit)?\n"
    stout+="use_nonlinear= yes\n"
    stout+="#Set to \"yes\" if you want to include baryonic effects in the power spectrum\n"
    stout+="use_baryons= no\n"
    stout+="#Sky fraction (20,000 sq deg for LSST)\n"
    if year==1 :
        stout+="f_sky= 0.35\n"
    else :
        stout+="f_sky= 0.42\n"
    stout+="\n"
    stout+="[Output parameters]\n"
    stout+="#Directory where all the data will be output\n"
    stout+="output_dir= outputs_y%d\n"%year
    stout+="#Prefix of all power-spectrum-related output files\n"
    stout+="output_spectra= run\n"
    stout+="#Directory where the Fisher information and plots will be stored\n"
    stout+="output_fisher= Fisher\n"
    stout+="\n"
    stout+="[Behaviour parameters]\n"
    stout+="#Cosmological  model\n"
    stout+="model= wCDM\n"
    stout+="#Use [Omega_c*h^2,Omega_b*h^2,A_s]? If no, the code will use [Omega_m,Omega_b,sigma_8]\n"
    stout+="use_cmb_params= no\n"
    stout+="#Do you wanna keep the Cl files?\n"
    stout+="save_cl_files= yes\n"
    stout+="#Do you wanna keep the CLASS param files?\n"
    stout+="save_param_files= yes\n"
    stout+="#Do you want to just generate the power spectra and not the Fisher matrix?\n"
    stout+="just_run_cls= yes\n"
    stout+="#File containing a \"measured\" power spectrum to estimate the associated parameter bias\n"
    stout+="bias_file= none\n"
    stout+="\n"
    f.write(stout)
    f.close()

write_params(1)
write_params(10)
