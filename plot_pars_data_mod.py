#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:43:01 2020

Compares the PA,... with the data
filed: 'sing_prof.....fits' of the DATA
long: string to select the model files froma dir. It selects the longitude of the image.

@author: sparenti
"""
import numpy as np
import matplotlib.pyplot as plt
import SPUtils_gen as spugen
import os
import glob

def plot_pars_data_mod(filed, long='*218*87*.fits'):
    
    path = (r"/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/"
            r"CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/")
    dirf = os.path.join(path, 'LAT_PROF/LASCO_LATPROF/LON218/')    # Read the input
    dirs = ['LAT_PROF/K-Cor_STR_INFO/', 'LAT_PROF/LASCO_LATPROF/LASCO_STR_INFO/']
    dird = ['RES_DATA/RES_K-COR/', 'RES_DATA/RES_LASCO/']
    
    if filed.find('LASCO') != -1:
        i = 1
    elif filed.find('K-Cor') != -1:
        i = 0
    else:
        raise  ValueError('Intrument still to be implemented')
    
    files = glob.glob(os.path.join(path, dirs[i]) + long)

# open the data file and prepare the plot
    colors = ['g', 'c', 'r', 'y', 'm', 'b']
    fig, ax = plt.subplots(1, 1, num=0, clear=True)

    image, h = spugen.rd_fitsf(os.path.join(path, dird[i]), filed)
    ax.plot(np.degrees(image[:, 3]), np.degrees(image[:, 3]), '--', color='darkorange')
    ax.plot(np.average(np.degrees(image[:, 3])), np.average(np.degrees(image[:, 3])), 
            'p', color='darkorange', mfc='none', label='AVG')
    ax.set_xlabel("PA-LASCO ($\deg$)")
    ax.set_ylabel("PA-model ($\deg$)") 
    
    ax2 = ax.twiny()
    ax2.set_xlabel("R$_\odot$")
    ax2.set_xticks(np.degrees(image[:, 3]))
    ax2.set_xticklabels(image[:, 0])
    ax2.set_xlim(ax.get_xlim())

    for k in range(len(files)):
        this_mod, this_h = spugen.rd_fitsf('', files[k])
        this_l = this_h["DATA_F"].split('_')[-1]
    #    if np.max(image[:, 2]) == 0:
        ax.plot(np.degrees(image[:, 3]), np.degrees(this_mod[:, 3]), "o",
                    label=this_l, color=colors[k])
        ax.plot(np.average(np.degrees(image[:, 3])), np.average(np.degrees(this_mod[:, 3])), 
                'p', color=colors[k], mfc='none')

#    else:
#        ax.errorbar(np.degrees(image[:, 3]), np.degrees(this_mod[:, 3]),
#                    yerr=image[:, 2], label=this_l)
    ax.legend(loc='best')
    
    
