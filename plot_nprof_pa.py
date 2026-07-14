#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:52:55 2020

Give a set of plot which compares the data and models intensities along the PA
filed: files containing the latitudinal profile of the DATA 
str_data: string to select the simulation files. It should contain the instrument
str_data: "latprof*LASCO*218*.fits"

@author: sparenti
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import Profiles_class as profcls
import SPUtils_WL as spuwl
import SPUtils_AIA as spuaia
import os

def plot_nprof_pa(filed=None, str_data=None, sv=None, err=None):
#    path = (r"/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/"
#            r"CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/")
    path = '/home/sparenti/WORK/GLOBAL_MODEL/PERI2_OUT/'
#    dirf = os.path.join(path, 'LAT_PROF/AIA_LATPROF/')    # Read the input
#    dirs = ['LAT_PROF/K-Cor_LATPROF/', 'LAT_PROF/LASCO_LATPROF/LON218/', 'LAT_PROF/AIA_LATPROF/']
    dirs = ['LAT_PROF/', 'LAT_PROF/']
#    dirf = ['LAT_PROF/K-COR_LATPROF/', 'LAT_PROF/LASCO_LATPROF/']         # files simulation
#    dird = ['RES_DATA/RES_K-COR/', 'RES_DATA/RES_LASCO/', 'RES_DATA/RES_AIA193/']
    dird = ['RES_DATA/', 'RES_DATA/']
    dirf = dirs
    
    if filed is not None:
        if filed[0].find('LASCO') != -1:
            i = 1
        elif filed[0].find('K-Cor') != -1:
            i = 0
        else:
            i=2
#            raise  ValueError('Intrument still to be implemented')
    else:
        if str_data is None:
            raise  ValueError('Need to pass either FILED or STR_DATA')
        else:
            if ((str_data.find("LASCO")  == -1) and
                (str_data.find("K-Cor") == -1) and
                (str_data.find("AIA") == -1)):
                raise  ValueError('Need to pass STR_DATA with instrument')
            if str_data.find("LASCO") != -1:
                i = 1
            elif str_data.find('K-Cor') != -1:
                i = 0
            else:
                i = 2
        print(i)
        filed = glob.glob(os.path.join(path, dird[i]) + str_data)
        if len(filed) > 0:
            print(len(filed), ' ..files found')
        else:
            print('No data files found')
        for ii in range(len(filed)):
            filed[ii] = os.path.basename(filed[ii])
        filed.sort()
#        print('files', filed)

            
 # open the data file and prepare the plot
#    colors = ['r', 'g', 'c', 'r', 'y'] #, 'm', 'b']
    colors = [ 'b',  'darkorange', 'g']
    ecolor=['lightsteelblue', 'peachpuff','lightgreen']
#    ecolor=['lightgreen','lightcyan','peachpuff']
    labels = ["No AR", "With AR", "Model 3" ]
    fonts = 10
    fig, ax = plt.subplots(2, 3, num=0, clear=True, figsize=(9, 5))   #K-Cor
#    fig, ax = plt.subplots(2, 2, num=0, clear=True, figsize=(9, 5))
    ax = ax.ravel()
    for j in range(len(filed)):
        print('Distance', j)
        if str_data.find("AIA")  == -1:
            str_fl = ('*' + filed[j].split('_')[2] + '*' +
                    filed[j].split('_')[3] + '*.fits')
            print('String to be searched:', str_fl)
        else:
            str_fl = ('*' + filed[j].split('_')[4] + '*' +
                      filed[j].split('_')[5] + '*.fits')
            print(str_fl, filed[j].split('_'))
# get the simu files for this data
        files = glob.glob(os.path.join(path, dirf[i]) + str_fl)
        print('Model files..', len(files))
        if str(26) in files[0] :
            files.reverse()
            print('reversed files', files)
# profile from  the data
        data_prof = profcls.profiles(os.path.join(path, dird[i]), filed[j])
        labeld = data_prof.inst
        if np.logical_or(data_prof.inst.find('K-Cor') != -1, 
                         data_prof.inst.find('LASCO') != -1):
            ax[0].legend(ncol=3, bbox_to_anchor=(2.8, 1.4), loc='upper right', fontsize=9)
#            ax[j].set_ylim(0, max(data_prof.data) *4)           #LASCO
            ax[j].set_ylim(0, max(data_prof.data)*1.5)            #k-cor
#            ax[j].set_xlim(0, 160)
            if j >= 3:
                ax[j].set_xlabel('Position Angle ($^\circ$)', fontsize=fonts)
            if (j == 0) or (j == 3):
                ax[j].set_ylabel('$10^{-10}$ B$_\odot$',fontsize=fonts)
            if err is True:
                if data_prof.inst.find('K-Cor') != -1:
                    errd = spuwl.get_KCor_perr(data_prof.data)
                else: 
                    errd = spuwl.get_LASCO_pBerr(data_prof.data, data_prof.x)
        else:
            labeld = 'AIA'
            ax[0].legend(ncol=4, bbox_to_anchor=(2., 1.4), loc='upper right', fontsize=fonts)  
            if (j == 0) or (j == 2):
                ax[j].set_ylabel(data_prof.data_u,fontsize=fonts)
            if j > 1: 
                ax[j].set_xlabel('Position Angle ($^\circ$)', fontsize=fonts)    
            if err is True:
                errd = spuaia.get_perr(data_prof.data, '193')
# plot                
        if err is None:
            ax[j].plot(np.degrees(data_prof.x), data_prof.data, color='black', label=labeld)
        else:
            ax[j].errorbar(np.degrees(data_prof.x), data_prof.data, \
              yerr=errd, color='black', label=labeld, ecolor='darkgray')
        ax[j].set_title("R$_\odot$= {}".format(data_prof.rsun), fontsize=11)
#        if (j is 4) or (j is 5):
#        if j >= 3: 
#            ax[j].set_xlabel('Position Angle ($^\circ$)', fontsize=9)
#        if (j is 0) or (j is 3): #or (j is 4): 
#            ax[j].set_ylabel(data_prof.data_u, fontsize=8)
#             ax[j].set_ylabel('$10^{-10}$ B$_\odot$',fontsize=9)

# profiles fromthe models  
        for k in range(len(files)):
            this_m = profcls.profiles('', os.path.join(path, files[k]))
            print('Distances:', this_m.rsun, data_prof.rsun)
            if this_m.rsun != data_prof.rsun:
                raise ValueError("The data and model files are not at the same distance")
                
            if err is True: 
                if this_m.inst.find('AIA') !=-1:
                     errm = spuaia.get_aia_cerr(this_m.data)
                else:
                     errm = np.zeros_like(this_m.data)
#            print('Plotting: ', this_m.file)
#            label = this_m.file.split('_')[-1]
#            label = label.split('.')[0]
#            label = this_m.inst + ' ' + this_m.file
            label = labels[k]
            if err is True:
                ax[j].errorbar(np.degrees(this_m.x), this_m.data, yerr=errm, color=colors[k], 
                  label=label, ecolor=ecolor[k])
            else:   
                ax[j].plot(np.degrees(this_m.x), this_m.data, color=colors[k], label=label)
#    ax[0].legend(ncol=4, bbox_to_anchor=(0.5, 1.4), loc='center right', fontsize=9)
#    ax[0].legend(ncol=4, bbox_to_anchor=(2.8, 1.4), loc='upper right', fontsize=9)   # K-cor
#    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=.3)
    fig.show()
    if sv is True:
        filegif='nprof_pa_' + data_prof.inst + '_' + "LON{:.1f}".format(np.degrees(data_prof.lon))
        fig.savefig(os.path.join(path, dirs[i]+filegif  + ".png"))
    return 