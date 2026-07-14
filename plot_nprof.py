#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 21:03:13 2020

Plot n radial profiles created with plot1prof.
nfiles2: data files ['file1, 'file2'....]
nfiles1: PLUTO files in the same radial range.

@author: sparenti
"""
import numpy as np
import matplotlib.pyplot as plt
import SPUtils as spu
import SPUtils_AIA as spuaia
import SPUtils_gen as spugen
import scipy.interpolate as interp
import matplotlib.cm as mplcm
import SPUtils_WL as spuwl
import SPUtils_mod as spumod
import os 

def plot_nprof(sv=None, pltr2=None, pltgibs=None, err=None):
    path = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/' 
    dirf156 = os.path.join(path, 'VICTOR_WET3D_RHO3/SIMU_156_12H/LASCO_156_12h/')
#                         r'AIA_193_12h_LASCO15h/AIA_12h_LASCO15h_PSF0/')
    dirf3 =  os.path.join(path, 'AVG_PROF/')
    dirf2 = os.path.join(path, 'RES_DATA/RES_LASCO/')
    dirf1 = os.path.join(path, 'RES_DATA/RES_K-Cor/')
 
#    dirf1 = os.path.join(path, 'IMG_DIR/LAT_PROF/LASCO_LATPROF/LASCO_STR_INFO/')    # Read the input
#    dirf87 = os.path.join(path, r'VICTOR_20191106_RHO1-2/VICTOR_087_RHO2/'
#                         r'AIA_20191106_087/AIA_87_20h_K-Cor20h/')
#                           r'AIA_20191106_087/AIA_87_20h_LASCO15h/')
#                         r'LASCO_20191106_087/LASCO_RHO2_87_20h/')
#                         r'AIA_035/AIA_35_12h_LASCO15h/')
#    dirf35 = os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_035_RHO1/'
#                         r'AIA_035/AIA_35_12h_LASCO15h/')
#    dirf5 = os.path.join(dirf3,'AIA_87_20h_LASCO_OLD_CRPIX/')
    dirf = [dirf1, dirf2, dirf3, dirf3, dirf3, dirf3]
    dirsav =os.path.join(path, 'MULTI_RAD_PROF/')

    nfiles2 = ['avg_prof_K-Cor_Imax0.8_LON202.1_ST55deg_f2018-11-07.fits', 
               'avg_prof_LASCOC2_Imax0.8_LON201.7_ST68deg_f23732114pB.fits']
#    nfiles2=['avg_prof_LASCOC2_Imax0.8_LON218.3_ST70deg_f23731980pB.fits']
#    nfiles2=['avg_prof_LASCOC2_Imax0.8_LON218.3_ST110deg_f23731980pB.fits']
    nfiles1=['avg_prof_PLUTO-K-Cor_Imax0.8_LON202.1_ST50deg_f0035.fits',
             'avg_prof_PLUTO-K-Cor_Imax0.8_LON202.1_ST50deg_f0087.fits',
             'avg_prof_PLUTO-K-Cor_Imax0.8_LON202.1_ST50deg_f156.fits',
             'avg_prof_PLUTO-LASCOC2_Imax0.8_LON201.7_ST50deg_f0087.fits']
#    nfiles1 = ['avg_prof_PLUTO-LASCOC2_Imax0.8_LON218.3_ST126deg_f0156.fits',
#               'avg_prof_PLUTO-LASCOC2_Imax0.8_LON218.3_ST126deg_f0087.fits',
#               'avg_prof_PLUTO-LASCOC2_Imax0.8_LON218.3_ST126deg_f0035.fits']
            #   'avg_prof_PLUTO-LASCOC2_LON218.3_ST68deg_f035.fits']
#    nfiles3 = ['avg_prof_PLUTO-K-Cor_LON202.1_ST68deg_f0087.fits'] 
#               'avg_prof_PLUTO-K-Cor_LON202.1_ST68deg_f156.fits',             
#               'avg_prof_PLUTO-K-Cor_LON202.1_ST68deg_f0035.fits']
    
    nfiles2 += nfiles1
    labs = ['K-Cor', 'LASCO', 'Model 1' , 'Model 2', 'Model 3', '']
#    labs = ['68 (55-80)', 'Im x 0.8', '68 (55-80)', 'Im x 0.8', '68 (55-80)',
#            '68 (55-80)', 'Im x 0.8', 'Im x 0.8']
#    ['120 (100-130)', 'Im x 0.8', '102 (95-110)', 'Im x 0.8', '102 (95-110)',
#            '115 (100-130)', 'Im*0.8', 'Im x 0.8']
#    labs = ['LASCO', 'Model 3',  'Model 2', 'Model 1']

    titgen = 'nprof'
    colors = ['black', 'black', 'g', 'b', 'r', 'b', 'lime', 'g', 'c', 'lime', 'm', 'r', 'c', 'm', 'y', 'oliverab'] #'y', 'b']
    linest =['solid', 'solid', 'solid', 'solid']
    ecolors=['darkgray', 'darkgray', 'lightgreen','lightsteelblue', 'peachpuff', 'lightsteelblue', 'lightcyan',  'peachpuff']
    fig, ax = plt.subplots(1,1, num=0, clear=True)
    fontsize=11
    
    for k in range(len(nfiles2)):
        print(nfiles2[k])
        filedata = 'f' + nfiles2[k].split('f', -1)[-2]
        dirs = dirf[k]
        data, h = spugen.rd_fitsf(dirs, nfiles2[k])
        print(data.shape)
        if 'avg' in nfiles2[k]:
            lab = 'AVG'
        elif 'max' in nfiles2[k]:
            lab = 'MAX'
        elif 'sing' in nfiles2[k]:
#            lab = str(h["theta"]) +'deg' #+ labs[k]
            lab=''
        else:
            raise ValueError("Wrong file selected")
            
        titgen = titgen + '_' + h["INST"].strip() +str(h["theta"]) 
#        print(titgen)
        if 'PLUTO' in h["INST"] or 'MULTIVP' in h["INST"]:
            if 'LASCO' in h["INST"]:
                linestyle=linest[1]
            else:
                linestyle=linest[2]                 #simu    
        else:
            linestyle=linest[0]                 # Data
            
 #       label = h['inst'] + ' '+ filedata +' @ ' + str(h["theta"]) +'deg'
#       label = h['inst'] + ' '+ filedata +' @ ' + lab #labs[k]
        label = labs[k] 
        if k is 0:
            good = data[:,0] > 1.1 #2.4
        else:   
            good = data[:,1]  > 5e-8 #.2 # 1e-9
        if err is None:
            ax.plot(data[good,0], data[good,1], label=label, color=colors[k],
                    linestyle= linestyle)
        else:
            ax.errorbar(data[good,0], data[good,1], yerr=data[good,2], label=label,
                        color=colors[k], linestyle= linestyle, ecolor=ecolors[k])
        if pltr2 is True:
            label2 = 'r^2 model'
            maxv = np.argmax(data[:,1])
            r2 = 1/data[good,0]**2 * data[maxv,1] * data[maxv,0]**2
            ax.plot(data[good,0], r2, linestyle='dotted', color=colors[k], 
                    label=label2)
        ax.legend(loc='best')
#    if pltgibs is True:
#        label3 = 'Gibson 99 streamer model'
#        gibs = spumod.Gibson99_str()
#        ax.plot(gibs[:,0], gibs[:,1], linestyle='-.', color=colors[k], 
#                label=label3)
     
    if nfiles2[0].find('AIA') != -1:
        ymin = 1#-1
        ymax = 8e2 #2. #1
    elif nfiles2[0].find('LASCO') != -1:
        ymin = .1#-1
        ymax = 1e1 #2. #1
    else: 
        ymin = .1 #-1
        ymax = 1e4 #2. #1
    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
#    ax.set_xlabel(h["x_unit"])
    ax.set_xlabel('R$_\odot$', fontsize=fontsize)
    ax.set_ylabel(h["d_unit"],  fontsize=fontsize)
#    ax.set_ylabel('$10^{-10}$ B$_\odot$')
#    ax.set_title('Radial cut at PA = {:4.1f}$^\circ$'.format(h["theta"]),  fontsize=fontsize)
    ax.set_title('Average pB for streamer SI')
    if sv is True:
#        fig.savefig(dirsav +  titgen + ".png")
        print('Saved file...', titgen + ".eps")
        fig.savefig(dirsav +  titgen + ".eps", format='eps')

    return