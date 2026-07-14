#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:59:00 2020

data1: model
data2: data

@author: sparenti
"""
#from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import SPUtils_gen as spugen
#import scipy.interpolate as interp
import os

def plot_nratio(nfiles=None, dirs=None, sv=None):
    
    if dirs is None:
#        dirsav = "/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/"
        path = '/mnt/c/Users/sarenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/' 
        dirf3 =  os.path.join(path, 'AVG_PROF/')
        dirf2 = os.path.join(path, 'RES_DATA/RES_LASCO/')
        dirsav = os.path.join(path, 'MULTI_RAD_PROF/')
        dirf87 =  os.path.join(path, r'VICTOR_20191106_RHO1-2/VICTOR_087_RHO2/'
                              r'AIA_20191106_087/AIA_87_20h_LASCO15h/')
        dirf35 =  os.path.join(path, r'VICTOR_20191106_RHO1-2/VICTOR_035_RHO1/'
                              r'AIA_035/AIA_35_12h_LASCO15h/')
        dirf157 =  os.path.join(path, 'VICTOR_WET3D_RHO3/SIMU_156_12H/'
                               r'AIA_193_12h_LASCO15h/AIA_12h_LASCO15h_PSF0/')
#        dirf1 = os.path.join(path, 'RES_DATA/RES_K-Cor/') # Read the input
        dirs = [dirf3,dirf3, dirf3, dirf3]
        
    if nfiles is None:
        nfiles2 = ['avg_prof_LASCOC2_LON201.7_ST64deg_f23732114pB.fits',
                   'avg_prof_LASCOC2_LON218.3_ST62deg_f23731980pB.fits',
                   'avg_prof_LASCOC2_Imax0.8_LON201.7_ST68deg_f23732114pB.fits',
                   'avg_prof_LASCOC2_Imax0.8_LON218.3_ST70deg_f23731980pB.fits']

        nfiles1 = ['avg_prof_PLUTO-LASCOC2_LON201.7_ST72deg_f0087.fits',
                   'avg_prof_PLUTO-LASCOC2_LON218.3_ST76deg_f0087.fits',
                   'avg_prof_PLUTO-LASCOC2_Imax0.8_LON201.7_ST50deg_f0087.fits',
                   'avg_prof_PLUTO-LASCOC2_Imax0.8_LON218.3_ST70deg_f0087.fits']
    
    fig, ax = plt.subplots(1,1, num=0, clear=True)
    titgen = 'ratio'
    colors = ['black', 'g', 'black','g', 'r', 'g', 'b', 'r', 'b','c', 'm', 'y']
    fontsize = 11
#    label2 = ['SI', 'SII' ]
    labels = ['SI (Fix, Nov. 7) LASCO / Model2', 'SI (Fix, Nov. 6) LASCO / Model2', 'SI (Nov. 7) LASCO / Model2',
              'SI (Nov. 6) LASCO / Model1', 'SII LASCO / Model2', 'SII LASCO / Model3',]
#    labels = ['Model2 / Model1', 'Model2 / Model3']

    if len(nfiles1) != len(nfiles2) and len(nfiles2) > 2:
           raise Exception("nfiles1 and nfiles2 should have the same size")
        
    for k in range(len(nfiles1)):
        data1, h1 = spugen.rd_fitsf(dirs[k], nfiles1[k])
        if len(nfiles2) == 1:
            j = 0
        else:
            j = k
        data2, h2 = spugen.rd_fitsf(dirf2, nfiles2[j])
#        data2, h2 = spugen.rd_fitsf(dirf87, nfiles2[j])
#   
#        print(data1[:,0].max(), data1[:,0].min())
#        print(data2[:,0].max(), data2[:,0].min())

        if np.logical_and(nfiles1[k].find('avg'), nfiles2[j].find('avg')) != -1:
            label1 = 'AVG '
        else: label1 = ""
        filedata1 = ' f' + nfiles1[k].split('f', -1)[-2]
#       label = (label1 + h2['inst'] + '/' + h1['inst'] + '@ ' 
#                 + str(h1["theta"]) +'deg ' + filedata1)
 
#        titgen = titgen + '_' + h1["INST"].strip() + h2["INST"].strip() + \
#               '_theta'+ str(h1["theta"])
        ratio, dist = spugen.plot_prof_ratio(data1, h1, data2, h2)
        if k < 2:
            linestyle = 'solid'
        else:
            linestyle = 'dashed'
        if h1["INST"].find('K-Cor') is True:
            ax.plot(dist[dist>1.12], ratio[dist>1.12], label=labels[k], color=colors[k],
                    linestyle=linestyle)
        else:
            ax.plot(dist[dist>2.4], ratio[dist>2.4], label=labels[k], color=colors[k],
                    linestyle=linestyle)
            
 # plot the line = 1
    line1 = np.ones(2)
    xline1 = [dist.min(), dist.max()]
#    ax.plot(xline1,line1,  '--', color='gray')

    ax.legend(loc='best')
    
#   titgen = titgen + '_LASCO_' + 'LON' + str(int(np.degrees(h1['LON']))) + '_'
    titgen = titgen + '_' + h2["INST"].strip() + '_' + h1["INST"].strip() + \
                   '_LON218_theta'+ str(h1["theta"])
#            '_LON' + str(int(np.degrees(h1['LON']))) + '_theta'+ str(h1["theta"])


  
#    titgen = 'ratio_Model2' + '_LON218_theta'+ str(h1["theta"])
#    titgen = titgen + '_' + h2["DATA_F"].split("_")[1] 
    print(titgen)
    ax.set_ylim(0, 1.)
#    ax.set_xlabel(h1["x_unit"])
    ax.set_xlabel('R$_\odot$', fontsize=fontsize)
    ax.set_ylabel('DATA / MODEL', fontsize=fontsize)
#    ax.set_ylabel('MODEL 2 / MODEL', fontsize=fontsize)
    
    strtit =  h1["DATA_F"].split("_") 
#    ax.set_title(strtit[1]) #+ ' LON ' + str(int(np.degrees(h1['LON']))))
    ax.set_title('', fontsize=fontsize)
    if sv is True:
#        fig.savefig(dirsav +  titgen + ".png")
        print('Saved file...', titgen + ".eps")
        fig.savefig(dirsav +  titgen + ".eps", format='eps')       
    return