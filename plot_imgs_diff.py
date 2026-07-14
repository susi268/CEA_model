#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:09:31 2020

Plot the relative difference of two images. 
pxbad = to select only part of the images to calcule the error nrse format[y1, Y2, x1, x2]

Image1: PLUTO
Image2: DATA

@author: sparenti
"""

import SPUtils_gen as spugen
import matplotlib.pyplot as plt
import os
import numpy as np

#path = (r"/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/"
 #       r"CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/")
path = ("/home/sparenti/WORK/GLOBAL_MODEL/")

#dirf2 = '/mnt/c/Data/SP_FITS/SDO/2018/'
#file2 = 'aia.lev1.193A_2018-11-06T15 00 52.84Z.image_lev1.fits'
#dirf2 = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/FITS/LASCO/'
dirf2 = '/home/sparenti/FITS/LASCO/'
#file2 = '20181107_201803_kcor_l1.5_extavg.fts'
file2 = '23879116pB.fts'
tit2 = "LASCO"

#dirf1 = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/VICTOR_WET3D/AIA_20H/SYN_AIA/'
#file1='PLUTO-AIA_2_NAX128_PSHAPE7.0_VOX810_PSF0_LON218.2_f87.fits'
#file1='PLUTO-AIA_2_NAX128_PSHAPE7.0_VOX810_PSF0_LON218.2_f35.fits'
#file1='PLUTO-AIA_2_NAX128_PSHAPE7.0_VOX806_PSF0_LON218.2_f156.fits'
#file1 = 'PLUTO-K-Cor_20h_NAX1024_PSHAPE24.0_VOX450_LON202.1_f0087.fits'
#file1 = 'PLUTO-LASCO_NAX512_PSHAPE49.6_VOX562_LON218.3_f0035.fits'
file1= 'PLUTO-LASCO_NAX512_PSHAPE50.0_VOX128_LON218.3_f0010.fits'
#dirf1 = os.path.join(path,'VICTOR_WET3D_RHO3/SIMU_156_12H/K_Cor_156_12h/')
#dirf1 = os.path.join(path,'VICTOR_WET3D_RHO3/SIMU_156_12H/AIA_193_12h_LASCO15h/AIA_12h_LASCO15h_PSF0/')
#dirf1 = os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_035_RHO1/'
#                     r'AIA_035/AIA_35_12h_LASCO15h/')
#                            r'LASCO_035/LASCO_RHO1_35_12h/')
#                     r'K-Cor_35/')
dirf1 = os.path.join(path, 'PERI2_OUT/')
#dirf1 = os.path.join(path, (r'VICTOR_20191106_RHO1-2/VICTOR_087_RHO2/K-Cor_20191106_087/'))
#                                r'AIA_20191106_087/AIA_87_20h_LASCO15h/'))
#dirf1 = os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_035_RHO1/AIA_035/AIA_35_12h_LASCO15h/')
                     
sav = None
#pxbad=[0, 128, 0, 64]
#pxbad=[0, 512, 0, 256]
#pxbad=[0, 512, 256, 512]
#pxbad=[0, 1024, 0, 512]
pxbad=[750, 1024,  0, 1024]
#pxbad=None

img1, h1, img2, h2 = spugen.open2imgs(dirf1, file1, dirf2, file2)

#tit2 = file1.split(".")[0] + "_" + h2["INSTRUME"]
tit2 = h2["INSTRUME"] + "_" + file1.split(".")[0]
#tit1 = h1["PLUTOFL"].split('.', -1)[1]
tit1 = h1["MODELFL"].split('.', -1)[1]
#titlep =  tit2 + " / " + tit1
if h1["WAVELNTH"] == 'logN':
    tit3 = 'Ne'
else:
    tit3 ='pB'
#title = tit2 + '_' + tit1 + '_' + tit3 
title = 'LASCO /PLUTO'
# This will do img2/img1
ratio = spugen.plot_imgs_ratio(img1, img2, h=h1, title=title, sv=sav, pxbad=pxbad)
print(tit2)
print(tit1)
#plt.ion()
fig2 = plt.figure(num=1, clear=True)
plt.imshow(np.log10(img2))
#plt.show()
#fig3 = plt.figure(num=2, clear=True)
#plt.imshow(np.log10(img2))
#ax2.set_title(file1)
##plt.hist(ratio, bins=50, range=(0, np.max(ratio)))
##plt.hist(ratio, bins=100)
#ax2.set_ylabel('Pixel frequency')
#ax2.set_xlabel(titlep)

    