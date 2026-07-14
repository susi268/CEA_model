#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 19:08:29 2020

Get an AIA image, 

Binning in option
Plots the wavelets components in option



@author: fauchere/ Susanna
"""

import wavelets
import cv2
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from rebin import rebin
from scipy.ndimage import label
import SPUtils as spu
import SPUtils_gen as spugen

dirf = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/IMG_BLUR/'
savefts = 1
save = 0       # save png


#aia = fits.getdata(file)
dirfa = '/mnt/c/Data/SP_FITS/SDO/2018/'
file = 'aia.lev1.193A_2018-11-06T12_00_52.84Z.image_lev1.fits'
aia, h = spugen.rd_fitsf(dirfa, file)
aia[aia < 1] = 1
pxbin = 128
aia, h = spugen.image_rebin(aia, h, (pxbin, pxbin))
cmap = plt.get_cmap("sdoaia193")

nlev = 3
w = wavelets.atrous(aia, nlev)

plotw = 0
if plotw:
    fig, ax2 = plt.subplots(int((nlev+1)/2 +1), int(nlev - (nlev/2) +1), num=0,  clear=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.4)
#    for j in (0, nlev):
#        ax2[j < int((nlev+1)/2), j < int(nlev - (nlev/2) +1)].set_title('Scale level: {}' .format(j))
#        ax2.imshow(w[j, :, :], origin='lower') 
    ax2[0, 0].set_title('Scale level: {}' .format(1))
    dum = ax2[0,0].imshow(w[0, :, :], origin ='lower')
    ax2[1, 0].set_title('Scale level: {}' .format(2))
    dum = ax2[1,0].imshow(w[1, :, :], origin ='lower')
    ax2[0, 1].set_title('Scale level: {}' .format(3))
    dum = ax2[0,1].imshow(w[2, :, :], origin='lower')
    ax2[1, 1].set_title('Scale level: {}' .format(4))
    dum = ax2[1,1].imshow(w[3, :, :], origin='lower')
    ax2[2, 0].set_title('Scale level: total' )
    dum = ax2[2,0].imshow(np.sum(w, axis = 0), origin='lower')
    ax2[2, 1].set_title('Original' )
    dum = ax2[2,1].imshow(aia, origin='lower')
    fig.savefig(dirf + 'Wavelets.png', bbox_inches="tight")
    

# mask = np.uint8(np.logical_or(np.abs(w[2]) > 20, np.abs(w[1]) > 20))
mask = np.uint8(np.logical_or(np.abs(w[2]) > 17, np.abs(w[1]) > 17))

# Morphological transformation
nk = 3
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nk, nk))
mask = cv2.erode(mask, kernel, iterations=1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nk, nk))
mask = cv2.dilate(mask, kernel, iterations=1)

maxsize = 1000
regions, nregions = label(mask)
for i in range(nregions):
    r = regions == i
    if r.sum() > maxsize:
        mask[r] = 0

byte = np.log10(aia)
mini = byte.min()
maxi = byte.max()
byte -= mini
byte /= maxi - mini
byte *= 255
byte = np.uint8(byte)

# Raplace the masked pixels
inpaint = cv2.inpaint(byte, mask, 3, cv2.INPAINT_NS)
inpaint = np.float32(inpaint)
inpaint /= 255
inpaint *= maxi - mini
inpaint += mini
inpaint = 10**inpaint

fig, ax = plt.subplots(1, 3, num=1,  clear=True)

ax[0].set_title('Original binned data')
dum = ax[0].imshow(np.log10(aia), vmin=1, vmax=3, origin='lower', cmap=cmap)
ax[0].set_ylabel('Pixel')
ax[0].set_xlabel('Pixel')
#dum = ax[0].imshow(np.log10(aia), vmin=1, origin='lower', cmap=cmap)
ax[1].set_title('Final data')
ax[1].set_ylabel('Pixel')
ax[1].set_xlabel('Pixel')
dum = ax[1].imshow(np.log10(inpaint), vmin=1, vmax=3, origin='lower', cmap=cmap)
ax[2].set_title('Mask')
dum = ax[2].imshow(mask, origin='lower')
ax[2].set_ylabel('Pixel')
ax[2].set_xlabel('Pixel')

# Save
filename = 'AIA193_masked_nk{}'.format(nk)
if (savefts):
    hdu = fits.PrimaryHDU(data=inpaint, header=h)
    hdu.writeto(dirf+filename+'.fits', output_verify="ignore", overwrite=True)

if save:
    filename = filename + '.png'
    fig.savefig(dirf + filename, bbox_inches="tight")