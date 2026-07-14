#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:20:40 2020

By Adrian Rosebrock on September 15, 2014 in Image Processing, Tutorials 
https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/

Image2 is the reference image which may have higher resolution

@author: sparenti
"""
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import SPUtils_gen as spugen
import SPUtils_AIA as spuaia
import SPUtils as spu
import scipy.ndimage

def mse(image1, image2):

    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
#    err = (np.sum(image1) - np.sum(image2))**2 / 2
    err /= float(image1.shape[0] * image2.shape[1])
    return err
    # return the MSE, the lower the error, the more "similar" the two images are


save = 0
# Get info image2 - AIA
dirf2 = '/mnt/c/Data/SP_FITS/SDO/2018/'
file2 = 'aia.lev1.193A_2018-11-06T20_00_52.84Z.image_lev1.fits'
dataf2, h2 = spugen.rd_fitsf(dirf2, file2)
tit2 = 'AIA20h'


# Get info image1 - SIMU
#dirf1 = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/VICTOR_20191106/'
dirf1 = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/VICTOR_WET3D/AIA_20h/'

#file1 = 'AIASynth193_VOX128_LON10.0_PSF1.fits'
file1 = 'AIA193_20h_VOX128_PSHAPE6.35_LON0.0_PSF1_WET.fits'
dataf1, h1 = spugen.rd_fitsf(dirf1, file1)
dataf1 *= h2["EXPTIME"]
tit1 = 'PLUTO WET3D'

#dataf1 /= np.max(dataf1)    # normalization to the max
print("===============================================")
print('Header units: {}'.format(h1['D_UNITS']))
h1["D_UNITS"] = "Dn"
print('New header units: {}'.format(h1["D_UNITS"]))
print("===============================================")

aia_obj = spu.aia_obj()
cmap = aia_obj.filter_cmap_dict[str(h1["WAVELNTH"])]

# Add check on need for binning AIA
dataf2_bin, h2 = spugen.image_rebin(dataf2, (h1["NAXIS1"],
                                                 h1["NAXIS2"]), h=h2)

# normalization to be removed.
#dataf2_bin /= np.max(dataf2_bin)

if dataf2_bin.shape != dataf1.shape:
    raise Exception("The two images have different shape")

for i in range(2):
    if h1['NAXIS' + str(i + 1)] != h2['NAXIS' + str(i + 1)]:
        raise Exception("The two images have different NAXIS.")
    elif h1['CRPIX' + str(i + 1)] != h2['CRPIX' + str(i + 1)]:
        raise Exception("The two images have different CRPIX.")
#    elif h1['CDELT' + str(i + 1)] != h2['CDELT' + str(i + 1)]:
#        raise Exception("The two images have different CRDEL.")
    elif h1['CRVAL' + str(i + 1)] != h2['CRVAL' + str(i + 1)]:
        raise Exception("The two images have different CRVAL.")

#ker = (0, 0)
#sigm = 1
#size = sigm
#sz = 2
#level = 4
#dataf2_bin_blr = spugen.img_gauss_flt(dataf2_bin, ker, sigm, cmap=cmap)
#dataf2_bin_med= spugen.img_median_flt(dataf2_bin, size, cmap = cmap)
#dataf2_bin_spk = spugen.im_med_spike(dataf2_bin, sz, level, cmap=cmap, save=True)

plot2prof(dataf1, h1, dataf2_bin, lat=64,  tit1=tit1, tit2=tit2, sv=None)

err = mse(dataf1, dataf2_bin)
#s = ssim(dataf1, dataf2_bin)

# setup the figure
fig = plt.figure(num=3)
# plt.suptitle("MSE: %.2f, SSIM: %.2f" % (err, s))
plt.suptitle("Mean Squared Error: %.2f \n Structural Similarity Measure: " % (err))
# plt.suptitle("Structural Similarity Measere: %.2f" % (s))

# show first image
ax = fig.add_subplot(1, 2, 1)
plt.imshow(dataf1**0.5, cmap=cmap, origin = 'lower')

# show the second image
ax = fig.add_subplot(1, 2, 2)
plt.imshow(dataf2_bin**0.5, cmap=cmap, origin='lower')
#plt.show()

if save:
    dirf = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/IMG_BLUR/'
    fig.savefig(dirf + "AIA{}".format(str(h1["WAVELNTH"])) + "{}"
                .format(h1["NAXIS1"]) +"_WET3D.png", bbox_inches="tight")
#plt.close()