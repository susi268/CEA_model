#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:56:39 2020

@author: sparenti

Various codes which deal with AIA data

"""

# import PlutoUtils as pu
import numpy as np
from astropy.io import fits
import scipy.interpolate as interp
#import sunpy.cm as cm
import sunpy.visualization.colormaps.cm as cm
import matplotlib.pyplot as plt
import fitsarray as fa
import SPUtils_gen as spugen
import tomograpy
import scipy.ndimage as ndimage



def image_header_from_aia(dirf, aiaf, pxbin, dtype=np.float64):
    """
    Generate a cubic map header from an AIA header

    Arguments
    ---------
    dirf, aiaif : location and AIA filename
    pxbin : bin applied to the AIA image
    
    Output
    ------
    header: pyfits header
    """
# gets aia info
    dataf, h = spugen.rd_fitsf(dirf, aiaf)
    
# generate header
    header = dict()
    if 'DATE_OBS' in h.keys():
        header['DATE_OBS'] = h["DATE_OBS"]
    else:
        header['DATE_OBS'] = h["DATE-OBS"]
    header['SIMPLE'] = True
    header['BITPIX'] = fa.bitpix_inv[dtype.__name__]
    header["MODELFL"] = ''
    header['NAXIS'] = 2
    header['INSTRUME'] = 'PLUTO-' + h["INSTRUME"]
    if 'DETECTOR' in h.keys():
        header['DETECTOR'] = h["DETECTOR"]
    else:
        header['DETECTOR'] = ""
    header['WAVELNTH'] = h["WAVELNTH"]
    header['PIXBIN'] = pxbin                          # Binning applied to AIA
#    header['EXPTIME'] = h["EXPTIME"]
    for i in range(2):
        header['NAXIS' + str(i + 1)] = round(h['NAXIS' + str(i + 1)] / pxbin)
#        header['CDELT'+ str(i + 1)] = np.arctan2(h["CDELT"+ str(i + 1)] * h["RSUN_REF"] / h["RSUN_OBS"]
#                                    * float(pxbin), h["DSUN_OBS"]) #*1.005)   # radians
        header['CDELT'+ str(i + 1)] = np.radians(h["CDELT"+ str(i + 1)] /3600.)
        header["CRVAL" + str(i + 1)] =  np.radians(h['CRVAL' + str(i + 1)]/3600.) / float(pxbin)
#        header['CRPIX' + str(i + 1)] = (h['CRPIX' + str(i + 1)] / pxbin ) # + 0.5)
        header['CRPIX' + str(i + 1)] = (h['CRPIX' + str(i + 1)] -1) / pxbin + 1


    header['LAT'] = np.radians(h["CRLT_OBS"])          # Assumes only 1 LAT
    header['LON'] = np.radians(h["CRLN_OBS"])          #np.radians(h["CRLN_OBS"])  # Assumes only 1 LON
    header["RSUN_OBS"] = h["RSUN_OBS"]                 # in arcsec
    header["radius"] = h["DSUN_OBS"] / h["RSUN_REF"]   # In R_sun
    header["PSF_COR"] = 0                           # True if the model include PSF
    header["CRTLN_0"] = None                        # Will set the CRTLN of the meridian of the magnetic map
    tomograpy.siddon.map_borders(header)
    return header


# Prep AIA DATA
def get_aia_prep(data,h):
    h["radius"] = h["DSUN_OBS"] / h["RSUN_REF"]   # In R_sun
    data /= h["EXPTIME"]
    print("AIA prep:")
    print('AIA original data units: {}'.format(h['PIXLUNIT']))
    h["D_UNIT"] = "Dn/s"
    print('New data header units: {}'.format(h["D_UNIT"]))
    h["CDELT1"] = np.radians(h["CDELT1"]/3600.)
    h["CDELT2"] = np.radians(h["CDELT2"]/3600.)
    h["LON"] = np.radians(h["CRLN_OBS"])
    print('New header keyword added: LON ', np.degrees(h["LON"]))
    print("AIA CDELT now in radians")
    return data, h

# Apply a dark spherical mask to an image
def apply_aia_mask(image_header, mask_start):
    x, y = np.ogrid[:image_header["NAXIS1"], :image_header["NAXIS2"]]
    x = (x - image_header["CRPIX1"]) * image_header["CDELT1"] + image_header["CRVAL1"]
    y = (y - image_header["CRPIX2"]) * image_header["CDELT2"] + image_header["CRVAL2"]
    r_x = image_header["radius"] * np.tan(x)
    r_y = image_header["radius"] * np.tan(y)
    r = np.sqrt((r_x - image_header["CRVAL1"])**2 +(r_y - image_header["CRVAL2"])**2)
    aia_mask = r < mask_start
    print("//////////////////////////////////////////")
    print("Dark mask applied for R {}".format(mask_start)) 
    print("//////////////////////////////////////////")
    return aia_mask


# Convolve an image with the AIA PSF (from Fred files)
# 'psfile' can be binned to the image size using the 'pxbin' keyword
def conv_psf_aia(image, psfile):
    dirfits = '/mnt/c/Users/Susanna Parenti/Documents/SOFTS/IDL_soft/AIA/FRED_PSF/'
    aia_psf, h = spugen.rd_fitsf(dirfits, psfile)
    pxbin = h["NAXIS1"] / image.shape[0]
    if (pxbin) != 1:
        aia_psf, h = spugen.image_rebin(aia_psf, image.shape, h=h, pxbin=pxbin)
        aia_psf[:] /= np.sum(aia_psf)               # normalization
    print('Total aia_psf:{}'.format(aia_psf.sum))
    for j in range(image.shape[2]): 
        image[:,:,j] = ndimage.convolve(np.squeeze(image[:,:,j]), aia_psf, mode='constant', cval=0.0)
    return image


# get the AIA calibration + CHIANTI errors (35%) 
# image: synthetic image
def get_aia_cerr(image):
    cerr_image = image * .35
    return cerr_image


# Add the poisson noise to an image and/or calculate the error
# the conversion Dn/ph is given in
# in ssw aia_V8_20171210_050627_response_table.txt
# noise: replace the input image with a noisy one 
def get_perr(image, band, noise=None):
    dn_ph = {'171':1.1215, '193':0.9513, '211':0.8784}
    image_ph = image / dn_ph[band] # convert DN to phot
#    if noise: noimage_ph = np.random.poisson(image_ph)
#    else: noimage_ph = None 
    perr_image = np.sqrt(image_ph) * dn_ph[band]
    return perr_image #, noimage_ph


def encode_aia(inpath, filename):
    import subprocess
    import os
    import pty
    p = subprocess.Popen(["ffmpeg",
                          "-framerate 25",
                          "-i " + inpath + os.sep + "*_%3d.png",
                          "-vcodec libx264",
                          "-pix_fmt yuv420p",
                          "-preset slow",
                          "-crf 17",
                          "-y " + filename])

    try:
        outs, errs = p.communicate(timeout=15)
    except TimeoutError:
        p.kill()
        outs, errs = p.communicate()

    p.wait() 
    
    pty.shell("ffmpeg", "-framerate 25 -i " + inpath + os.sep + "*_%3d.png -y " + filename)
    
    