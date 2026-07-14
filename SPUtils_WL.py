#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:46:31 2020

@author: sparenti
"""
import SPUtils as spu
import SPUtils_gen as spugen
import SPUtils_AIA as spuaia
# import pyPLUTO as pp
import numpy as np
import tomograpy
import fitsarray as fa
from scipy import ndimage
import matplotlib.cm as mplcm

def image_header_from_WL(dirf, lfile, pxbin, source, dtype=np.float64):
    """
    Generate a cubic map header from an WL (LASCO) header for the simulation

    Arguments
    ---------
    dirf, lfile : location and LASCO filename
    pxbin : bin applied to the LASCO image
    source : 'PLUTO', 'MULTIVP'
 
 SOHO LON = 3.758 rad, LAT = 0.065 rad,  D = 1.46619 10^8 km, 210.66 R_sun    (6/11/2018)
 C2 FOV: 12.4 Rsun
 
    Output
    ------
    header: pyfits header
    """
    
# gets WL info
    dataf, h = spugen.rd_fitsf(dirf, lfile)
    print('Reading file....', lfile)
    binwl = 1024./h['NAXIS1']                       # binning of the WL data image
    print("image_header_from_WL: binwl: {}".format(binwl) )

# generate header
    header = dict()
#    header["PLUTOFL"] = ''
    header["MODELFL"] = ''
    header['PIXBIN'] = pxbin                        # Binning applied to WL
    if 'DATE_OBS' in h.keys():
        header['DATE_OBS'] = h["DATE_OBS"]
    else:
        header['DATE_OBS'] = ''
    header['SIMPLE'] = True
    header['BITPIX'] = fa.bitpix_inv[dtype.__name__]
    header['NAXIS'] = h['NAXIS']
    header['INSTRUME'] = source + '-' + h["INSTRUME"] #'PLUTO-' + h["INSTRUME"]
    if 'DETECTOR' in h.keys():
        header['DETECTOR'] = h["DETECTOR"]
    else:
        header['DETECTOR'] = ""
    if 'EXPTIME' in h.keys():
        header['EXPTIME'] = h['EXPTIME']
    else:
        header['EXPTIME'] = " "
    if "CRLT_OBS" in h.keys():
        header['LAT'] = np.radians(h["CRLT_OBS"])      # Assumes only 1 LAT
    else:
        header['LAT'] = 0.
    if "CRLN_OBS" in h.keys():
        header['LON'] = np.radians(h["CRLN_OBS"])      # Assumes only 1 LAT
    else:
        header['LON'] = 0.
    if 'WAVELNTH' in h.keys():
        header['WAVELNTH'] = h["WAVELNTH"]
    else:
        header['WAVELNTH'] = ""
    if "RSUN_OBS" in h.keys():
        header["RSUN_OBS"] = h["RSUN_OBS"]           # in arcsec
    else:
        header["RSUN_OBS"] = ""    
    for i in range(2):
            header['NAXIS' + str(i + 1)] = int(round(h['NAXIS' + str(i + 1)])/float(pxbin))
    header["radius"] = 210.6 #h['R_SOHO'] # for LASCO   #212.45     # In R_sun 210.6
    if h["INSTRUME"].find('LASCO') != -1:
         header = pop_PLASCO_head(header, h, pxbin, binwl)
 #   elif h["INSTRUME"].find('EUI') != -1:
    elif h["INSTRUME"].find('Metis') != -1:
         header = pop_METIS_head(header, h, pxbin)     
    else:
        for i in range(2):
            header['CDELT'+ str(i + 1)] = np.radians(h['CDELT'+ str(i + 1)]* binwl/3600) * pxbin         # radians
            header["CRVAL" + str(i + 1)] =   np.radians(h["CRVAL" + str(i + 1)]/3600)/ float(pxbin)
#            header['CRPIX' + str(i + 1)] = h['CRPIX' + str(i + 1)]/ float(pxbin)
            header['CRPIX' + str(i + 1)] = (h['CRPIX' + str(i + 1)] -1) /float(pxbin) + 1

        if h["INSTRUME"].find('K-Cor') != -1:
            header = pop_PLKCor_head(header, h)
#    header["dmap"] = mplcm.plasma            
    tomograpy.siddon.map_borders(header)
    return header


#   Populate the MLSO MODEL header (header) with the data header (hdata)
#   FOV 1.05 to 3 solar radii
def pop_PLKCor_head(header, hdata):
    this_str = header['INSTRUME'].split('-', 1)
    header['INSTRUME'] =  this_str[0] + '-K-Cor'    #"PLUTO-K-Cor"
    header['DATE_OBS'] = hdata["DATE-OBS"]
    header['D_UNIT'] = hdata["BUNIT"]
    header['LON'] = np.radians(hdata['CRLN_OBS'])   #- np.pi
    print('=========================================================================')
    print('pop_PLKCor_head:')
    print('LON: ' + str(header['LON']) + ' LAT: ' + str(header["LAT"]))
    print('For the day', header['DATE_OBS'])
    print('=========================================================================')
    return header


def pop_PLASCO_head(header, hdata, pxbin, binwl):

    """Populate the LASCO MODEL header (header) with the data header (hdata)
    LON 3.758 (215.3) @ 12hh
    LON 3.81 (218.3)  @ 15h   in rad
    """

    header['LAT'] =  -4.8                     # Assumes only 1 LAT
    header['LON'] = 67.7   #2.9 22/2  #3.81 LASCO 6/11/18 @15h
    header['WAVELNTH'] = hdata["FILTER"]
    header['D_UNIT'] = hdata["UNIT"]
    header['radius'] = hdata["R_SOHO"]
    print('pop_PLASCO_head: model I, ',header['D_UNIT'])
    for i in range(2):
        if hdata["DETECTOR"] == 'C2':
            header['CDELT'+ str(i + 1)] = np.radians(11.9 * binwl * 1./3600.) * pxbin # radians
            header["CRVAL" + str(i + 1)] = 0. / float(pxbin)
            header['CRPIX' + str(i + 1)] = header['NAXIS' + str(i + 1)] / (2.)
 #           header['CRPIX1'] = 511.2/ (2. * pxbin)         # CRPIX from LASCO WL and 2 from binning in the pB images.
 #           header['CRPIX2'] = 507.5 / (2. * pxbin)
        else:                               # C3
            header['CDELT'+ str(i + 1)] = np.radians(56. * 1./3600.)* float(pxbin) # radians  
    print('=========================================================================')
    print('pop_PLASCO_head:')
    print('ATTENTION: for LASCO the LAT and LON are set manually in pop_PLASCO_head')
    print('LON: ' + str(header['LON']) + ' LAT: ' + str(header["LAT"]))
    print('Updated the solar distance radius', str(header["radius"]))
#    print('For the day 2021-02-15')
    print('For the day', header['DATE_OBS'])
    print('=========================================================================')
    header["RSUN_OBS"] = np.degrees(np.arctan2(6.96349e5, 1.46619e8)) * 3600.   # in arcsec
    return header


#   Populate the METIS MODEL header (header) with the data header (hdata)
def pop_METIS_head(header, hdata, pxbin):
#    binwl = 2048./header['NAXIS1']
    binwl = hdata['NBIN1']                # Assumes same binning in the two direction
    this_str = header['INSTRUME'].split('-', 1)
    header['INSTRUME'] =  this_str[0] + '-METIS'
    header['DETECTOR'] = 'WL'
#    header['LAT'] =  np.radians(hdata["CRLT_OBS"])                     # Assumes only 1 LAT
#    header['LON'] =   np.radians(hdata["CRLN_OBS"])                    #0 #np.pi
    for i in range(2):
        header['CDELT' + str(i + 1)] = np.radians(hdata["CDELT1"] / 3600.) * pxbin  # radians
        header['CRPIX' + str(i + 1)] = hdata['CRPIX' + str(i + 1)] / float(pxbin) + 1     # center of the image which not
#   header["CRVAL1"] = np.radians(hdata["CRVAL1"] / 3600.) / float(pxbin)
#    header["CRVAL2"] = np.radians(hdata["CRVAL2"] / 3600.) / float(pxbin)
#    header['CRPIX1'] =  hdata['SUN_XCEN']         # Because the image of the Sun is not centered
#    header['CRPIX2'] =  hdata['SUN_YCEN']
    header["CRVAL1"] = np.radians(-(hdata['SUN_XCEN']- hdata['CRPIX1']) * hdata["CDELT1"] / 3600.) / float(pxbin)
    header["CRVAL2"] = np.radians(-(hdata['SUN_YCEN'] - hdata['CRPIX2']) * hdata["CDELT2"] / 3600.)/ float(pxbin)

    header["radius"] = hdata["DSUN_OBS"] / hdata["RSUN_REF"]    # distance in Rsun
    header["D_UNIT"] = hdata["BUNIT"]    # "Dn/s"
    header["DATE_OBS"] = hdata["DATE-OBS"]
    header["INN_FOV"] = hdata["INN_FOV"]       #deg
    header["OUT_FOV"] = hdata["OUT_FOV"]       #deg
    header["IO_XCEN"] = hdata["IO_XCEN"]       # center of the internal Occ. in pixels
    header["IO_YCEN"] = hdata["IO_YCEN"]
    header["FSPIX1"] = hdata["FSPIX1"]        # center of the external Occ. in pixels
    header["FSPIX2"] = hdata["FSPIX2"]
    header["SUN_XCEN"] = hdata["SUN_XCEN"]
    header["SUN_YCEN"] = hdata["SUN_YCEN"]
    header["UNIT_CONV"] = hdata['HISTORY'][11]
    print('=========================================================================')
    print('POP_METIS_HEAD:')
    print('=========================================================================')
#    header["colmap"] = mplcm.Greens_r
    return header
    
    
def mk_LASCO_rot_wrhead(data, h):
    
    """
    Add missing keywords in the pB LASCO DATA image, add roll
    Rotate the pB image of ROLLANGL.
    Note Positive values mean counter-clockwise rotation (cv2)
    Note ndimage.rotate rotate Positive values counter-clockwise rotation 
    Change the Units for pB from 10^-10 Bsuns(DN/pixels.sec) to linear
    Apply a mask at R = 2.2
    Note: the R_SOHO is wrong, so added keyword 'radius'
    LON = 3.758 @12h
    LON = 3.81  @15h
    LON = 3.52  @21h
 
    """
    binwl = 1024./h['NAXIS1']
#    h["radius"] = 210.6
    h['radius'] = h["R_SOHO"]
    if h["FILENAME"].find('Ne') != - 1:
        h["D_UNIT"] = "cm^-3"
        print("mk_LASCO_rot_head: the  UNIT keyword has been corrected to", h["D_UNIT"])
    else:
        h["D_UNIT"] = h.pop(("UNIT"))
    print(h["FILENAME"])
    if h["FILENAME"] == '23731980pB.fts':
            h['LAT'] =  0.065  # 0.03  #@ 7/11   #  0.065  @ 6/11/18 
            h['LON'] =  3.81   #3.52  #7/11 #3.81    @15h    3.758 @12h
    elif h["FILENAME"] == '23732114pB.fts':
            h['LAT'] =  0.03  #@ 7/11   #  0.065  @ 6/11/18 
            h['LON'] =  3.52  #7/11 #3.81    @15h    3.758 @12h
    elif (h["FILENAME"] == '23833796pB.fts') :     # 15/2/21 @ 09:
        h['LAT'] = -0.12
        h['LON'] = 0.69
    elif (h["FILENAME"] == '23834679pB.fts'):
        h['LAT'] = -0.12
        h['LON'] = 5.3
    else:
        raise ValueError("The input file is not known")
        
    for i in range(2):
        if h["DETECTOR"] == 'C2':
            h['CDELT'+ str(i + 1)] = np.radians(11.9 * 1./3600.) * binwl  # radians
            h["CRVAL" + str(i + 1)] = 0.
            h['CRPIX' + str(i + 1)] = h['NAXIS' + str(i + 1)] / (2.)
        else : # C3
            h['CDELT'+ str(i + 1)] = np.radians(56. * 1./3600.) # radians
    rot_im = ndimage.rotate(data,- h["ROLLANGL"] , reshape=False)
    wl_mask = spuaia.apply_aia_mask(h, 2.2)
    rot_im[wl_mask] = 1e-10 
    print("===============================================")
    print('mk_LASCO_rot_wrhead:')
    print("DETECTOR: {}".format(h["DETECTOR"]) ) 
    print("New keywords added to the data header: CDELT, CRVAL, CRPIX, radius")
    print("New keywords added to the data header by hand: LON {:4.1f}, LAT {:4.1f}"\
          .format(np.degrees(h["LON"]), np.degrees(h["LAT"]))) 
    print("Image rotated of ROLLANGL: {}".format(h["ROLLANGL"]))
    print("===============================================")
    return rot_im, h 
 
    
    
def get_LASCO_pBerr(image, rad):
    """ 
    Get the error on the pB. 
    rad = distance in Rsun of the same shape as image
    """
    if rad.shape != image.shape:
        raise Exception('rad and image should have the same shape')
#    pberr_image = np.ones(image.shape)
    pberr_image = image * 0.15
#    pberr_image[rad < 2.9] = image[rad < 2.9] * 0.3
#    pberr_image[rad > 2.9] = image[rad >= 2.9] * 0.09
    return pberr_image
    


# Prepare the K-Cor data for a comparison with the model
def mk_KCor_prep(data, h, mask=None):
     data *= 1e10                # to be consistent with PLUTO
     h['INSTRUME'] = "K-Cor"
     for i in range(2):
         h['CDELT'+ str(i + 1)] = np.radians(h['CDELT'+ str(i + 1)] * 1./3600.)
#     h['UNIT'] = '1e-10 ' + h['BUNIT']
     h['D_UNIT'] = '1e-10 ' + h['BUNIT']    
     h['radius'] = 1./np.tan(np.radians(h["RSUN_OBS"]/3600))
     h['LON'] = np.radians(h["CRLN_OBS"]) 
     if mask == True:
         wl_mask = spuaia.apply_aia_mask(h, 1.1)
         data[wl_mask] = 1e-10
     print("===============================================")
     print('mk_KCor_prep:')
     print('Renamed INSTRUME to:', h['INSTRUME'])
     print('Data now in unit of ', h['D_UNIT'])
     print('New keyword radius  has been added:', h['radius'])
     print('New keyword LON has been added (equal to CRLN_OBS):',  h['LON'])
     print('CDELT converted in radians')
#     print('Mask at 1.1 Rsun applied to the K-Cor data')
     print("===============================================")
     return data, h


# Ger the background noise of the K-Cor data.
def get_KCor_perr(image):
    perr_image = np.ones(image.shape)
    perr_image[:] = 3e-9 *1e10            # in units of 1e-10 B/B0
    print('Error provided in 1e-10 B/Bo')
    print('=================================================')
    return perr_image


def WL_centered_stack(pshape, shape, pxbin, n_images=1., radius=1.,
                   min_lon=0., max_lon=np.pi, fill=0., dtype=np.float64):
    """
    Generate a stack with centered image and circular trajectory data.
    """
    from tomograpy.simu import circular_trajectory_data
    header = image_header_from_WL(pshape, shape, pxbin, dtype=dtype)
    header.update({'n_images': n_images})
    header.update({'radius': radius})
    header.update({'min_lon': min_lon})
    header.update({'max_lon': max_lon})
    data = circular_trajectory_data(**header)
    data[:] = fill
    return data.astype(dtype)


def WL_centred_im_header(pshape, shape, fill=0., dtype=np.float64):
    """
    Generate a centered cubic map header

    Arguments
    ---------
    pshape : physical shape
    shape : shape in pixels

    Output
    ------
    cube: 3d FitsArray
    """
    header = image_header_from_WL(pshape, shape, dtype=dtype)
    # generate cube and exit
    map = fa.fitsarray_from_header(header)
    map = fa.InfoArray(data=map, header=dict(header))
    map[:] = fill
    return map

def apply_METIS_mask(image_header):
    """
    It uses "IO_XCEN" to fix the center of the Internal Occ,
    # as we put the Sun in the center      of the image (CRPIX = 'SUN_XCEN', see 'pop_METIS_head') -> NO
    :param image_header: 
    :return: Metis mask
    """

    #mask_start = image_header["radius"] * np.radians(image_header["INN_FOV"])
    mask_start = np.radians(image_header["INN_FOV"])
    print("Calculated the METIS inner mask:", mask_start)

    y, x = np.ogrid[:image_header["NAXIS2"], :image_header["NAXIS1"]]
    dycen = image_header["NAXIS2"] / 2 + image_header["IO_YCEN"] - image_header["SUN_YCEN"]
    dxcen = image_header["NAXIS1"] / 2 + image_header["IO_XCEN"] - image_header["SUN_XCEN"]

    y = (y - dycen) * image_header["CDELT2"] #+ image_header["CRVAL2"]
    x = (x - dxcen) * image_header["CDELT1"] #+ image_header["CRVAL1"]
#    xx = (x - image_header["IO_XCEN"] + 1) * image_header["CDELT1"]    #rad
#    yy = (y - image_header["IO_YCEN"] + 1) * image_header["CDELT2"]
#    r_x = image_header["radius"] * np.tan(x)
#    r_y = image_header["radius"] * np.tan(y)
    r =  np.sqrt(x** 2 + y** 2)

    metis_mask = r < mask_start
    print("//////////////////////////////////////////")
    print("Dark mask applied for R {}".format(mask_start))
    print("//////////////////////////////////////////")
    return metis_mask