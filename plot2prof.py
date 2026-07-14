#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:26:44 2020
 
Plot the vertical or horizontal or radial profiles from 2 images of the same size
long and lat are in pixels. Be careful, the imshow plots the Y0 north.   
    aiai: image2 is a AIA image and it will be fliped.
    tit1 & 2: for the title in the images
    image1: PLUTO
    image2: AIA - WL data
    theta  for the radial cut. 0 deg at the equator

@author: sparenti
"""


from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import SPUtils as spu
import SPUtils_AIA as spuaia
import SPUtils_gen as spugen
import scipy.interpolate as interp
import matplotlib.cm as mplcm
import os


def plot2prof(image1=None, image2=None, h1=None, long=None, lat=None, sv=None, theta=None,
              tit1=None, tit2=None, shift=None, er=None, wf=None, pltr2 = None):
    
    path = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/'
    dirf = "/mnt/c/UsersSusanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/"
#    dirf1 = os.path.join(path, 'VICTOR_WET3D_RHO3/SIMU_156_12H/AIA_193_12h_LASCO15h/AIA_12h_LASCO15h_PSF0/')
#    dirf1 = os.path.join(path, 'VICTOR_WET3D/AIA_20H/SYN_WL/K-Cor/')
#    dirf1 = os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_035_RHO1/'
#                            r'AIA_035/AIA_35_12h_K-Cor20h/')
    dirf1 = os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_087_RHO2/'
#                         r'LASCO_20191106_087/LASCO_RHO2_87_20h/')
                        r'AIA_20191106_087/AIA_87_20h_LASCO15h/')
#    dirf2 = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/FITS/LASCO/'
    dirf2 = '/mnt/c/Data/SP_FITS/SDO/2018/'
    if image1 is None:
        file1 = 'PLUTO-AIA_2_NAX128_PSHAPE7.0_VOX810_PSF0_LON218.2_f87.fits'
        print("Opening file {}".format(file1))
    if image2 is None:
        file2 = 'aia.lev1.193A_2018-11-06T15 00 52.84Z.image_lev1.fits'
        print("Opening file {}".format(file2))
        image1, h1, image2, h2 = spugen.open2imgs(dirf1, file1, dirf2, file2)
        
#    data1 = 'f' + h1["PLUTOFL"].split('.', -1)[1]
    data1 = 'f' + h1["MODELFL"].split('.', -1)[1]

    if image1.shape != image2.shape:
        raise Exception("The two images have different size")

    if shift is not None:
        image1=np.roll(image1, shift, axis=(0, 1))

    rsun = spugen.get_rsun(h1)

    if lat is not None:
        dcut = 'Hori'
        px = int(lat)
        pxtext ='pixel ' + str(px)
        image1_cut = np.squeeze(image1[lat, :])
        image2_cut = np.squeeze(image2[lat, :])
        xx = np.squeeze(rsun[int(h1["CRPIX1"]), :])
#        xx = np.squeeze(rsun[px, :])
        xx[0:int(h1["CRPIX1"])] = - xx[0:int(h1["CRPIX1"])]
        x_plt = np.squeeze(np.indices(image1_cut.shape))
        y_plt = np.zeros(image1_cut.shape) + lat
    elif long is not None:
        dcut = 'Vert'
        px = int(long)
        pxtext='pixel ' + str(px)
        image1_cut = np.squeeze(image1[:, long])
        image2_cut = np.squeeze(image2[:, long])
        xx = np.squeeze(rsun[:, int(h1["CRPIX2"])])
#        xx = np.squeeze(rsun[:,px])
        xx[0:int(h1["CRPIX2"])] = - xx[0:int(h1["CRPIX2"])]
        y_plt = np.squeeze(np.indices(image1_cut.shape))
        x_plt = np.zeros(image1_cut.shape) + long
    elif theta is not None:
#        rad = np.linspace(0., 1.414*(image1.shape[0]/2)-1, num=image1.shape[0])
        rad = np.linspace(0., 2.1*(image1.shape[0]/2)-1, num=image1.shape[0])
        x_ori, y_ori = np.indices(image1.shape, sparse=True)
#        x = rad * np.cos(np.radians(theta)) + image1.shape[1]/2.
#        y = rad * np.sin(np.radians(theta)) + image1.shape[0]/2.
        x = rad * np.sin(np.radians(theta)) + image1.shape[1]/2.
        y = rad * np.cos(np.radians(theta)) + image1.shape[0]/2.
        int1 = interp.RegularGridInterpolator((y_ori.ravel(), x_ori.ravel()), image1,
                                              bounds_error=False, fill_value=0.0)
        int2 = interp.RegularGridInterpolator((y_ori.ravel(), x_ori.ravel()), image2,
                                              bounds_error=False, fill_value=0.0)
                                                      
        image1_cut = int1((y, x))
        image2_cut = int2((y, x)) 
        x_plt = x[0: int(h1["NAXIS1"]/2) -1]
        y_plt = y[0: int(h1["NAXIS2"]/2) -1]
        dcut = 'Radial'
        px =  int(theta)
        pxtext = 'Angle ' + str(px)
        if "D" in h1.keys(): 
            xx = rad * np.tan(h1["CDELT1"]) * h1["D"]
        else:
            xx = rad * np.tan(h1["CDELT1"]) * h1["RADIUS"]
        print("x_max", np.max(xx))    
    else:
        raise Exception("you have to set long or lat keywords")

#    ratio = np.ones_like(image1_cut)
    ratio = np.zeros_like(image1_cut)    
#    good = ((image2_cut != 0) & (image1_cut != 0))  
    good = (image1_cut > 1e-9)
    ratio[good] = image2_cut[good]/image1_cut[good]
    print("x_max", np.max(xx[good]))  

    lon = np.degrees(h1["LON"])
    if (h1["INSTRUME"].find('LASCO') != - 1) or (h1["INSTRUME"].find('K-Cor') != - 1): 
        print('LASCO or K-Cor data')
#        lon = np.degrees(h1["LON"])
        if h1["WAVELNTH"] == 'Ne':
            band = 'Ne'
            vmin = 4
            vmax = 6
            ylim_min = 2e3
            ylim_max = 2e5
        else:
            band = 'pB'
            vmin = -1 #-1 #-3
            vmax =  2.5 #1
            ylim_min = 1e-1 #1e-2 # 5e-4
            ylim_max = 5e3 #50
        phot_err = np.zeros_like(image1_cut[good])
        gentit= h1["INSTRUME"] + "{}".format(h1["DETECTOR"]) + '_' +\
                band + "_LON{0:.1f}_".format(lon)+ dcut
#        gentit = tit1
#        titplot = gentit + " cut at {}".format(pxtext)
#        titlegif1 = 'prof_' + gentit + '{}'.format(str(px)) + "_" + data1 
#        titlegif2 = 'imgs_' + gentit + '{}'.format(str(px)) + "_" + data1 + ".png"
        cmap=mplcm.plasma

    elif (h1["INSTRUME"].find('AIA') != - 1 or h1["INSTRUME"].find('EUI') != -1):
        print("{} data".format(h1["INSTRUME"]))
        band = str(h1["WAVELNTH"])
#        lon = np.degrees(h1["LON"]) #- np.pi)
        if h1["INSTRUME"].find('AIA') != - 1:
            h1["INSTRUME"] = 'AIA'
            phot_err = spuaia.get_perr(image2_cut[good], band)
            print('Calculate the noise error for image2 (AIA)')
        gentit = h1["INSTRUME"] + "{}".format(h1["WAVELNTH"]) + "_LON{0:.1f}_".format(lon) + dcut
#        titplot = "AIA {} ".format(h1["WAVELNTH"]) + "_LON{0:.1f}_".format(lon)
#        titlegif = 'AIA{}_20h_'.format(h1["WAVELNTH"]) + 'LON{0:.1f}_'.format(lon) 
#        + dcut + " cut at {}".format(pxtext)
        aia_obj = spu.aia_obj()
        cmap = aia_obj.filter_cmap_dict[band]
        vmin = -.1
        vmax = 3.5
        ylim_min = 1e-1
        ylim_max = 1.5e3
# get the errors
        if er != None:
            print("==================================================")    
            print('Calculate calibration error for image1 (PLUTO)')
            cerr = spuaia.get_aia_cerr(image1_cut[good])
    else:
        print('TBD')
    print("==================================================")

    
#    gentit = (file1.split(".")[0] + '_' + 
#             "LON{:.1f}_".format(np.degrees(h1["LON"])) + dcut)
    gentit3 = (file2.split(".")[0] + '_' + 
              "LON{:.1f}_".format(np.degrees(h1["LON"])) + dcut)
    titplot = gentit +  " cut at {}".format(pxtext)
    titlegif1 = 'prof_' + gentit + '{}'.format(str(px)) + "_" + data1 
    titlegif3 = 'prof_' + gentit3 + '{}'.format(str(px)) + "_" + data1 
    
    titlegif2 = 'imgs_' + gentit + '{}'.format(str(px)) + "_" + data1 + ".png"   
    print(titplot, titlegif1, titlegif3)

# plot
    fig, ax = plt.subplots(2,1, num=0, clear=True)
    ax[0].set_title(titplot)
#    if h1["INSTRUME"].find('LASCO') != - 1 or (h1["INSTRUME"].find('K-Cor') != - 1):
    if er is None:
        ax[0].plot(xx[good], image1_cut[good], label=tit1)
        ax[0].plot(xx[good], image2_cut[good], label=tit2)
#        ax[0].set_yscale('log')
#        ax[0].set_ylim(ylim_min, ylim_max)
    else:
        ax[0].errorbar(xx[good], image1_cut[good], yerr=cerr, label=tit1)
        ax[0].errorbar(xx[good], image2_cut[good], yerr=phot_err, label=tit2)

    if pltr2 is True:
        label2 = '1/r^2 model'
        maxv = np.argmax(image1_cut)
        r2 = 1/xx[good]**2 * image1_cut[maxv] * xx[maxv]**2
        ax[0].plot(xx[good], r2, linestyle='dotted', label=label2)

    xlim_min = .8
    xlim_max = 1.1
    ax[0].legend(loc='best')
    ax[0].set_ylabel(h1["D_UNIT"])
#    ax[0].set_xlim(xlim_min, xlim_max)
    ax[0].set_yscale('log')
    ax[1].set_xlabel('R$_\odot$')
    ax[1].set_ylabel("DATA/MODEL")
    ax[1].plot(xx[good], ratio[good], color="g")
#    ax[1].set_ylim(0.1, 5)
#    ax[1].set_xlim(xlim_min, xlim_max)
    if tit1 and tit2:
        ax[1].set_ylabel(tit2 + "/" + tit1)
    if sv:
#        fig.savefig(dirf + 'AIA{}_20h_'.format(h1["WAVELNTH"]) + 
#                    'LON{0:.1f}_'.format(lon) 
        fig.savefig(dirf1 + titlegif1+ ".png", bbox_inches="tight")
        print("Image saved")
 
    fig1, ax1 = plt.subplots(1, 2, num=1, clear=True)
    ax1[0].set_title(tit1)
    ax1[1].set_title(tit2)
    ax1[0].imshow(np.log10(image1), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax1[0].plot(x_plt, y_plt)
    ax1[1].imshow(np.log10(image2), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax1[1].plot(x_plt, y_plt)
    
    if sv:
        print(dirf1)
        fig1.savefig(dirf1 +  titlegif2)
 
#        else:
#            fig1.savefig(dirf + 'AIAS{}'.format(h1["WAVELNTH"])+ 
#                         '_LON{0:.1f}'.format(lon) + ".png", bbox_inches="tight")
#               fig1.savefig(dirf + 'imgs_' + titlegif + '{}'.format(str(px))+ ".png", bbox_inches="tight")
        print("Image saved")
        
    if wf is True:
            header = {"x_unit": 'R$_\odot$', "inst": h1["INSTRUME"], 
                  "d_unit": h1["D_UNIT"], "data_f": titplot, "theta": theta}
            data3 = np.column_stack((xx[good],image1_cut[good]))
            header2 = {"x_unit": 'R$_\odot$', "inst": h2["INSTRUME"], 
                  "d_unit": h2["D_UNIT"], "data_f": titplot, "theta": theta}
            data4 = np.column_stack((xx[good], image2_cut[good]))
            spugen.wr_fits(dirf1, data3, filename=titlegif3, h=header, overwrite=True)
            spugen.wr_fits(dirf1, data4, filename=titlegif2, h=header2, overwrite=True)
            #print("Written file...", titlegif1)
            
    return
