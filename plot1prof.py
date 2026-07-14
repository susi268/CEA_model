#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:19:02 2020

@author: sparenti

Reads a fits files with data and produce a radial cut at a wanted angle from the
equator. 
Theta: input angle in degree
mask : apply polemask
srad: save the radial line to be opliotted

"""
import numpy as np
import matplotlib.pyplot as plt
import SPUtils_gen as spugen
import SPUtils_WL as spuwl
import SPUtils as spu
import SPUtils_AIA as spuaia
import scipy.interpolate as interp
import matplotlib.cm as mplcm
import os

def plot1prof(file=None, theta=0, sv=None, wf=None, pxbin=None, pltr2=None, 
              sun=None, mask=None, srad=None):
 
    path = (r"/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/"
            r"CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/")
#    dirf1 = os.path.join(path, '/RES_DATA/RES_AIA193/')
#    dirf1 = os.path.join(path, (r'VICTOR_20191106_RHO1-2/VICTOR_087_RHO2/'
#                                r'AIA_20191106_087/AIA_87_20h_K-Cor20h/'))
#                                r'LASCO_20191106_087/LASCO_RHO2_87_20h/'))
#                                r'AIA_20191106_087/AIA_87_20h_LASCO15h/'))
#    dirf1 = os.path.join(path, (r'VICTOR_20191106_RHO1-2/VICTOR_035_RHO1/'
#                         r'LASCO_035/LASCO_RHO1_35_12h/'))
#                         r'/AIA_035/AIA_35_12h_LASCO15h/'))
#    dirf1 = os.path.join(path, 'VICTOR_WET3D/AIA_20H/SYN_WL/K-Cor/')
#    dirf1 = os.path.join(path, 'VICTOR_WET3D_RHO3/SIMU_156_12H/'
#                         r'LASCO_156_12h/')
#                        r'/AIA_193_12h_LASCO15h/AIA_12h_LASCO15h_PSF0/')
#    dirf1='/mnt/c/Users/Susanna Parenti/Documents/LAVORO/FITS/MLSO/'
#    dirf1 = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/FITS/LASCO/'
    dirf1 =  '/mnt/c/Data/SP_FITS/SDO/2018/'

    if file is None:
        file = '23731980pB.fts'
    image, h = spugen.open1img(dirf1, file)
    
 # Bin if needed (newshape=[nx,ny])
    if pxbin != None:
        newshape=(int(h["NAXIS1"]/pxbin), int(h["NAXIS2"]/pxbin))
        image, h = spugen.image_rebin(image, newshape, h=h)
#    rad = np.linspace(0., 1.414*(image.shape[0]/2)-1, num=image.shape[0])
    rad = np.linspace(0., 2.15*(image.shape[0]/2)-1, num=image.shape[0])
    x_ori, y_ori = np.indices(image.shape, sparse=True)
#    x = rad * np.cos(np.radians(theta)) + image.shape[1]/2.
#    y = rad * np.sin(np.radians(theta)) + image.shape[0]/2.
    x = rad * np.sin(np.radians(theta)) + (image.shape[1]/2.)
    y = rad * np.cos(np.radians(theta)) + (image.shape[0]/2.)

    int1 = interp.RegularGridInterpolator((y_ori.ravel(), x_ori.ravel()), 
                                          image, bounds_error=False, fill_value=0.0)
    image_cut = int1((y, x))

#    x_plt = x[0: int(h["NAXIS1"] / 2) - 1]
#    y_plt = y[0: int(h["NAXIS2"] / 2) - 1]
    
    dcut = 'Radial'
    px = int(theta)
    pxtext = 'angle ' + str(px)
    if "D" in h.keys():
        xx = rad * np.tan(h["CDELT1"]) * h["D"]
        x_plt = (x[0: int(h["NAXIS1"] / 2) - 1] - x[0])  * np.tan(h["CDELT1"]) * h["D"]
        y_plt = (y[0: int(h["NAXIS2"] / 2) - 1] - y[0])* np.tan(h["CDELT2"]) * h["D"]
    else:
        xx = rad * np.tan(h["CDELT1"])* h["radius"]
        x_plt = (x[0: int(h["NAXIS1"] / 2) - 1] - x[0]) * np.tan(h["CDELT1"]) * h["radius"]
        y_plt = (y[0: int(h["NAXIS2"] / 2) - 1] - y[0]) * np.tan(h["CDELT2"]) * h["radius"]

# Some definition for the plot 
    if (h["INSTRUME"].find('LASCO') != - 1) or (h["INSTRUME"].find('K-Cor') != - 1): 
     if ("WAVELNTH" in h.keys() and h["WAVELNTH"] == 'Ne') or ("FILENAME" in h.keys() and h["FILENAME"].find('Ne') != - 1):
#        if h["WAVELNTH"] == 'Ne':
         band = 'Ne'
         vmin = 4
         vmax = 6
         ylim_min = 2e3
         ylim_max = 2e8
         unit = 'cm^-3'
         label='Ne'
     else:
         band = 'pB'
         print('pB data')
         vmin = -1#-1
         vmax = 3 #2. #1
         ylim_min = 1 #5e-2  1 
         ylim_max = 1e4 #50 #1e4  #5e3
         label = 'pB'
     cmap=mplcm.plasma    
    else: 
        band = str(h["WAVELNTH"])
        print('UV data')
        vmin = -1#-1
        vmax = 3.5 #2. #1
        ylim_min = 1e-2 
        ylim_max = 1e3 
        label = 'AIA ' + band
        aia_obj = spu.aia_obj()
        cmap = aia_obj.filter_cmap_dict[band]

          
    if "MODELFL" in h.keys():
        if 'adapt' in h["MODELFL"]:             # MULTIVP
            data1 = 'f' + h["MODELFL"].split('.', -1)[0]
        else:                                   # PLUTO
            data1 = 'f' + h["MODELFL"].split('.', -1)[1]
        str1 = h["INSTRUME"] + "{}".format(h["DETECTOR"])
        if 'AIA' in h["INSTRUME"]:
            err = spuaia.get_aia_cerr(image_cut)
        else:
            print('Error non implemented yet')
            
    elif "FILENAME" in h.keys():                # LASCO data
        data1 = 'f' + h["FILENAME"].split('.', -1)[0]
        str1 = h["INSTRUME"] + "{}".format(h["DETECTOR"])
        err = spuwl.get_LASCO_pBerr(image_cut, xx)
    elif h["INSTRUME"] == 'AIA_2':
        data1 = 'f' + h['DATE-OBS'].split('T')[0]
        str1 = "AIA"
        err = spuaia.get_perr(image_cut, band)
    else:                                       # K-Cor data
        data1 = 'f' + h['DATE-OBS'].split('T')[0]
        str1 = 'K-Cor'
        err = spuwl.get_KCor_perr(image_cut)


#plot         
    
    gentit= (str1 + '_' + band + '_' + "LON{:.1f}_".format(np.degrees(h["LON"])) + dcut)
    titplot = gentit + " cut at {}".format(pxtext)
    titlegif1 = 'sing_prof_' + gentit + '{}'.format(str(px)) + "_" + data1 
    print(titlegif1)
#    titlegif1 = 'sing_fred_test2'
#    titplot = 'sing_fred_test2'
    good = image_cut > 1e-9
    fig, ax = plt.subplots(1,1, num=0, clear=True)
    fig.suptitle(titplot)
    fig2, ax2 = plt.subplots(1,1, num=1, clear=True)
    fig2.suptitle(titplot)
    if np.max(err) == 0:
        ax.plot(xx[good], image_cut[good], label=label )
    else:
        ax.errorbar(xx[good], image_cut[good], yerr=err[good], label=label)
    
    if pltr2 is True:
        label2 = '1/r^2 model'
        maxv = np.argmax(image_cut)
        r2 = 1/xx[good]**2 * image_cut[maxv] * xx[maxv]**2
        ax.plot(xx[good], r2, linestyle='dotted', label=label2)
     
    ax.set_yscale('log')
    ax.set_ylim(ylim_min, ylim_max)
#    ax.set_xlim(.9, 4.1)
    ax.set_xlabel("Solar radii")

    if "D_UNIT" in h.keys(): 
        D_UNIT = h["D_UNIT"]
    elif "UNIT" in h.keys():
        D_UNIT = h["UNIT"]
    else:
        D_UNIT = ""
    ax.set_ylabel(D_UNIT)
#    ax.set_ylabel('10^-8  Bsun')
    ax.legend(loc='best')

    rsun = spugen.get_rsun(h)
    ysun = rsun[:, int(h['NAXIS1']/2)]
    xsun = rsun[int(h['NAXIS2']/2), :]
    
    if mask is not None:
     m1, m2 = spugen.apply_poles_mask(h, mask)
     image[m1] = 0
     image[m2] = 0

     
    ax2.imshow(np.log10(image), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
               extent=[-xsun.max(), xsun.max(), -ysun.max(), ysun.max()])
    ax2.plot(x_plt, y_plt)
    ax2.set_xlabel("R$_\odot$")
    ax2.set_ylabel("R$_\odot$")
    
#  Plot the Sun circle
    if sun is True:
        rsun1 = plt.Circle((0,0), radius=1, color='white')
        ax2.add_artist(rsun1)

    if sv:
        fig.savefig(dirf1 +  titlegif1+ ".png")
        fig2.savefig(dirf1 +  'img_'+ titlegif1+ ".png")
        print("saved images", titlegif1+ ".png")
        print("saved images",'img_'+ titlegif1)

    data = np.column_stack((xx, image_cut, err))
#    print(image.shape)
    header = {"x_unit": 'R$_\odot$', "inst": h["INSTRUME"], 
                  "d_unit": D_UNIT, "data_f": titplot, "theta": theta, "lon" : h["LON"]}
    if wf is True:
#        titlegif1 = titlegif1
        spugen.wr_fits(dirf1, data, filename =titlegif1, h=header, overwrite=True)
    if srad is True:
        datar = np.column_stack((x_plt, y_plt))
        head = {"file": file, "theta": theta}
        filer = 'r_' + titlegif1
        spugen.wr_fits(dirf1, datar, filename=filer, h=head, overwrite=True)
    return