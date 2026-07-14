#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:54:59 2020

@author: sparenti
"""
import numpy as np
import matplotlib.pyplot as plt
import SPUtils_gen as spugen
import SPUtils_AIA as spuaia
import SPUtils as spu
import scipy.interpolate as interp
import matplotlib.cm as mplcm
import os


def get_lat_slice(r, image=None, file=None, h=None, pxbin=None, plt_max=None,
                  nocol= None, sv=None, r_mask=None, wf=None):
    """
    get the data slice the closest to r (no interpolation for the moment)
    r = [r1, r2..]  in solar radii
    plt_max: plot the position of the max of intensity. 
    plt_max = [x_min, x_max] in degrees
    wf = True wite file to be used by 'profiles' class in Profiles_class
    r_mask: put a white disc for R1
    nocol = one color for the cuts
    ex : cuts, x = get_lat_slice(r, file=file, plt_max=[90, 150])
    """
# Read the data
#    path = (r'/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/'
#            r'VSWMC_DATA/SIMU_UV/IMG_DIR/')
    path = '/home/sparenti/WORK/GLOBAL_MODEL/PERI2_OUT/'
#    path = 'C://Users//sparenti//Documents//LAVORO//GLOBAL_MODEL//CEA_SOLO_SIMU//VICTOR_20210521_SOLO_PER1 - 2//PER2_SIMU//TEST_CODE//'
    dirf1 = path
#    dirf1 = '/home/sparenti/FITS/MLSO/'
#    dirw = os.path.join(path, 'LAT_PROF/LASCO_LATPROF/LON218/')
#    dirw = os.path.join(path, 'K-COR_LATPROF/')
#    dirw = os.path.join(path, 'LAT_PROF/KCor_LATPROF/')
#    dirw = '/home/sparenti/WORK/GLOBAL_MODEL/TEST_CODE/'
    dirw = os.path.join(path, 'LAT_PROF/')
#    dirw=path

    if file is None and image is None:
        raise ValueError("either file or image should be passed")
    if file is not None:
        image, h = spugen.open1img(dirf1, file)
    else:        
        if image is not None and h is None:
            raise ValueError("Both image ang h should be passed")
        image, h = spuaia.get_aia_prep(image,h)
        print('AIA data prep')
# Bin if needed
    if pxbin is not None:
        image, h = spugen.image_rebin(image, (pxbin, pxbin), h=h)

# Prepare for the plot
    if nocol is True:
        colors =['b', 'b', 'b', 'b', 'b', 'b']
    else:    
        colors = ['g', 'c', 'r', 'y', 'm', 'b', 'limegreen', 'goldenrod']
    fig, ax = plt.subplots(1, 1, num=0, clear=True)
    if (h["INSTRUME"].find('LASCO') != - 1):
        vmin = -1#-1
        vmax = 2. #1
        ylim_min = 5e-2 # 1 
        ylim_max = 100 #50 #1e4  #5e3
        cmap = mplcm.plasma
    elif (h["INSTRUME"].find('K-Cor') != - 1): 
        vmin = -1#-1
        vmax = 3 #2. #1
        ylim_min = 1 #5e-2  1 
        ylim_max = 3e3 #50 #1e4  #5e3
        cmap = mplcm.plasma
    elif  (h["INSTRUME"].find('AIA') != - 1):
        vmin = -1
        vmax = 3 
        ylim_min = .1
        ylim_max = 4e2
        aia_obj = spu.aia_obj()
        cmap = aia_obj.filter_cmap_dict[str(h['WAVELNTH'])]
#        data_mask = spuaia.apply_aia_mask(h, 1.5)
#        image[~data_mask] = 0.


    print(h["INSTRUME"])    
    ax.set_xlabel('Position Angle (deg)')
    ax.set_ylabel(h["D_UNIT"]) 
#    ax.set_title("Latitudinal profile at $R_\odot$ = {}".format(r)) #+ ("{0:5.1f}"* len(r)).format(*r))
#    ax.set_ylim(0.07, 50)
    ax.set_ylim(ylim_min, ylim_max)
    fig2, ax2 = plt.subplots(1, 1, num=1, clear=True)
    ax2.set_xlabel('Rsun')
    ax2.set_ylabel('Rsun')
    if "MODELFL" in h.keys():
        tit2 = h["MODELFL"]
    elif "FILENAME" in h.keys():
        tit2 = h["FILENAME"]
    else:
         tit2 = h["DATE-OBS"]
    titplot = h['INSTRUME'] + ' LON{:.1f} '.format(np.degrees(h["LON"])) + tit2
    ax2.set_title(h['INSTRUME'] + ' ' + tit2 + 
                  "\n LON = {:.1f}".format(np.degrees(h['LON'])))
    ax.set_title("Latitudinal profile for {}".format(titplot) ) #+  
#                 "\n $R_\odot$ = {}".format(r)) #, np.degrees(h['LON'])))

# Prepare the cut
    if "RADIUS" in h.keys():
        radius = h["RADIUS"]
    else:
        radius = h["D"]
    x, y = np.indices(image.shape, sparse=True)
    x = np.tan((x - h["CRPIX1"]) * h["CDELT1"] + h["CRVAL1"]) * radius
    y = np.tan((y - h["CRPIX1"]) * h["CDELT1"] + h["CRVAL1"]) * radius
    pa = (np.linspace(0., 2*np.pi, num=image.shape[0])) #- np.pi/2)
    cuts = np.zeros((image.shape[0], len(r)))
    inter = interp.RegularGridInterpolator((y.ravel(), x.ravel()), image,
                                           bounds_error=False, fill_value=0.0)
    r_str = 'R'
    max_I = np.zeros(len(r))        # Values of the max of I at each R
    max_I_pa = np.zeros(len(r))     # Index of the max I
    for i in range(len(r)):
        r_str = r_str + str(r[i]) + '_'
#        x_r = r[i] * np.cos(pa)
#        y_r = r[i] * np.sin(pa)
        x_r = r[i] * np.sin(pa)
        y_r = r[i] * np.cos(pa)
        new_int = inter((y_r, x_r))
       
# Marks the maximumn within plt_max
        if plt_max is not None:
            if len(plt_max) != 2:
                raise ValueError("plt_max = [x_min, x_max] in radians")
            good = np.logical_and(np.degrees(pa) >= plt_max[0], 
                                      np.degrees(pa) <= plt_max[1])
            datamax = new_int[good].max()
            indexmax = np.argmax(new_int[good])
            ax.plot([np.degrees(pa[good][indexmax]),
                    np.degrees(pa[good][indexmax])],
                    [0, datamax], linestyle='dotted', color = colors[i])
            print("***** PA_m = {0:3.1f}:".format(np.degrees(pa[good][indexmax])), "******")
            label_max = "  PA$_m$ = {0:5.1f}$^\circ$".format(np.degrees(pa[good][indexmax]))
        else:
            label_max =""
            datamax = 0.
            indexmax = 0
            good =  np.arange(len(pa))

# Plot
        ax.plot(np.degrees(pa), new_int, label="R$_\odot$= {}".format(r[i]) + label_max,
                color=colors[i])
        ax2.plot(x_r, y_r, color=colors[i])

        max_I[i] = datamax
        max_I_pa[i] = pa[good][indexmax]
        cuts[:, i] = new_int
        del new_int

    ax.legend(loc=1)
    fig.show()

    if len(r) != 1:
        ax.set_yscale('log')

    ax2.imshow(np.log10(image), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
               extent=[-x.max(), x.max(), -y.max(), y.max()])
    if r_mask == True:
        rsun1 = plt.Circle((0, 0), radius=1, color='white')
        ax2.add_artist(rsun1)
    ax2.text(x_r[0] +.1, y_r[0], 'PA =' + str(np.degrees(pa[0])), color="black")

# Save images and data
    if (sv is True) or (wf is True):
        if "MODELFL" in h.keys():
            print("MODELFL")
            if 'adapt' in h["MODELFL"]:             # MULTIVP
                data1 = 'f' + h["MODELFL"].split('.', -1)[0]
            else:                                   # PLUTO
                data1 = 'f' + h["MODELFL"].split('.', -1)[1]
            str1 = h["INSTRUME"] + "{}".format(h["DETECTOR"])
        elif "FILENAME" in h.keys():                # LASCO data
            data1 = 'f' + h["FILENAME"].split('.', -1)[0]
            str1 = h["INSTRUME"] + "{}".format(h["DETECTOR"])
        elif h["INSTRUME"].find('AIA') != - 1:
            data1 = 'f' + h['DATE-OBS'].split(':')[0]
            str1 = h["INSTRUME"] + "_{}".format(h["WAVELNTH"])
        else:                                       # K-Cor data
            data1 = 'f' + h['DATE-OBS'].split('T')[0]
            str1 = 'K-Cor'
        gentit = str1 + "_LON{:.1f}".format(np.degrees(h['LON'])) + "_" + r_str
        titlegif1 = 'latprof_' + gentit + data1
        if sv is True:
            fig.savefig(dirw + titlegif1 + ".png")
            fig2.savefig(dirw + 'img_' + titlegif1 + ".png")
            print("saved images", titlegif1 + ".png")
            print("saved images", 'img_' + titlegif1)
        if wf is True:
            err = np.zeros_like(cuts)
            for i in range(len(r)):
                titlefits = 'latprof_' + str1 + \
                "_LON{:.1f}".format(np.degrees(h['LON'])) + \
                '_R' + str(r[i]) + '_' + data1
                data = np.column_stack((pa, cuts[:,i].ravel(), err[:,i].ravel()))
                header = {"x_unit": "Radians", "inst": h["INSTRUME"],
                          "d_unit": h["D_UNIT"], "FILE": titplot,
                          "r_dist": str(r[i]), "max_I": str(max_I[i]),
                          "max_I_pa": str(max_I_pa[i]), "LON": h["LON"]}
                spugen.wr_fits(dirw, data, filename=titlefits, h=header, 
                               overwrite=True)
    fig.show()
    fig2.show()
    return
#    return cuts, pa