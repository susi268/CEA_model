#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 12:00:01 2020

Extract the image data within LIMITS and average along the latitude
two options: 
    without the keyword 'avg_Imax' get the average over limits
    with the keyword 'avg_Imax' get the average of 'ratio' from the I_max within limits.
    This means that limits should be large enough.
    avg_Imax: ex: = 0.8
    sv_Imax: save the pa and r of the interval of avg_Imax to be plotted later.
limits = [thetamin, thetamax]  in degrees
svlim: save the limits to be  plotted later
   
@author: sparenti
"""


import numpy as np
import matplotlib.pyplot as plt
import SPUtils_gen as spugen
import SPUtils_AIA as spuaia
import SPUtils_WL as spuwl
import scipy.interpolate as interp
import matplotlib.cm as mplcm
import Profiles_class as profcls
import os
import Polar_class as pol_class


def plot_avg_prof(file, limits, plt_max=None, wf=None, sv=None, pxbin=None, 
                  errp=None, avg_Imax=None, sv_Imax=None):
    
    path = (r"/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/"
            r"CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/")
    dirf = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/FITS/LASCO/'  #LASCO_2018110721h/'
#    dirf = os.path.join(path,'VICTOR_WET3D_RHO3/SIMU_156_12H/K_Cor_156_12h/')
##    dirf = os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_035_RHO1/'
#                               'LASCO_035/LASCO_RHO1_35_12h/')
#                               r'K-Cor_35/')
#    dirf = os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_087_RHO2/LASCO_20191106_087/LASCO_RHO2_87_20h/') #LASCO_87_IMALONKcor/')
#    dirf = os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_087_RHO2/'
#                        r'K-Cor_20191106_087/')
#                        r'LASCO_20191106_087/LASCO_RHO2_87_20h/LASCO_87_IMALONKcor/')
#                       r'LASCO_20191106_087/LASCO_RHO2_87_20h/')
#    dirf = os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_087_RHO2/K-Cor_20191106_087/')
    dirw =os.path.join(path, 'AVG_PROF/')
#    dirw = dirf  #os.path.join(path, 'RES_DATA/RES_LASCO/')


# Get the polar map
    polar = pol_class.polar_class(dirf=dirf, file=file, pxbin=pxbin)
    r_pol = polar.rsun
    img_polaroll = polar.data
    h = polar.h

#   Prepare to plot
    fig, ax = plt.subplots(1, 1, num=0, clear=True)
    fig2, ax2 = plt.subplots(1, 1, num=1, clear=True)
    if file.find("LASCO") != -1 or "LASCO" in h["INSTRUME"]!= -1:
        ax2.set_ylim(1e-1, 1e2)
        ax2.set_xlim(2.17, 6.6)
        vmin = -.01
        vmax = 1.2
        rpmax = 6.             # to impose a max in R when selecting the data for averaging 
        xplot_min = 2.25       # min to plot the map 
        axes_rmin = 2.25       # min to calculate the axes of the max I
        print("minimum distance for average position of the exes: ",  axes_rmin )
    elif (file.find("cor") != -1) or (file.find("Cor") != -1) is True:
        xplot_min = 1.15
        ax2.set_ylim(.1, 1e4)
        ax2.set_xlim(xplot_min, 3)
        vmin = 1
        vmax = 3
        axes_rmin = 2#1.15
        if file.find("PLUTO") != -1:
            rpmax = 3
        else:    
            rpmax = 2.18 #2.4 
#        image_cut = np.where(r_p < rpmax, image_cut, 0)

    else: 
        print("Not implemented yet")
        

#  Select data within LAT and radial limits 
    sub_ima, goodr = polar.get_sub_polar_r([axes_rmin, rpmax])  #in R
    polar.data = sub_ima
    polar.rsun = polar.rsun[goodr]
    sub_ima, good = polar.get_sub_polar(limits)     # in PA
    polar.data = sub_ima
    polar.x = polar.x[good]
    r_p =  polar.rsun
    pa = np.degrees(polar.x)
    pa_rp0 = np.zeros(r_p.shape) + limits[0]
    pa_rp1 = np.zeros(r_p.shape) + limits[1]


#  Plot polar image
    im=ax.imshow(np.log10(img_polaroll),  extent=[r_pol.min(), r_pol.max(), 0, 360],
                                    origin='lower', aspect='auto', vmin=vmin, 
                                    vmax=vmax, cmap='RdYlBu_r')
    ax.set_xlabel('$R_\odot$', fontsize=10)
    ax.set_ylabel('PA ($^\circ$)', fontsize=10)
#    clb = plt.colorbar(im, ax = [ax], location = 'left')
#    clb.set_label('log(B/B$_\circ$)')

    ax.plot(r_p, pa_rp0, color='m')
    ax.plot(r_p, pa_rp1, color='m')
    ax.set_xlim(xplot_min,  rpmax)
    ax.set_ylim(40, 160)
    ax.set_title('PA: {:4.1f}$^\circ$ - {:4.1f}$^\circ$'.format(limits[0], limits[1]))
#    ax.set_title("K-Cor - November 7", fontsize=12) #K-Cor - November 7")


# get the max for each LAT
    if plt_max is True:
        max_ind = polar.get_ImaxPA()            #sub_image=sub_ima)
        rgood = np.where(r_p < rpmax)
        ax.plot(r_p[rgood], pa[max_ind[rgood]], color='black')
        if avg_Imax is not None:
            image_cut, img_cut_pa = polar.get_avg_from_max(ratio=avg_Imax, silent=True)
            ax.plot(r_p[rgood], np.degrees(img_cut_pa[rgood,0].ravel()), color='black')
            ax.plot(r_p[rgood], np.degrees(img_cut_pa[rgood,1].ravel()), color='black')
            max_str = '_Imax' + str(avg_Imax)
        else:
            image_cut = np.average(sub_ima, axis=0)
            max_str = ''
        
    if "MODELFL" in h.keys():
        if 'adapt' in h["MODELFL"]:             # MULTIVP
            data1 = 'f' + h["MODELFL"].split('.', -1)[0]
        else:                                   # PLUTO
            data1 = 'f' + h["MODELFL"].split('.', -1)[1]
        str1 = h["INSTRUME"] + "{}".format(h["DETECTOR"])
        err = np.zeros_like(image_cut)
        print("Error not yet implemented")
    elif "FILENAME" in h.keys():                # LASCO data
        data1 = 'f' + h["FILENAME"].split('.', -1)[0]
        str1 = h["INSTRUME"] + "{}".format(h["DETECTOR"])
        err = spuwl.get_LASCO_pBerr(image_cut, r_p)
    elif h["INSTRUME"] == 'AIA_2':
        data1 = 'f' + h['DATE-OBS'].split('T')[0]
        str1 = "AIA"
        err = spuaia.get_perr(image_cut, band)
    else:                                       # K-Cor data
        data1 = 'f' + h['DATE-OBS'].split('T')[0]
        str1 = 'K-Cor'
        err = spuwl.get_KCor_perr(image_cut)

    pos = str(round(np.average(limits)))
    LON = "_LON{:.1f}".format(np.degrees(h["LON"]))
    gentit = str1 +  max_str + LON + "_ST" +  pos + "deg_" + data1
    print(gentit)

    if errp is None:
        ax2.plot(r_p, image_cut)
    else:
        ax2.errorbar(r_p, image_cut, yerr=err)
   
    if "D_UNIT" in h.keys(): 
        D_UNIT = h["D_UNIT"]
    elif "UNIT" in h.keys():
        D_UNIT = h["UNIT"]
    else:
        D_UNIT = ""
    ax2.set_ylabel(D_UNIT)
    ax2.set_xlabel('$R_\odot$', fontsize=10)
    ax2.set_yscale('log')
    ax2.set_title('Average profile at PA: {:4.1f}$^\circ$ - {:4.1f}$^\circ$'.format(limits[0], limits[1]))

    if sv:
        titlegif = 'img_avgprof_' + gentit
        fig.savefig(dirw + titlegif + ".png")
#        fig2.savefig(dirf1 +  'img_'+ titlegif1+ ".png")
        print("saved images", titlegif + ".png")
#        print("saved images",'img_'+ titlegif1)
        
    if wf is True:                            #  write I_max vr R
        titlefits = 'avg_prof_' + gentit
        data = np.column_stack((r_p, image_cut, err))
        header = {"x_unit": "Solar radii", "inst": h["INSTRUME"], 
                  "d_unit": h["D_UNIT"], "data_f": gentit, "theta": pos,
                   "lon" : h["LON"]}
        spugen.wr_fits(dirw, data, filename=titlefits, h=header, overwrite=True)
    
    if sv_Imax is True:
        int_Imax=np.column_stack((image_cut, img_cut_pa[rgood,0].ravel(), 
                                  img_cut_pa[rgood,1].ravel(), r_p[rgood]))
#                print('int_Imax.shape;', int_Imax.shape)
        Imax_name= 'dPA_Imax_' + gentit
        header_Imax = {"file": file, "limits_1": limits[0], "limits_2": limits[1],
                       "x_unit": "Solar radii", "y_unit":'rad'}
        spugen.wr_fits(dirw, int_Imax,  filename=Imax_name, h=header_Imax, overwrite=True)
        

    return #image_cut