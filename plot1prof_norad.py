#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:19:16 2020

From the n latitudinal profiles extracts the peaks of intensity within the given
xlimits (a streamer for instance). Then fit I_max vs PA and PA vs. R".

- Either pass the list of files (files key), or a string to search for files 
(filestr key). Ex: ilestr='latprof_PLUTO*f0035.fits' 
- xlimits: [xmin_deg, xmmax_deg] selects a sub latitudinal profile to be analised

@author: sparenti
"""

import numpy as np
import matplotlib.pyplot as plt
import SPUtils_gen as spugen
import SPUtils_AIA as spuaia
import scipy.interpolate as interp
import matplotlib.cm as mplcm
import Profiles_class as profcls
import os
import glob


def plot1prof_norad(files=None, sv=None, xlimits=None, filestr='latprof_PLUTO',
                    wf=None):
    
    path = (r"/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/"
            r"CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/")
    dirf = os.path.join(path, 'LAT_PROF/LASCO_LATPROF/LON218/')    # Read the input
#    dirf = '/mnt/c/Users/Users/Susanna Parenti/Documents/LAVORO/FITS/MLSO/'
#    dirf = os.path.join(path, 'RES_DATA/RES_LASCO/')
    if files is None:
        files = glob.glob(dirf + filestr)
#    dirw = os.path.join(path, 'LAT_PROF/LASCO_LATPROF/LON218/')    
#    dirw = os.path.join(path, 'LAT_PROF/K-Cor_LATPROF/K-Cor_STR_INFO/')  # Write the output
    dirw = os.path.join(path, 'LAT_PROF/LASCO_LATPROF/LASCO_STR_INFO/')
# Prepare for the plot
    colors = ['g', 'c', 'r', 'y', 'm', 'b', 'limegreen']
    fig, ax = plt.subplots(1, 1, num=0, clear=True, figsize=(5, 6.4))   # I vs theta
    ax.set_xlabel('Position Angle (deg)')
    ax.set_yscale('log')
    fig2, ax2 = plt.subplots(1, 1, num=1, clear=True)  # theta vs R
    ax2.set_xlabel("R$_\odot$")
    ax2.set_ylabel("Position Angle of I_max (deg)")
    fig3, ax3 = plt.subplots(1, 1, num=2, clear=True)  # theta vs R
    ax3.set_xlabel("R$_\odot$")
    
    nmax = np.zeros(len(files))
    nmax_pa = np.zeros(len(files))
    nr = np.zeros(len(files))
    err = np.zeros(len(files))             #  by now
    for i in range(len(files)):
        filei = os.path.basename(files[i])
        this_prof = profcls.profiles(dirf=dirf, file=filei)
        print("..............................................")
        print("Treating file: ", files[i])
        if i == 0:
            if this_prof.inst.find('LASCO') != - 1:
                vmin = -1#-1
                vmax = 2. #1
                ylim_min = .1 #5e-2  1 
                ylim_max = 100 #50 #1e4  #5e3
            elif this_prof.inst.find('K-Cor') != - 1: 
                vmin = -1#-1
                vmax = 3 #2. #1
                ylim_min = 5 #5e-2  1 
                ylim_max = 3e3 #50 #1e4  #5e3
            elif this_prof.inst.find('AIA') != - 1:
                vmin = -1
                vmax = 3 
                ylim_min = 1
                ylim_max = 4e2 
            ax.set_ylabel(this_prof.data_u) 
            ax.set_ylim(ylim_min, ylim_max)
            LON = "_LON{:.1f}".format(np.degrees(this_prof.lon))
        if xlimits !=None:
            this_prof.get_sub_prof(xlimits)
            this_prof.plot_intmax()
#            print(this_prof.max_I, this_prof.max_I_pa)
        elif xlimits is None and this_prof.max_I == 0:
            raise ValueError("No I_max defined, xlimits should be passed")
        nmax[i] = this_prof.max_I
        nmax_pa[i] = this_prof.max_I_pa
        nr[i] = this_prof.rsun
        ax.plot(np.degrees(this_prof.x), this_prof.data, label="R$_\odot$= " + 
                str(this_prof.rsun), color=colors[i])

# Fit I vs PA
    cut_pa = np.linspace(nmax_pa.min()-.05, nmax_pa.max(), num=20)
#    interpol = interp.interp1d(nmax_pa, nmax,  kind='linear') #fill_value="extrapolate")
#    cut = interpol(cut_pa)
    cut_fit = np.polyfit(np.degrees(nmax_pa), nmax, 1)
    cut = cut_fit[1] + (np.degrees(cut_pa) * cut_fit[0])
    dPA_obs = np.degrees(nmax_pa.max() - nmax_pa.min())
    dPA_obs2 = abs(np.degrees(nmax_pa[0] - nmax_pa[len(files)-1]))
    print('dPA_obs2', dPA_obs2)
    print('dPA_obs', dPA_obs)
    
#    ax.plot(np.degrees(cut_pa), np.log10(cut))
#    ax.plot(np.degrees(cut_pa), cut, label='linear fit')
    if np.max(err) == 0:
        ax.plot(np.degrees(nmax_pa), nmax, "o")
    else:
        ax.errorbar(np.degrees(nmax_pa), nmax, yerr=err[good], label=label)

    
    ax.legend(loc='best')
    ax.set_title(this_prof.title + "\n dPA$_{{obs}}$: {0:4.1f}".format(dPA_obs) +
                 u"\N{DEGREE SIGN}")


# fit PA down to 1 Rsun
    cut_rfit3 = np.poly1d(np.polyfit(nr, np.degrees(nmax_pa), 3))
    cut_rfit1 = np.poly1d(np.polyfit(nr, np.degrees(nmax_pa), 1))
    rfit = np.linspace(1, nr.max(), num=10)
    ax2.plot(nr, np.degrees(nmax_pa), "o", label=this_prof.inst)
    ax2.plot(rfit, cut_rfit3(rfit), label='poly 3')
    ax2.plot(rfit, cut_rfit1(rfit), label='poly 1')
    ax2.legend(loc='best')
    struc_pos = str(round(np.degrees(np.average(nmax_pa))))
    ax2.set_title("dPA " + 'in ' + this_prof.title + '\n Structure average position ' + 
                  struc_pos +  u"\N{DEGREE SIGN}")

# fit I vs. Rsun

    rfit = np.linspace(nr.min()-.05, nr.max(), num=20)
#    cut_norad_x = np.linspace(nr.min()-.05, nr.max(), num=20)
    interpol = interp.interp1d(nr, nmax, kind='cubic', fill_value="extrapolate")
    cut_norad = interpol(rfit)
#    coef = np.polyfit(nr, nmax, 2) #, domain=[1, nr.max()])    
#    cut_norad = np.polynomial.polynomial.polyval(cut_norad_x, coef)
#    cut_norad = np.poly1d(np.polyfit(nr, nmax, 3))
    ax3.plot(nr, nmax, "s", label=this_prof.inst)
    ax3.plot(rfit, cut_norad, label='spline')
#    ax3.plot(rfit, cut_norad, label ='spline')
    ax3.legend(loc='best')
    ax3.set_ylabel("I_max " + this_prof.data_u)
    ax3.set_yscale('log')
    ax3.set_title(this_prof.title + '\n Structure average position ' + 
                  struc_pos +  u"\N{DEGREE SIGN}")

    if this_prof.title.find("PLUTO") == -1:
#        fl1 = this_prof.title.split(".")[0]
        fl1 = this_prof.title.split(".")[1]
        fl = fl1.split(" ")[1]
        print("Data")
    else:
#        fl = this_prof.title.split(".")[1]
         fl = this_prof.title.split(".")[2]
         
    gentit = this_prof.inst + LON + "_ST" + (struc_pos) + "deg_f" + fl
   
    titlegif1 = 'sing_profmax_' + gentit
    if sv is True:
        
        titlegif = "I_PA_" + gentit
        titlegif2 = "PA_R_" + gentit
        titlegif3 = "norad_I_R_" + gentit
        fig.savefig(dirw + titlegif + ".png")
        fig2.savefig(dirw + titlegif2 + ".png")
        fig3.savefig(dirw + titlegif3 + ".png")
        
        print("saved images", titlegif)
        print("saved images", titlegif2)
        print("saved images", titlegif3)


    if wf is True:                            #  write I_max vr R
        titlegif1 = 'sing_profmax_' + gentit
        data_max = np.column_stack((nr, nmax, err, nmax_pa))
        header_max = {"x_unit": "Solar radii", "inst": this_prof.inst,
                      "d_unit": this_prof.data_u, "pa_u": "Radians",
                      "data_f": titlegif3, "theta": struc_pos, "lon": this_prof.lon}
        spugen.wr_fits(dirw, data_max, filename =titlegif1, h=header_max, overwrite=True)

    return  this_prof