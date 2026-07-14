#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 15:43:39 2020

Utils used by the code that creats synthetic images

@author: sparenti
"""


from PLUTO_soft import PlutoUtils as pu
from PLUTO_soft import pyPLUTO as pp
import numpy as np
from astropy.io import fits
import glob
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import fitsarray as fa
import SPUtils_gen as spugen
import tomograpy

#wdir = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/VICTOR_EX/'
#wdir = '/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/CEA_SOLO_SIMU/VICTOR_20210521_SOLO_PER1-2/PER1_SIMU/'
#wdir = '/home/sparenti/WORK/GLOBAL_MODEL/SOLO_PERI2_DATA/1502/'
wdir = '/mnt/c/Users/sparenti/Documents/LAVORO/GLOBAL_MODEL/20210115_WP/'

class synth_ini(object):
    def __init__(self, wdir, filename=None, dtype="dbl"):
#    def __init__(self, wdir, filename=None, dtype="vtk"):
        self.dtype = dtype
        # Outputs in the database
        if(filename == None):
            try:
                self.outnum=pu.find_last_output(self.dtype,self.data_dir)
            except:
                print(("No {0} found in {(1}".format(self.dtype,self.data_dir)))
                raise interp.RegularGridInterpolator
        else:
            out=filename.split('.')
            self.outnum=int(out[1])
            self.dtype=out[2]
 
        self.coronal_radius=1.0
        self.wdir=wdir
# Useful constants (cgs) 
        self.kb=1.3807e-16
        self.mp=1.6726e-24

# PLUTO normalization
#        self.v0=4.367e7
#        self.n0=1e8
#        self.l0=6.9599e10 
        print('wdir: ', self.wdir)
        pu.get_userdefconst(self)
#        pu.get_userdefconst()
        # breakpoint()
        self.v0 = self.UNIT_VELOCITY
        self.n0 = self.UNIT_DENSITY / self.mp
        self.l0 = self.UNIT_LENGTH
        self.b0 = np.sqrt(4 * np.pi * self.UNIT_DENSITY * self.v0**2)
        


"""
 Create a geeral header for the MODEL (PLUTO) data
"""
def PLUTO_wr_header(data, source, im_pshape=None, dtype=None):

# generate header
    header = dict()
#    header["PLUTOFL"] = ""
    header["MODELFL"] = ""
    header['PIXBIN'] = ""                        # Binning applied to WL
    header['DATE_OBS'] = ""
    header['SIMPLE'] = True
    header['BITPIX'] = fa.bitpix_inv[dtype.__name__]
#    header['NAXIS'] = data.ndim   # if called by pluto2D_prep
    header['NAXIS'] = 2
    header['INSTRUME'] = source + '-WL'   # 'LASCO' #"PLUTO-LASCO"
    header['DETECTOR'] = ""
    header['WAVELNTH'] = ""
    header['D_UNIT'] = ""
    header['PIXBIN'] = ""                        # Binning applied to WL
    header['EXPTIME'] = ""
    header["radius"] = 212.45  #210.6
    for i in range(2):
        if source == 'FLOR':
            if im_pshape is None:
                raise ValueError("PLUTO_wr_header: you should pass 'im_pshape'")    
            header['NAXIS' + str(i + 1)] = 256
            header['CDELT'+ str(i + 1)] = im_pshape/ header['NAXIS' + str(i + 1)]
            header["CRVAL" + str(i + 1)] = 0
            print("Set the header for CME simu:")
            print("IM_PSHAPE: ", np.degrees(im_pshape))
            print("NAXIS: {}, CDELT: {}, CVAL: {}".format(header['NAXIS' + str(i + 1)],  
                  header['CDELT'+ str(i + 1)],  header["CRVAL" + str(i + 1)]))
        else:    
            header['NAXIS' + str(i + 1)] = data.shape[i]  # if called by pluto2D_prep
            header['CDELT'+ str(i + 1)] = ""         # radians
            header["CRVAL" + str(i + 1)] = ""
        header['CRPIX' + str(i + 1)] = (header['NAXIS' + str(i + 1)] - 1)/ 2 + 1          
#    header['CRPIX1'] = (header['NAXIS1'] + 1)/ 2 - 4
#    header['CRPIX2'] = (header['NAXIS2'] + 1)/ 2 - 4
        
    header['LAT'] = 0                     # radians 
    header['LON'] = 0                      # np.radians
    header["RSUN_OBS"] = np.degrees(np.arctan2(6.96349e5, 1.46619e8)) * 3600.   # in arcsec
    tomograpy.siddon.map_borders(header)
    return header


def pluto2carr_phi(D, phi0_car):
    """
    Convert the PLUTO longitudes (LON = 0 at meridian) in Carrington.
    phi0_car: Carrington LON (radians) of the wanted day/h
    """
    phi_car = np.linspace(phi0_car, D.x3.max() + phi0_car,
                          D.n3_tot) % (2*np.pi)
    return phi_car


def load_pluto_logTN(D, mod_sun, GEOM="Spherical"):
    """load PLUTO density and temperature"""
    D.logT = np.log10(mod_sun.mp * mod_sun.v0**2 / (2 * mod_sun.kb) * D.prs / D.rho)
    # n from rho in SI
    D.logN = np.log10(D.rho * mod_sun.n0 * 1e6)
    return D


def pluto2D_prep(D, pluto_par, rtp, pshape, z=0, svf=None, svg=None):
    """
     From a 3D PLUTO parameter extract a 2D data on the plane of the sky given by
     coords2carr
     Prepare for a comparison with a similar image, for instance, from LASCO:
         Select the plane of the sky [rtp] in Carrington coords
         Select the parameter to be plotted 
         Interpolate the parameter on the rtp plane.         
     pluto_par: d.logN,  logT or B
     pshape: Dimension (in Rsun) of the 2D image to be extracted extract 
     shape: Dimension in Rsun of the 2D image to extract 
     rtp : [r, theta, phi] in Carrington convention (degrees)
 
"""

    
    wdir = '/mnt/c/Users/sarenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/VICTOR_EX/'
    shape = 512
    y, x = np.indices((shape, shape), dtype='float32')
    x = (x * pshape/(shape -1)) - (pshape/2)
    y = (y * pshape/(shape -1)) - (pshape/2)
    z=z
# Get the Carrington Coords of the plane xy at rtp
    xc, yc, zc = spugen.coords2carr(x, y, z, rtp[1], rtp[2])
# Transform in spherical coords
    r_c = np.sqrt(xc**2 + yc**2 + zc**2)
    phi_c = (np.arctan2(yc, xc)  + 2*np.pi ) % (2*np.pi)      #  - pi < arctang1 < pi 
    theta_c = np.arctan2(zc, np.sqrt(xc**2 + yc**2))
    theta_c = (np.pi/2) - theta_c          # Pluto uses the co-latitude
    
    if pluto_par == 'Ne':
        this_par = 10**(D.logN - 6)
        this_par[this_par < 0] = 0
 #       this_par[:, 96//2, 192//4 * 3] = 1e13
        this_par_print = 'Ne'
        unit = "cm^-3"
    elif pluto_par == 'T':
        this_par = D.logT
        this_par_print = 'T'
        unit = "K"
    elif pluto_par.find('B') != -1:       # Br
        if pluto_par == 'Br':
            this_par = D.bx1
        elif pluto_par == 'Bt':      # Btheta
            this_par = D.bx2
        else:                        # Bphi
            this_par = D.bx3
        this_par_print = pluto_par
        unit = "Gauss"
    else:
        raise Exception("You need to pass one PLUTO parameter")
    this_par2 = np.concatenate((this_par, this_par[:, :, 0:1]), axis=2)
    this_par2 = np.concatenate((this_par[:, :, this_par.shape[2] - 
                                         1 : this_par.shape[2]], this_par2), axis=2)

    x3_1 = ((np.arange(1, dtype=np.float32)+1.)*D.dx3[0])+np.max(D.x3)
    x3_2 = ((np.arange(1, dtype=np.float32)-1.)*D.dx3[0])+np.min(D.x3)
    x3 = np.concatenate((D.x3, x3_1))
    x3 = np.concatenate((x3_2, x3))

    print("Selected", this_par_print)
# Interpolate pluto_par on the wanted plane
    func_int = interp.RegularGridInterpolator((D.x1, D.x2, x3),
                                               this_par2, bounds_error=False,
                                               fill_value=0.0)
    plan = func_int((r_c, theta_c, phi_c))
# Get the header
    header = PLUTO_wr_header(plan, 'PLUTO')
    for i in range(2):
      header['CDELT'+ str(i + 1)] = np.arctan2(pshape/shape, header["radius"])
      header["CRVAL" + str(i + 1)] = 0.
    header['D_UNIT'] = unit
    header['WAVELNTH'] = this_par_print
    header['LON'] = np.radians(rtp[2])
    header['LAT'] = np.radians(rtp[1])
    if hasattr(D, 'plfile'):
        header['MODELFL'] = D.plfile
# Plot
    dist = np.tan(header['CDELT1']) * header['NAXIS1'] * header["radius"]

    fig, ax = plt.subplots(1,1, num=2, clear=True)
    if pluto_par == 'Ne':
        ax.imshow(np.log10(plan), origin='lower', extent=[-dist/2, dist/2, -dist/2, dist/2])
              #vmin =4, vmax=8)
    elif pluto_par == 'T':
        ax.imshow(plan, origin='lower', extent=[-dist/2, dist/2, -dist/2, dist/2],
                  vmin=5.5, vmax=6.3)
    else:    #br
        ax.imshow(plan, origin='lower', vmin=-1, vmax=1) 
    ax.set_ylabel('Rsun')
    ax.set_xlabel('Rsun')
    ax.set_title(pluto_par)
# Save
    fl = D.plfile.split('.')[-2]
    filename = 'PLUTO_' + fl + '_' + this_par_print +'_LAT' + str(rtp[1]) + '_LON' + str(rtp[2])

    if svg == True:
        fig.savefig(wdir + filename + ".png", bbox_inches="tight")
        print("Written the file", filename +'.png')            
    if svf != None:
       
        spugen.wr_fits(wdir, plan, h = header, filename=filename)
    return plan, header



def plot_pluto_logNT(D, rtp, mod_sun=None, sv=None, xlim = [1, 6.6]):
    """    
     Plot either Ne or T vs dime, from PLUTO output
     rtp = [:, 90, 270] = [:, 49, 145]  
     rtp = [Ellipsis, 50, 145]  from grid.out file    
     dime (TBD): dimension of rtp to plot. Choise: 1, 2 or 3
    """
    
#    wdir = '/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/VICTOR_EX/SImulation_2019_11_06_87_rho2/'
    wdir = '/home/sparenti/WORK/GLOBAL_MODEL/TEST_CODE/'
    if mod_sun != None:
        load_pluto_logTN(D, mod_sun, GEOM="Spherical")
    fig, ax = plt.subplots(2,1)
    ax[0].set_title('Profile at ({}, {:2.1f},{:2.1f})'.format(rtp[0], np.degrees(D.x2[rtp[1]]),np.degrees(D.x3[rtp[2]])))
    ax[0].plot(D.x1, D.logN[rtp[0], rtp[1], rtp[2]] - 6)
    ax[0].set_ylabel("logN (cm-3)")
    ax[0].set_xlabel("R_sun")
#    ax[0].set_yscale('log')
    ax[0].set_xlim(xlim[0], xlim[1])
#    ax[0].set_ylim(1e4, 5e6)
#    ax[0].set_ylim(4,7)

    ax[1].set_title('')
    ax[1].plot(D.x1, D.logT[rtp[0], rtp[1], rtp[2]])
    ax[1].set_ylabel("logT")
    ax[1].set_xlabel("R_sun")
    ax[1].set_xlim(xlim[0], xlim[1])
    ax[1].set_ylim(5.6, 6.3)
    if sv != None:
        fig.savefig(wdir + 'PLUTO_NT.png')
    return


def plot_B(D, h, pshape):
    """
      From the model object extracts B and project it in a 2D plane. 
      This will be protted 
      h: header of the image to oplot B
    """
    wdir = '/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/VICTOR_EX/WET_3D_HLL_PSP_E1/'
    mod_sun = synth_ini(wdir, filename=h["MODELFL"])
    D.plfile = h["MODELFL"]
#    if not num in h["MODELFL"]:
#        raise Exception("The simulation file is from another model")

    rtp = [Ellipsis, h["LAT"], h["LON"]]

    br, hpl = pluto2D_prep(D, 'Br', rtp, pshape, z=0)
    br *= mod_sun.b0                    # normalization factor
    bt, hbt = pluto2D_prep(D, 'Bt', rtp, pshape, z=0)
    bt *= mod_sun.b0
    bp, hbp = pluto2D_prep(D, 'Bp', rtp, pshape, z=0)
    bp *= mod_sun.b0

    rsun = spugen.get_rsun(h)
    ysun = rsun[:, int(h['NAXIS1']/2)]
    xsun = rsun[int(h['NAXIS2']/2), :]
 
    N_lines = 5
    r_seed = 1.1 * np.ones(N_lines)
    th_seed = np.linspace(0, np.pi, num=N_lines)
    x_seed = r_seed * np.cos(th_seed)
    y_seed = r_seed * np.sin(th_seed)
    
    # Get field lines, assuming bx1 and bx2 are 2D in (r,theta) plane of the sky
    Im = pp.Image()
    [flx, fly] = Im.ASfieldlinesSph(br, bt, D.x1, D.x2, D.dx1, D.dx2,
                                    x_seed, y_seed, step_min=1.e-3, rmax=7, rs=1)

    # Plot field lines
    for ilx, lx in enumerate(flx):
        plt.plot(lx, fly[ilx], color='w')
    return


class aia_obj(object):
    def __init__(self):
       self.logN_em_min=8
       self.logN_em_max=18
       self.logN_em_units="log10"
       self.logT_em_min=4
       self.logT_em_max=8
       self.logT_em_units="log10"

       self.emiss_units="dn/s/m"
       self.aia_em = None 
       self.filter_cmap_dict={"94":plt.get_cmap("sdoaia94"),
                              "131":plt.get_cmap("sdoaia131"),
                              "171" :plt.get_cmap("sdoaia171"),
                              "174" :plt.get_cmap("sdoaia171"),
                             "193":plt.get_cmap("sdoaia193"),                              
                              "211":plt.get_cmap("sdoaia211"),
                              "304":plt.get_cmap("sdoaia304"),                              
                              "335":plt.get_cmap("sdoaia335")}

       
       
    def read_aia_emiss(self, aia_filter, aia_dir="/mnt/c/Users/sparenti/Documents/LAVORO/FITS/CHIANTI_FITS/"):
        if aia_filter != '174':
            this_st="_24-jul-2012_sun_coronal_1992_feldman_ext.abund_chianti.ioneq_synthetic.fits"  #AIA
        else:
            this_st="_filter25_sun_coronal_1992_feldman_ext.abund_chianti.ioneq_synthetic.fits"    #EUI
        aia_f = glob.glob(aia_dir+ "*"+ aia_filter + this_st)
        if aia_f: 
            print("READ_AIA_EMISS - file fits for the emissivities: ")
            print(aia_filter + ' ' + aia_f[0])
            print("-------------------")
        else:
            print("-------------------")
            print("file fits not found")

        self.aia_em=fits.getdata(aia_f[0])
        self.logN_em=np.linspace(self.logN_em_min, self.logN_em_max, self.aia_em.shape[0])
        self.logT_em=np.linspace(self.logT_em_min, self.logT_em_max, self.aia_em.shape[1])
        self.aia_filter=aia_filter

# Interpolate the emissivities in logT and logN from the PLUTO model output         
    def emis_interpol(self, D):
        emis_interpolator=interp.RegularGridInterpolator((self.logN_em,
                                                          self.logT_em),
                                                         self.aia_em,
                                                         bounds_error=False,
                                                         fill_value=0.0)
#        D.logN[D.logN < self.logN_em_min] = 0.
#        D.logT[D.logT < self.logT_em_min] = 0.
        return emis_interpolator((D.logN, D.logT))
 
  
    


