#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 08:55:52 2020

Create a class containing the image in polar coordinates

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

path = (r"/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/"
        r"CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/")
#dirf = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/FITS/LASCO/'
#            dirw = os.path.join(path, 'AVG_PROF/')       
#    dirf = os.path.join(path,'VICTOR_WET3D_RHO3/SIMU_156_12H/LASCO_156_12h/')
#    dirf = os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_035_RHO1/LASCO_035/LASCO_RHO1_35_20h/')
#dirf = os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_087_RHO2/LASCO_20191106_087/LASCO_RHO2_87_20h/')

class polar_class():
    
    def __init__(self, dirf=None, file=None, data=None, x=None, rsun=None, 
                 pxbin=None):        
       
        if file is not None:
            # Create the polar image
#            nr = 512
            pa = np.arange(360)
            data, h = spugen.open1img(dirf, file)
            if pxbin != None:
                newshape=(int(h["NAXIS1"]/pxbin), int(h["NAXIS2"]/pxbin))
                data, h = spugen.image_rebin(data, newshape, h=h)

            x1, y1 = spugen.get_1xy(h)
            nr = h["NAXIS1"]
#            r = np.sqrt(x1[256:]**2 + y1[256:]**2)
            r = np.sqrt(x1[int(h["CRPIX1"]):]**2 + y1[int(h["CRPIX2"]):]**2)
#            print(r.min(), r.max())
            r_p = np.linspace(r.min(), r.max(), num=nr)
            img_polar = spugen.cart2polar(data, h=h, output_shape=(360, nr))
            img_polar = img_polar[::-1,:]
            img_polaroll = np.roll(img_polar, 90, 0)

            # fills the structure
            self.data = img_polaroll
            self.x = np.radians(pa)
            self.data_u = h["d_unit"]
#            self.err = np.zeros(img_polaroll.shape)
            self.inst = h["INSTRUME"]
            self.rsun = r_p
            self.title = file
            if "MODELFL" in h.keys():
                self.file = h["MODELFL"]
            elif 'FILENAME' in h:
                self.file = h['FILENAME']
            else:
                self.file = h['DATE-OBS'].split('T')[0]    #  K-Cor
            self.lon = h["LON"]
            self.h = h
            if ("radius") in h:
                self.radius=h["radius"]
            else:
                self.radius=h["D"]
        else:
            if file is None and data is None:
                raise ValueError('You should pass either a file or data')
            else:                
                self.data = data
                if x is None:
                    raise ValueError('You have to pass X (PA in degrees)')
                else:
                    self.x = np.radians(x)
                self.data_u = None
#                self.err = np.zeros(data.shape)   # by now
                if rsun != None: 
                    self.rsun = rsun    
                else:
                    raise ValueError('You have to pass RSUN')
                self.file = None
                self.h=None
        self.x_u = 'radians'
        self.rsun_u = 'Rsun'
        self.max_I = None
        self.max_I_x = None
        self.emin_x = []
        self.emax_x = []


    def get_sub_polar(self, limits):
        """
        Select data (image) within limits in polar angle
        """
        pa = np.degrees(self.x)
        good = np.logical_and(pa >= limits[0], pa <= limits[1])
        sub_ima = self.data[good, :]
        return sub_ima, good
    
    def get_sub_polar_r(self, limits):
        """
        Select data (image) within limits in solar distance
        """
        good = np.logical_and(self.rsun >= limits[0], self.rsun <= limits[1])
        sub_ima = self.data[:,good]
        return sub_ima, good

    
    def get_ImaxPA(self, sub_image=None, sub_image_pa=None):
        """
        Get the max for each PA 
        """
        if sub_image is None:
            sub_image = self.data
#        max_I = sub_image.max(axis=0)
        max_ind = sub_image.argmax(axis=0)
        pos_ax = np.average(self.x[max_ind])
        print("Average position of the axes {}".format(str(np.degrees(pos_ax))))
        return max_ind


    def get_avg_from_max(self, ratio=0.3, silent=True):
        """
        Get the averaged of data on a dPA which corresponds to I_max*ratio
        """
        cut = np.zeros(len(self.rsun))
        cut_pa = np.zeros((len(self.rsun), 2))
        for kk in range(len(self.rsun)):
            prof = profcls.profiles(data=self.data[:,kk].ravel(), 
                                    x=self.x, 
                                    rsun=[self.rsun[kk]])
            this_cut = prof.get_sub_Imax(silent=silent, ratio=ratio)
            cut[kk] = np.average(this_cut)
            cut_pa[kk,:] = [prof.emax_x, prof.emin_x]     #extremes of the cut
        print("Get_avg_from_max done.")    
        return cut, cut_pa
        
        

    def get_Imax_dist(self):
        """
        Once the position of I_max for each R is find, it make statistics
        """
        
        