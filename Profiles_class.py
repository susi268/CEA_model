#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:58:52 2020

Create an object for a data profile (1D) obtained runing plot_lat slice, or instance.
pass either data, x, r or dir, file, obtained running plot_lat_prof 

@author: sparenti
"""
import numpy as np
import matplotlib.pyplot as plt
import SPUtils_gen as spugen
import math
import os

path = (r'/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/'
            r'VSWMC_DATA/SIMU_UV/IMG_DIR/')
#dirw = os.path.join(path, 'LAT_PROF/LASCO_LATPROF/LON218/')
dirw = os.path.join(path, 'RES_DATA/RES_LASCO/')


class profiles():
    def __init__(self, dirf=dirw, file=None, data=None, x=None, rsun=None):
        if file != None:
            data, h = spugen.rd_fitsf(dirf, file)
            self.data = data[:, 1].ravel()
            self.x =  data[:, 0].ravel()
            self.x_u =  h["x_unit"]
            self.data_u = h["d_unit"]
            self.err = data[:, 2].ravel()
            if "MAX_I" in h.keys():
                self.max_I = h["max_I"]
            else:
                self.max_I = []
            if "MAX_I_PA" in h.keys(): 
                self.max_I_pa = h["max_I_pa"]
            else:
                self.max_I_pa = []
            self.emin_x = []
            self.emax_x = []
            self.inst = h["inst"]
            self.rsun = h["r_dist"]
            self.title = h["FILE"]
            self.file = file
            self.lon = h["LON"]
        else:
            if file is None and data is None:
                raise ValueError('You should pass either a file or data')
            else:
                if len(data) != len(x):
                    raise ValueError(" Data and X should have the same size")
                self.data = data.ravel()
                self.x = x.ravel()
                self.data_u = None
                self.x_u = None
                self.err = np.zeros(self.data.shape)   # by now
                if rsun != None: 
                    self.rsun = rsun[0]
                else:
                    self.rsun = None
                self.file = None
                self.max_I = []
                self.max_I_pa = []
                self.emin_x = []
                self.emax_x = []
                self.inst = None
                self.title = None
                self.file = None
                self.lon = None


    def get_sub_prof(self, xlimits, plot=True):
        """
        xlimits = [lim_min, lim_max] in degrees
        case_lim: say at what correstponds the xlimits. "FREE", "1e" 
        """
        xlimits=np.radians(xlimits)
        if len(xlimits) != 2:
            raise ValueError("xlimits shoud be [lim_min, lim_max]")
        good = np.logical_and(self.x >= xlimits[0], self.x <= xlimits[1])
        if plot is True:
            fig = plt.figure(num=5, clear=True)
            plt.plot(np.degrees(self.x), self.data, color='b')
            plt.plot(np.degrees(self.x[good]), self.data[good], color='m')
            plt.title('Sub-structure set to {:4.1f}$^\circ$ - {:4.1f}$^\circ$'.format(np.degrees(xlimits[0]), np.degrees(xlimits[1])))
            plt.ylabel(self.data_u)
            plt.xlabel('Position Angle (deg)')

        self.x = self.x[good]
        self.data = self.data[good]
        self.err = self.err[good]
        print('------------------------------------------------------------')
        print('')
        print('GET_SUB_PROF: Data limited to (degrees): {:4.1f} - {:4.1f}'.format(np.degrees(xlimits[0]),
                                              np.degrees(xlimits[1])))
        print("=============")
#        print('------------------------------------------------------------')

    def get_1e(self, plot=True, sv=None, silent=None):
        """
        Provides 1/e value from the max of the intensity profile (data 1D).
        Returns the size in degrees of the structure at the 1/e peak value.
        x in degrees
        """
#        plt.figure()
        datamax = self.data.max()
        indexmax = np.argmax(self.data)
        maxangle = self.x[indexmax]
        e_data = datamax * math.exp(-1)
        good = self.data.ravel() >= e_data
#        print(np.argmin(good), np.argmax(good))
        goodmin = self.data[good][0]
#        goodmax = self.data[np.max(np.where(good))]
        goodmax = self.data[good][-1]
#
#        if abs(goodmax - goodmin) >= 
        minx = np.min(self.x[good])
        maxx = np.max(self.x[good])
        self.emin_x = minx
        self.emax_x = maxx
        this_size = np.degrees(np.abs(maxx - minx))
        if silent is not None:
            print('------------------------------------------------------------')
            print('')
            print("GET_1e:")
            print("=======")
            print('Max/e ', e_data, self.data_u)
            print('Max of I @ angle (degrees)', np.degrees(maxangle))
            print('Extremes I values', goodmin, goodmax)
            print('Extremes (degrees): {: 6.1f} {: 6.1f}'.format(np.degrees(minx), np.degrees(maxx)))
            print('Structure size (degrees): ', this_size)
        if plot is True:
            plt.plot(np.degrees(self.x), self.data, color='b')
            plt.plot(np.degrees(self.x[good]), self.data[good], color='g')
            plt.plot(np.degrees(minx), e_data, marker="D", color='black')
            plt.plot(np.degrees(maxx), e_data, marker="D", color='black')
            plt.plot(np.degrees(maxangle), datamax, marker="D", color='black')
            plt.xlabel('PA (degrees)')
            plt.ylabel(self.data_u)
            titlef = self.file.split('f', -1)[2]
            plt.title("{} f{}, {}R$_\odot$ - Size at $I_0$/e: {:5.1f}$^\circ$".format(self.inst, titlef, self.rsun, this_size))
        print('------------------------------------------------------------')
        if sv is True:
            filename = 'size1e_pa{:4.1f}'.format(np.degrees(maxangle))
            if self.file is not None:
                out = self.file.split('.fits')
                filename = filename + '_' + out[0]
            else:
                filename = filename + '_R' + str(self.rsun)
 #           if os.path.isfile(dirw + filename +".png"):
 #               print("Filename already exists")
 #               os.remove(dirw + filename +".png")
            plt.savefig(dirw + filename + ".png")
            print('Save file ', filename)
        return this_size
    
    
    def get_sub_Imax(self, ratio=0.3, silent=True):
        """
          Return a substructure (along self.x) of data within I_max*ratio
          max_index: you can already have the index of the max
        """

        datamax = self.data.max()
        indexmax = np.argmax(self.data)
        maxangle = self.x[indexmax]
        ratio_data = datamax * ratio
        good = self.data.ravel() >= ratio_data
        goodmin = self.data[good][0]
        goodmax = self.data[good][-1]
        minx = np.min(self.x[good])
        maxx = np.max(self.x[good])
        self.emin_x = minx
        self.emax_x = maxx
        if silent is not True:
            plt.plot(np.degrees(self.x), self.data, color='b')
            plt.plot(np.degrees(self.x[good]), self.data[good], color='g')
            plt.plot(np.degrees(minx), ratio_data, marker="D", color='black')
            plt.plot(np.degrees(maxx), ratio_data, marker="D", color='black')
            plt.plot(np.degrees(maxangle), datamax, marker="D", color='black')
            plt.xlabel('PA (degrees)')
            plt.ylabel(self.data_u)
#            titlef = self.file.split('f', -1)[2]
#            plt.title("{} f{}, {}R$_\odot$ - Size at $I_0$/e: {:5.1f}$^\circ$".format(self.inst, titlef, self.rsun, this_size))

            print('------------------------------------------------------------')
            print('get_sub_Imax:')
            print("Extraction the sub data within ", str(ratio), " from the peak")
            print("=======")
            print('Max*ratio ', ratio_data, self.data_u)
            print('Max of I @ angle (degrees)', np.degrees(maxangle))
            print('Extremes I values', goodmin, goodmax)
            print('Extremes (degrees): {: 6.1f} {: 6.1f}'.format(np.degrees(minx), np.degrees(maxx)))
        

#        sub_prof = self.get_sub_prof([np.degrees(minx), np.degrees(maxx)], plot=None)
        return self.data[good]
        
    
    def get_1r2(self):
        """
        get the function 1/r^2
        """
        maxv = np.argmax(self.data)
        r2 = 1/self.x**2 * self.data[maxv] * self.x[maxv]**2
        return r2
    
    
    def plot_intmax(self):
        """
        Plot a vertical line from the peak of the data. Can be used to follow 
        the non radial expantion of a structure
        """
        datamax = self.data.max()
        indexmax = np.argmax(self.data)
#        plt.plot(np.degrees([self.x[indexmax], self.x[indexmax]]), [0, datamax], 
#                linestyle='dotted')
        print("***** PA_m = {0:3.1f}:".format(np.degrees(self.x[indexmax])), "******")
        self.max_I = datamax
        self.max_I_pa = self.x[indexmax]

        
    def get_Itot(self, xlimits=None, sv=None):
        """
        Integrate the intensity within 1/e from the peak value. 
        xlimits select the interval of PA where to find the peak of I.In deg.
        Use get_sub_prof and get_1e
        """
        if xlimits is not None:
            self.get_sub_prof(xlimits, plot=True)
        self.get_1e(sv=sv)
        self.get_sub_prof(np.degrees([self.emin_x, self.emax_x]), plot=None)
        I_tot = self.data.sum()
        print('------------------------------------------------------------')
        print('')
        print("GET_Itot within the xlimits: {:10.1f}".format(I_tot), self.data_u)
        print("========")
        return I_tot