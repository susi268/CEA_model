#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:36:51 2019

@author: sparenti
"""

import numpy as np
import pyPLUTO as pp
import PlutoUtils as pu
import scipy.interpolate
import matplotlib.pyplot as plt

#w_dir='/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SOUMITRA_EX/'
wdir = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/VICTOR_EX/SImulation_2019_11_06/'


class pippo():
    def __init__(self, wdir,GEOM="Spherical",filename='data.0087.vtk',
                 filter=None,dtype="vtk",view_theta=np.pi/2.,view_phi=0.0,res=200):
        self.data_dir=wdir

        self.res=res

        out=filename.split('.')
        self.outnum=int(out[1])
        self.dtype=out[2]
        
        self.kb=1.3807e-16
        self.mp=1.6726e-24

        # PLUTO normalization
        self.v0=4.367e7
        self.n0=1e8
        self.l0=6.9599e10

        D=pp.pload(self.outnum,w_dir=self.data_dir,datatype=self.dtype)
        pu.get_caseparams(D)
        pu.get_userdefconst(D)
 
        D.T=self.mp*self.v0**2/(2*self.kb)*D.prs/D.rho
        self.iTemp=scipy.interpolate.RegularGridInterpolator((D.x1,D.x2,D.x3),D.T,bounds_error=False,fill_value=0.0)
        self.iDens=scipy.interpolate.RegularGridInterpolator((D.x1,D.x2,D.x3),D.rho,bounds_error=False,fill_value=0.0)
            
        self.r,self.theta,self.phi=np.mgrid[0.0:2.5:(self.res+1)*1j,0:np.pi:(self.res+1)*1j,0:2*np.pi:(2*self.res+1)*1j]

        if (GEOM=="Cartesian"):
            self.Tsph=self.iTemp((self.r*np.sin(self.theta)*np.cos(self.phi),self.r*np.sin(self.theta)*np.sin(self.phi),self.r*np.cos(self.theta)))
            self.Nsph=self.iDens((self.r*np.sin(self.theta)*np.cos(self.phi),self.r*np.sin(self.theta)*np.sin(self.phi),self.r*np.cos(self.theta)))*self.n0
        elif (GEOM=="Spherical"):
            self.Tsph=self.iTemp((self.r,self.theta,self.phi))
            self.Nsph=self.iDens((self.r,self.theta,self.phi))*self.n0

        pu.save_elements([self.r,self.theta,self.phi,self.Tsph,self.Nsph],self.data_dir+"KernelEm")
    
    
tt = pippo(wdir)
plt.plot(tt.r[:,0,0], tt.Nsph[:,0,0])  
    