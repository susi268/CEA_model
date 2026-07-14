#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:50:29 2020

Reads a data file and make statistics 

@author: sparenti
"""

import numpy as np
from astropy.io import fits
import scipy.interpolate as interp
import sunpy.cm as cm
import matplotlib.pyplot as plt

mdir ='/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/'


def rd_fitsf(file): 
    data = fits.getdata(mdir+file)
    fig=plt.figure()
    em_exp=0.1
    plt.imshow(data[:,:,0]**em_exp, extent=[-2,2,-2,2])
    return data

ima=rd_fitsf('pippo.fits')

# Make the histogram

  plt.hist(ima, bins=125)