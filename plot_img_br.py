#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:18:32 2020

Plot an image with Br in the plane of the sky

@author: sparenti
"""

import PlutoUtils as pu
import SPUtils as spu
import pyPLUTO as pp
import numpy as np
from astropy.io import fits
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import SPUtils_gen as spugen
import os

path = (r"/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/"
        r"CEA/VSWMC/VSWMC_DATA/")
dirm = os.path.join(path, "VICTOR_EX/WET_3D_HLL_PSP_E1/")
dirf = os.path.join(path, 'SIMU_UV/IMG_DIR/VICTOR_WET3D/AIA_20H/SYN_WL/LASCO_pB/')
file = 'PLUTO-LASCO_20h_NAX512_PSHAPE91.0_VOX336_LON215.3_WET.fits'
image, h = spugen.open1img(dirf, file)
#path = spugen.set_path(h)
num = h["MODELFL"].split('.')[1]
shape = 13  # Rsun

d = pp.pload(num, w_dir=dirm, datatype='vtk')
mod_sun = spu.synth_ini(dirm, filename=h["MODELFL"])
d.bx1 = d.bx1 * mod_sun.b0             # normalization factor
d.plfile = h["MODELFL"]
if not num in h["MODELFL"]:
    raise Exception("The simulation file is from another model")

rtp = [Ellipsis, h["LAT"], h["LON"]]
plan, hpl = spu.pluto2D_prep(d, 'Br', rtp, shape, z=0)

rsun = spugen.get_rsun(h)
ysun = rsun[:, int(h['NAXIS1']/2)]
xsun = rsun[int(h['NAXIS2']/2), :]

# Plot
cmap = mplcm.plasma
vmin = -3
vmax = 2.5 #1
ylim_min = 1e-4  #5e-4
ylim_max = 5e3  #50 
fig, ax = plt.subplots(1, 1, num=0, clear=True)
#fig.suptitle(titplot)

rsun1 = plt.Circle((0,0), radius=1, color='white')
ax.imshow(np.log10(image), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
           extent=[-xsun.max(), xsun.max(), -ysun.max(), ysun.max()])
ax.add_artist(rsun1)
