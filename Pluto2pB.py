#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:38:59 2020

Reads the output from .dbl or vtk (Pluto model) and projects the pB
along a 2pi orbit.
Steps:
    Extract the quantity (image_quant) to project


@author: sparenti
"""


import time
import numpy as np
import tomograpy
import lo
import matplotlib.pyplot as plt
import SPUtils as spu
import SPUtils_gen as spugen
import pyPLUTO as pp
import scipy.interpolate as interp

tt = time.time()
save = 0   # save the images in png
savef = 1  # save one image in fits file


# Read the output from the Pluto model .dbl or vtk
wdir = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/VICTOR_EX/SImulation_2019_11_06/'
mdir = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/'
num = 87
d = pp.pload(num, w_dir=wdir, datatype='vtk')
mod_sun = spu.synth_ini(wdir, filename="data.0087.vtk")
d = spu.load_pluto_logTN(d, mod_sun, GEOM="Spherical")
density=10**d.logN

# Define the observer distance and orbit
radius = 100             # observer distance (in solar radii)
max_lon = np.max(d.x3)   # phi max
min_lon = 0.
date_obs = "2018-11-06T12:00:00Z"
n_steps = 1            # observer orbit steps
#longit = np.linspace(min_lon, max_lon, n_steps)/(2.*np.pi)  # to be annotated on the imege


# Define the observed object to be projected
shape = int(d.n1_tot*0.5)       # number of voxels along each axis
pshape = 8                      # physical dimension of the object (solar radii)
object_header = tomograpy.siddon.centered_cubic_map_header(pshape, shape)
im_pshape = tomograpy.siddon.fov(object_header, radius)  # FOV in radiant
# im_pshape = 0.06
im_shape = 128                 # number of image pixels in X & Y (NAXIS)
image_header = tomograpy.siddon.centered_image_header(im_pshape, im_shape)
image_header['radius'] = radius
image_header['max_lon'] = max_lon
image_header['DATE_OBS'] = date_obs

#obj = tomograpy.simu.object_from_header(object_header, fill=0)
obj = tomograpy.centered_cubic_map(pshape, shape)

# Create x,y, z to be converted to spherical
x, y, z = np.indices(obj.shape)
CDELT1 = object_header["CDELT1"]
CRPIX1 = object_header["CRPIX1"]
CRVAL1 = object_header["CRVAL1"]
x = (x - CRPIX1) * CDELT1 + CRVAL1
y = (y - CRPIX1) * CDELT1 + CRVAL1
z = (z - CRPIX1) * CDELT1 + CRVAL1


# convert x,y,z to spherical for interpolating iaia_em into x,y,z
Rsph = np.sqrt(x**2 + y**2 + z**2)
p = np.sqrt(x**2 + y**2)
theta = np.arctan2(p, z)
phi = (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)
print(time.time() - tt)

x3_1 = ((np.arange(1)+1.)*d.dx3[0])+np.max(d.x3)
x3_2 = ((np.arange(1)-1.)*d.dx3[0])+np.min(d.x3)
x3 = np.concatenate((d.x3, x3_1))
x3 = np.concatenate((x3_2, x3))
density2 = np.concatenate((density, density[:, :, 0:1]), axis=2)
density2 = np.concatenate((density[:, :, density.shape[2] - 1: density.shape[2]], density2), axis=2)

# Prepare to interpolate density2
interpolator = interp.RegularGridInterpolator((d.x1, d.x2, x3),
                                              density2,
                                              bounds_error=False,
                                              fill_value=0)
print('Begin Interpolator', time.time() - tt)
# interpolate
#obj[:] = interpolator((Rsph, theta, phi))
obj = tomograpy.centered_cubic_map(pshape, shape)
obj[:] = 1
#obj[obj < 0] = 0
print('End Interpolation', time.time() - tt)

# Prepare for the projection along an orbit
#data = tomograpy.simu.circular_trajectory_data(n_images = n_steps, **image_header)
data = tomograpy.centered_stack(im_pshape, im_shape, n_images=n_steps, radius=radius, max_lon=np.pi)

# model
kwargs = {"pb":"pb", "obj_rmin":0.0, "data_rmin":1.05}
P, D, obj_mask, data_mask = tomograpy.models.thomson(data, obj, u=.5, **kwargs)
# projection
t = time.time()
data[:] = (P * obj.ravel()).reshape(data.shape)
#data[np.isnan(data)] = 0
print("projection time : " + str(time.time() - t))

# Plot the projection along the orbit
fig = plt.figure()
ax = fig.add_subplot(111)
em_exp = 0.5
xy_values = radius*np.tan((np.linspace(0, im_pshape, num=6) - (im_pshape/2.)))
# im = ax.imshow(data[::-1, ::-1, 0]**em_exp, extent=[xy_values.min(),
im = ax.imshow((np.rot90(data[:, :, 0], 1))**em_exp,
               extent=[xy_values.min(),
               xy_values.max(), xy_values.min(), xy_values.max()])

# im=ax.pcolormesh(data[:,:,0]**0.1,cmap=cmap)

ax.set_aspect("equal")
# ax.set_title(date_obs + "  " + "AIA filter: " + aia_fl)
# im = plt.imshow(data[:,:,0]**0.1, origin='lower')
plt.xlabel('Solar Radii')
plt.ylabel('Solar Radii')
clb = plt.colorbar(im)
#clb.set_label('(Emiss)^(%s)' % em_exp)

for i in range(n_steps):
    im.set_array(np.rot90(data[:, :, i], 1)**em_exp)
#  plt.annotate('Long: %s * $\pi$ ' % longit[i], xy=(0.25, y.min()+.25), color='white')
    plt.pause(0.05)
