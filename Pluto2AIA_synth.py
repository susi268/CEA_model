#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:04:23 2019

@author: sparenti

Reads the output from .dbl or vtk (Pluto model) and projects the AIA 
emissivities along a 2pi orbit.
Steps:
    Extract the quantity (image_quant) to project 
    


"""

import time
import numpy as np
import tomograpy
import matplotlib.pyplot as plt
import SPUtils as spu
import SPUtils_gen as spugen
import pyPLUTO as pp
import scipy.interpolate as interp
import SPUtils_AIA as spuaia
import tomograpy.siddon
import os.path

# sys.path.insert(1, '')

# Various initial conditions
tt = time.time()
save = 1  # save the images in png
savef = 1  # save one image in fits file
psfile = 'aia_psf193.fits'
psf = 1    # Apply the AIA PSF

#wdir = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/VICTOR_EX/SImulation_2019_11_06/'
wdir = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/VICTOR_EX/WET_3D_HLL_PSP_E1/'
mdir = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/VICTOR_WET3D/'
aia_dir = '/mnt/c/Data/SP_FITS/SDO/2018/'
aia_f = 'aia.lev1.193A_2018-11-06T12_00_52.84Z.image_lev1.fits'

# Read the output from the model (.dbl)
num = 156
plfile = "data.0156.vtk"
d = pp.pload(num, w_dir=wdir, datatype='vtk')
mod_sun = spu.synth_ini(wdir, filename=plfile)
d = spu.load_pluto_logTN(d, mod_sun, GEOM="Spherical")

#  Given an AIA filt[30:100, 30:100]er as imput extract the emissivities and
#  interpolate them on the T and N output from PLUTO
aia_flt = "193"
aia_obj = spu.aia_obj()
aia_obj.read_aia_emiss(aia_flt)

#os.path.isfile('iaia_emis_' + num + ) 

print(time.time() - tt)
iaia_em = aia_obj.emis_interpol(d).astype(np.float32)
print('Emissivity interpolation', time.time() - tt)

max_lon = np.max(d.x3)          # phi max
min_lon = 0.
date_obs = "2018-11-06T12:00:00Z"
n_steps = 1                     # observer orbit steps
longit = np.linspace(min_lon, max_lon, n_steps)/(2.*np.pi)  # to be annotated on the imege

# defining the cube that will be projected
vox = 1.5
shape = int(d.n1_tot*vox)      # number of voxels along each axis
pshape = 1.27 * 5.          # from AIA in Rsun pysical shape of object (ie width of cube)


object_header = tomograpy.siddon.centered_cubic_map_header(pshape, shape)
#im_pshape = tomograpy.siddon.fov(object_header, radius)  # FOV in rad

im_shape = 128             # number of image pixels in X & Y (NAXIS)
pxbin = 32                    # bin that will be applied to AIA (4096/bin = imspahe)
image_header = spuaia.image_header_from_aia(aia_dir, aia_f, pxbin, dtype=np.float64)
im_pshape = image_header["NAXIS1"] * image_header["CDELT1"]    # AIA FOV in rad
            
#image_header = tomograpy.siddon.centered_image_header(im_pshape, im_shape)

obj = tomograpy.simu.object_from_header(object_header, fill=0, dtype=np.float32)
print(time.time() - tt)

# Create x,y, z to be converted to spherical
x, y, z = np.indices(obj.shape).astype(np.float32)
CDELT1 = object_header["CDELT1"]
CRPIX1 = object_header["CRPIX1"]
CRVAL1 = object_header["CRVAL1"]
x -= CRPIX1  # x = (x - CRPIX1) * CDELT1 + CRVAL1
x *= CDELT1
x += CRVAL1
y -= CRPIX1  # y = (y - CRPIX1) * CDELT1 + CRVAL1
y *= CDELT1
y += CRVAL1
z -= CRPIX1  # z = (z - CRPIX1) * CDELT1 + CRVAL1
z *= CDELT1
z += CRVAL1

pxbin = 128

# convert x,y,z to spherical for interpolating iaia_em into x,y,z
Rsph = np.sqrt(x*x + y*y + z*z)
p = np.sqrt(x*x + y*y)
theta = np.arctan2(p, z)
del p
del z
phi = (np.arctan2(y, x) + 2*np.pi) % (2*np.pi)
del x
del y
print("Spherical coordinates", time.time() - tt)

x3_1 = ((np.arange(1)+1.)*d.dx3[0])+np.max(d.x3)
x3_2 = ((np.arange(1)-1.)*d.dx3[0])+np.min(d.x3)
x3 = np.concatenate((d.x3, x3_1))
x3 = np.concatenate((x3_2, x3))
iaia2 = np.concatenate((iaia_em, iaia_em[:, :, 0:1]), axis=2)
iaia2 = np.concatenate((iaia_em[:, :, iaia_em.shape[2] - 1:iaia_em.shape[2]], iaia2), axis=2)

# Multiply the voxel size by m^3 to have Dn/s
print('===============================================')
print('Emissivity units: {}'.format(aia_obj.emiss_units))
iaia2 = iaia2 * 6.957e8 #* object_header['CDELT1'] 
image_header['D_UNITS'] = 'Dn/s'
print('New emissivity units: Dn/s')
print("===============================================")

# Prepare to interpolate iaia2
interpolator = interp.RegularGridInterpolator((d.x1, d.x2, x3),
                                              iaia2,
                                              bounds_error=False,
                                              fill_value=0)
print('Begin Interpolator', time.time() - tt)
# interpolate.5
obj[:] = interpolator((Rsph, theta, phi))
obj[obj < 0] = 0
del Rsph
del theta
del phi
print('End Interpolation', time.time() - tt)


# Prepare for the projection along an orbit
data = tomograpy.simu.circular_trajectory_data(n_images=n_steps, 
                                min_lon=image_header["LON"], **image_header, dtype=np.float32)


# projection
print('Projection start', time.time() - tt)
tomograpy.siddon.projector(data, obj, obstacle="sun")
data[data < 0] = 0
#data *= 7e8             # to get unit of Dn as the integration code use unit r_sun
print('Projection end', time.time() - tt)

# Convolve with the AIA psf
if (psf):
    data = spuaia.conv_psf_aia(data, psfile)
    
# Get the same orientation than the AIA image
data = np.moveaxis(data, 0, 1)

# Apply the AIA mask
data_mask = spuaia.apply_aia_mask(image_header, 1.5)
data[~data_mask] = 0.

# Plot the projection along the orbit
cmap = aia_obj.filter_cmap_dict[aia_flt]
fig = plt.figure(num=0, clear=True)
ax = fig.add_subplot(111)
em_exp = 0.1
xy_values = image_header["radius"]*np.tan((np.linspace(0, im_pshape, num=6) - (im_pshape/2.)))
long = np.degrees(image_header["LON"] - np.pi)
# im = ax.imshow(data[::-1, ::-1, 0]**em_exp, extent=[xy_values.min(),
#im = ax.imshow((np.rot90(data[:, :, 0], 1))**em_exp,
im = ax.imshow(data[:, :, 0]**em_exp, origin='lower',
               extent=[xy_values.min(),
               xy_values.max(), xy_values.min(), xy_values.max()], cmap=cmap)

ax.set_aspect("equal")
ax.set_title(date_obs + "  " + "AIA filter: " + aia_flt + '\n LON; {:.2f},'.format(long) 
            + ' PSF correction: ' + str(psf))
# im = plt.imshow(data[:,:,0]**0.1, origin='lower')
plt.xlabel('Solar Radii')
plt.ylabel('Solar Radii')
clb = plt.colorbar(im)
clb.set_label('(Emiss)^(%s)' % em_exp)
filesav = str("AIAS{}".format(aia_obj.aia_filter) + "_VOX{}".format(im_shape)
          + "_PSHAPE{}".format(pshape)+"_LON{:.1f}".format(long) + 
          '_PSF' + str(psf) + "_WET")

#for i in range(n_steps):
#    im.set_array(np.rot90(data[:, :, i], 1)**em_exp)
#    im.set_array(data[:, :, i])
#  plt.annotate('Long: %s * $\pi$ ' % longit[i], xy=(0.25, y.min()+.25), color='white')
#    plt.pause(0.05)
if save:
    fig.savefig(mdir + filesav + ".png".format(i), bbox_inches="tight")

# spu.encode_aia(mdir, 'toto.avi')

if(save):
    plt.close()
else:
    plt.show()

print('Total time', time.time() - tt)

if (savef):
    spugen.wr_fits(mdir, data, extra_h=object_header, filename=filesav)

# ffmpeg -framerate 25 -i AIASynth171_%3d.png -vcodec libx264 -preset slow -crf 17 -y toto.avi
