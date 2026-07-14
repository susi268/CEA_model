#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:04:23 2019

@author: sparenti

Reads the output from .dbl or vtk (Pluto model) and projects the AIA 
emissivities along a 2pi orbit.

input: pluto_long, CRTLT_0.
        image_header['LON'] should be the CRTLN of the wanted observer

This version uses a different interpolation function: map_coordinates

"""

import time
import numpy as np
import tomograpy
import matplotlib.pyplot as plt
import SPUtils as spu
import SPUtils_gen as spugen
#import scipy.interpolate as interp
import SPUtils_AIA as spuaia
import tomograpy.siddon
import os.path
import scipy.ndimage
import SPUtils_mod as spumod

# Various initial conditions
source = 'PLUTO'
tt = time.time()
save = 0                # save the images in png
savef = 0              # save one image in fits file
psfile = 'aia_psf193.fits'
psf = 0                 # Apply the AIA PSF using psfile
pluto_long =  np.pi      # phi = 0 for Pluto in the CAR map for 6/11/2018
mask_fov = 1             # apply a mask
n_steps = 1                     # observer orbit steps
aia_flt = "211"

path = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/'
#path = '/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/CEA_SOLO_SIMU/'
#mdirs = [os.path.join(path, 'VICTOR_20210521_SOLO_PER1-2/PER2_SIMU/')]
#mdirs = [os.path.join(path, 'VICTOR_WET3D_RHO3/SIMU_156_12H/'
#                      r'AIA_193_12h_LASCO15h/AIA_12h_LASCO15h_PSF0/')]
#mdirs = [os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_035_RHO1/AIA_035/AIA_35_12h_LASCO15h/')]
mdirs = [os.path.join(path, 'VICTOR_20191106_RHO1-2/VICTOR_087_RHO2/AIA_20191106_087/AIA_87_20h_LASCO15h/')]
#aia_dir = '/mnt/c/Data/SP_FITS/SDO/2018/'
aia_dir = '/mnt/c/sparenti/Documents/LAVORO/FITS/FITS_SDO/'
aia_f = 'aia.lev1.211A_2018-11-06T15 00 33.62Z.image_lev1.fits'
#aia_dir = '/mnt/c/Users/sparenti/Documents/LAVORO/FITS/SOLO_FITS/FSI/'
#aia_f = 'solo_L1_eui-fsi174-image_20200618T040052316_V01.fits'
#dirtest='/mnt/c/Users/Susanna Parenti/Documents/LAVORO/CMEs/FLORIANR_CMEs/FL_RES/'

models = ['PLUTO'] #, 'MULTIVP']
nums =[87, None]
plfiles = ["data.0087.vtk"]
pluto_longs = [np.pi]             #phi = 0 for Pluto in the CAR map for 6/11/2018
CRTLN_0 = [np.radians(215.4)]   # CRTLN of the meridian for 6/11/2018 @20h (phi_0 Pluto) simu 87
#CRTLN_0 = [np.radians(220.01)]   # CRTLN of the meridian for 6/11/2018 @12h (phi_0 Pluto) simu 156 & 35 

index = models.index(source)
if index == -1:
    raise ValueError('Must specifiy the data source')     

# Read the output from the model
plfile = plfiles[index]
CRTLN_0 = CRTLN_0[index]
pluto_long = pluto_longs[index]
d = spumod.mod_obj(plfile, source, nums[index])

#  Given an AIA filter as imput extract the emissivities and
#  interpolate them on the T and N output from PLUTO

aia_obj = spu.aia_obj()
aia_obj.read_aia_emiss(aia_flt)

print(time.time() - tt)
iaia_em = aia_obj.emis_interpol(d).astype(np.float32)
print('Emissivity interpolation', time.time() - tt)

# defining the cube that will be projected
vox = 1 #3.6  #2.5
shape = int(d.n1_tot*vox)      # number of voxels along each axis
pshape = 7.        # from AIA in Rsun pysical shape of object (ie width of cube)

object_header = tomograpy.siddon.centered_cubic_map_header(pshape, shape)
#im_pshape = tomograpy.siddon.fov(object_header, radius)  # FOV in rad

#max_lon = np.pi * 2
im_shape = 128             # number of image pixels in X & Y (NAXIS)
pxbin = 32.                   # bin that will be applied to AIA (4096/bin = imspahe)
image_header = spuaia.image_header_from_aia(aia_dir, aia_f, pxbin, dtype=np.float64)
im_pshape = image_header["NAXIS1"] * image_header["CDELT1"]    # AIA FOV in rad

# Update img header
image_header["MODELFL"] = plfile
image_header["CRTLN_0"] = CRTLN_0
image_header["PSF_COR"] = psf


obj = tomograpy.simu.object_from_header(object_header, fill=0, dtype=np.float32)
print('Time to start change of coords')
print(time.time() - tt)

# object coords from cartesian to spherical, to be interpolated in the model coords
Rsph, theta, p = spugen.get_cart2spher(obj.shape, obj.header["CDELT1"],
                                       obj.header["CRPIX1"], obj.header["CRVAL1"])

# Phi in Pluto reference frame for the 6/11/2018 @20H
#phi = (np.arctan2(y, x) + 2*np.pi - (CRTLN_0 - pluto_long))  % (2*np.pi)
phi = (p + 2*np.pi - (CRTLN_0 - pluto_long)) % (2*np.pi)

print("Spherical coordinates", time.time() - tt)

x3_1 = ((np.arange(1, dtype=np.float32)+1.)*d.dx3)+np.max(d.x3)
x3_2 = ((np.arange(1, dtype=np.float32)-1.)*d.dx3)+np.min(d.x3)
x3 = np.concatenate((d.x3, x3_1))
x3 = np.concatenate((x3_2, x3))
iaia2 = np.concatenate((iaia_em, iaia_em[:, :, 0:1]), axis=2)
iaia2 = np.concatenate((iaia_em[:, :, iaia_em.shape[2] - 1:iaia_em.shape[2]], iaia2), axis=2)
#iaia2[110:114, 48, 0:50] = 15

# Multiply the voxel size by m^3 to have Dn/s
print('===================================================')
print('Emissivity units: {}'.format(aia_obj.emiss_units))
iaia2 = iaia2 * 6.957e8 #  the integration is done in Rsun
image_header['D_UNIT'] = 'Dn/s'
print('New emissivity units: Dn/s')
print("===================================================")

# Prepare to interpolate iaia2 in the model coords. 
# Interpolate first the cartesian coods transformed in spherical 
print('Begin Interpolator', time.time() - tt)
rc = np.interp(Rsph, d.x1, np.linspace(0, d.x1.shape[0]-1, d.x1.shape[0], dtype=np.float32))
tc = np.interp(theta, d.x2, np.linspace(0, d.x2.shape[0]-1, d.x2.shape[0], dtype=np.float32))
pc = np.interp(phi, x3, np.linspace(0, x3.shape[0]-1, x3.shape[0], dtype=np.float32))
coords = np.stack((rc.ravel(), tc.ravel(), pc.ravel()), axis=0).astype(np.float32)
obj[:] = scipy.ndimage.map_coordinates(iaia2, coords, order=1, mode='constant', cval=0).reshape(obj.shape)
#obj[Rsph < 1] = 0
#obj[:] = 0
#obj[np.logical_and(Rsph <= 1.15,Rsph > 1.1)] = 10
#bj[Rsph > 1.15] = 1

del Rsph
del theta
del phi
print('End Interpolation', time.time() - tt)

#image_header["LON"] = # np.pi #- np.radians(160)

# Prepare for the projection along an orbit
data = tomograpy.simu.circular_trajectory_data(n_images=n_steps, 
                                min_lon=image_header["LON"], **image_header, 
                                dtype=np.float32)
#                                max_lon=max_lon, dtype=np.float32)

# projection
print('Projection start', time.time() - tt)
tomograpy.siddon.projector(data, obj, obstacle="sun")
data[data < 0] = 0
print('Projection end', time.time() - tt)

# Convolve with the AIA psf
if (psf) == 1:
    data = spuaia.conv_psf_aia(data, psfile)

# Get the same orientation than the AIA image
data = np.moveaxis(data, 0, 1)

# Apply the AIA mask
if mask_fov == 1:
    data_mask = spuaia.apply_aia_mask(image_header, 1.5)
    data[~data_mask] = 0.

# Apply pole mask (+-5 degrees from the poels)
#m1, m2 = spugen.apply_poles_mask(image_header, 6)
#data[m1] = 0
#data[m2] = 0

# Plot the projection along the orbit
cmap = aia_obj.filter_cmap_dict[aia_flt]
fig = plt.figure(num=0, clear=True)
ax = fig.add_subplot(111)
longs = np.degrees([h["LON"] for h in data.header])
em_exp = 0.25
xy_values = image_header["radius"]*np.tan((np.linspace(0, im_pshape, num=6) - (im_pshape/2.)))
#long = np.degrees(image_header["LON"]) # - np.pi)
im = ax.imshow(data[:, :, 0]**em_exp, origin='lower',
               extent=[xy_values.min(),
               xy_values.max(), xy_values.min(), xy_values.max()], cmap=cmap)

ax.set_aspect("equal")
titleplt = image_header["INSTRUME"] + " " + image_header["DETECTOR"] + aia_flt + '\n {}'.format(image_header["DATE_OBS"])
            #+ '\n LON; {:.2f},'.format(long) + ' PSF correction: ' + str(psf)
#ax.set_title(titleplt)
plt.xlabel('R$_\odot$')
plt.ylabel('R$_\odot$')
#clb = plt.colorbar(im)
#clb.set_label('(Emiss)$^(%s)$' % em_exp)

if n_steps > 1:
    ani, fig = spugen.mk_animation(fig, data, im, ax, n_steps, annot=longs)

#filesav = str("AIA{}".format(aia_obj.aia_filter) + "_20h_VOX{}".format(im_shape)
#          + "_PSHAPE{}".format(pshape)+"_LON{:.1f}".format(long) + 
#          '_PSF' + str(psf) + "_WET_VICTOR")

filesav = str("{}".format(image_header['INSTRUME'])  +
              "_NAX{}".format(image_header["NAXIS1"]) +
              "_PSHAPE{:.1f}_".format(pshape) + 'VOX{}'.format(shape) + '_PSF' + str(psf) +
              "_LON{:.1f}".format(np.degrees(image_header['LON'])) + '__lmax20_f'+ str(nums[index]))
filesav='test'
#for i in range(n_steps):
#    im.set_array(data[:, :, i]**em_exp)
#  plt.annotate('Long: %s * $\pi$ ' % longit[i], xy=(0.25, y.min()+.
#    if save:
#        fig.savefig(mdirs[index] + filesav + ".png".format(i), bbox_inches="tight")

# spu.encode_aia(mdir, 'toto.avi')

if(save):
    if n_steps > 1:
        ani.save(mdirs[index] + filesav + ".mp4", writer='ffmpeg', fps=10, 
                 dpi=100, metadata={'title': filesav}) 
    else:    
        fig.savefig(mdirs[index] + filesav + ".png", bbox_inches="tight")
#    plt.close()
else:
    plt.show()

print('Total time', time.time() - tt)

if (savef):
#    spugen.wr_fits(dirtest, data, extra_h=object_header, filename=filesav)
    spugen.wr_fits(mdirs[index], data, filename=filesav)

print(filesav)

# ffmpeg -framerate 25 -i AIASynth171_%3d.png -vcodec libx264 -preset slow -crf 17 -y toto.avi
