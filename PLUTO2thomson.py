
"""
Calculate a pB synthetic image from an input model.

Inputs:
    model file .vtk or other (also to be changed in SPUt)
    data file (wlf) to get header info: .fts
    Note: in SPUtils.PLUTO_wr_header, SPUtils_WL.mk_LASCO_rot_wrhead,  image_header_from_WL
    set the radius keyword.
"""

#!/usr/bin/env python
import time
import numpy as np
import tomograpy
import matplotlib.pyplot as plt
import SPUtils_gen as spugen
import SPUtils_WL as spuwl
import SPUtils_AIA as spuaia
import scipy.ndimage
#import matplotlib.cm as mplcm
import SPUtils_mod as spumod
import SPUtils as spu
import os 

import matplotlib as mtp
#print(mtp.get_backend())
#mtp.use('Gtk4Agg')                       # plot an interactive window
#mtp.use("TkAgg")

# Various initial conditions
source = 'PLUTO'
tt = time.time()
save = 1                       # save the images in png
savef = 1                      # save one image in fits file
psf = 0                          # Apply the PSF
masks = {'LASCO': 2.2, 'K-Cor': 1.1,  'WL': 1.5, 'METIS': None}    #Rsun
#masks = {'LASCO': 1.01, 'K-Cor': 1.1, 'METIS': 3.2}
dtype = np.float64               # ls np.flo20210215_180130_kcor_l2_extavg.fts'at32 ou np.float64
pxbin = 1                       # bin that will be applied to WL (imsaphe)
n_steps = 1                # observer orbit steps
#max_lon = 0 #np.pi*2               # max_long for projecting along an orbit

#path = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/'
#path = r'C:/Users/sparenti/Documents/LAVORO/GLOBAL_MODEL/CEA_SOLO_SIMU/VICTOR_20210521_SOLO_PER1-2/PER2_SIMU/'
path ='/home/sparenti/WORK/GLOBAL_MODEL/WP_3D_Metis_20210115_lmax85_Murteira/'
#wldir ='/home/sparenti/FITS/METIS/Metis_L2_pB_15jan2021/'
wldir = '/home/sparenti/FITS/LASCO/'

#wlf ='solo_L2_metis-vl-pb_20210115T003001_V01.fits'            # data file to get the header info'
wlf ='23829899pB.fts'

models = ['PLUTO', 'MULTIVP', 'GIBS', 'LAMY', 'FLOR']
nums =['0020', None, None, None, None]
plfiles = ["data.0020.dbl", "annotpt_201811051200_R001_cube.txt", 'Gibson99.pdf',
           'sing_prof_LASCOC2_Ne_Radial50_f23731980Nes.fits', 'rthinner_z25.0050.hdf5']
#pluto_longs = [np.pi, 0, 0, 0, 0]             #phi = 0 for Pluto in the CAR map for 6/11/2018
pluto_longs = [0, 0, 0, 0, 0]
#CRTLN_0s = [np.radians(218.2), 0, 0, 0, 0]   # CRTLN of the meridian for 6/11/2018 @20h (phi_0 Pluto) simu 86
#CRTLN_0s = [np.radians(220.01), 0, 0, 0]   # CRTLN of the meridian for 6/11/2018 @12h (phi_0 Pluto) simu 156 & 35
CRTLN_0s = [0, 0, 0, 0, 0]

#mdirs = [os.path.join(path,'/VICTOR_20191106_RHO1-2/VICTOR_035_RHO1/LASCO_035/LASCO_RHO1_35_12h/'),
#dirs = [os.path.join(path,'VICTOR_20191106_RHO1-2/VICTOR_087_RHO2/LASCO_20191106_087/LASCO_RHO2_87_20h/'),
#mdirs = [os.path.join(path, 'VICTOR_WET3D/AIA_20H/SYN_WL/K-Cor/'),   #PLUTO
#dirs = [os.path.join(path, 'TEST_CODE/'),
mdirs = [os.path.join(path,'WP_20210115_MU_OUT/'),
         os.path.join(path, 'MULTIVP_RES/LASCO_MULTIVP_20181106/'),     #MULTI-VP
         os.path.join(path, 'MOD_GIBSON99/'),
         os.path.join(path, 'VICTOR_WET3D/AIA_20H/SYN_WL/LASCO_Ne/'), 
         '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/CMEs/FLORIANR_CMEs/FL_RES/']
index = models.index(source)
if index == -1:
    raise ValueError('Must specifiy the data source')     

# Read the output from the model
plfile = plfiles[index]
CRTLN_0 = CRTLN_0s[index]
pluto_long = pluto_longs[index]
d = spumod.mod_obj(plfile, source, nums[index])

# defining the cube that will be projected
vox = 0.5                # 2.5 to get 558
shape = int(d.n1_tot*vox)      # number of voxels along each axis
pshape = 50  #50 LASCO                    # Rsun physical shape of object (ie width of heliospheric cube)

# Inizialize the objects
object_header = tomograpy.siddon.centered_cubic_map_header(pshape, shape)
obj = tomograpy.centered_cubic_map(pshape, shape, fill=0, dtype=dtype) 
obj.header = object_header

# object coords from cartesian to spherical, to be interpolated in the model coords
Rsph, theta, p = spugen.get_cart2spher(obj.shape, obj.header["CDELT1"],
                                       obj.header["CRPIX1"], obj.header["CRVAL1"])

# converts carrington longitude to pluto longitude (between 0 and 2pi )
phi = (p + 2*np.pi - (CRTLN_0 - pluto_long))  % (2*np.pi)

#phi_carr = np.arctan2(y, x)  # cube is in carrington coordinates
print("Spherical coordinates", time.time() - tt)

# Make the juction between first and last element of the cube
x3_1 = ((np.arange(1, dtype=np.float32)+1.)*d.dx3)+np.max(d.x3)
x3_2 = ((np.arange(1, dtype=np.float32)-1.)*d.dx3)+np.min(d.x3)
x3 = np.concatenate((d.x3, x3_1))
x3 = np.concatenate((x3_2, x3))
logN2 = np.concatenate((d.logN, d.logN[:, :, 0:1]), axis=2)
logN2 = np.concatenate((d.logN[:, :, d.logN.shape[2] - 1:d.logN.shape[2]], logN2), axis=2)
#logN2[30:60, 96//2, 48] =25  #pluto
#logN2[:, 90//2:, 107]=25   #multivp

# interpolate N in the cube
rc = np.interp(Rsph, d.x1, np.linspace(0, d.x1.shape[0]-1, d.x1.shape[0], dtype=dtype))
tc = np.interp(theta, d.x2, np.linspace(0, d.x2.shape[0]-1, d.x2.shape[0], dtype=dtype))
pc = np.interp(phi, x3, np.linspace(0, x3.shape[0]-1, x3.shape[0], dtype=dtype))
coords = np.stack((rc.ravel(), tc.ravel(), pc.ravel()), axis=0).astype(dtype)

# Fill the obj
obj[:] = scipy.ndimage.map_coordinates(10**logN2, coords, order=1, mode='constant', cval=0).reshape(obj.shape)
#obj[Rsph < 2] = 10
#obj[Rsph > 2] = 1
#obj[Rsph < 0.1] = 0
#obj[Rsph < 1.5] = 1e10
print('End Interpolation', time.time() - tt)
print("obj centered")

# Prepare for the projection along an orbit (data)
# Read the observation fits file and get the header information
if source != 'FLOR':
    image_header = spuwl.image_header_from_WL(wldir, wlf, pxbin, source, dtype=dtype)
    im_pshape = image_header["NAXIS1"] * image_header["CDELT1"]    # WL FOV in rad
else:
    im_pshape = np.arctan2(pshape/2, 210.6)    # 210 Rsun assumed distance  # WL FOV in rad
    image_header = spu.PLUTO_wr_header([], source, im_pshape=im_pshape, dtype=dtype)


#print("LON",np.degrees(image_header['LON']))
# Update img header
image_header["MODELFL"] = plfile
if n_steps > 1:
    max_lon = (image_header["LON"] + 2 * np.pi )   # % (2*np.pi)
else :
    max_lon = 2 * np.pi
data = tomograpy.simu.circular_trajectory_data(n_images=n_steps,
                                               min_lon=image_header["LON"],  max_lon=max_lon,
                                               **image_header, dtype=dtype)

# Thomson model
kwargs = {"pb":"pb", "obj_rmin":0.0, "data_rmin":1.05}
P, D, obj_mask, data_mask = tomograpy.models.thomson(data, obj, u=.5, **kwargs)

# projection
t = time.time()
data[:] = (P * obj.ravel()).reshape(data.shape)
#tomograpy.siddon.projector(data, obj, obstacle="sun")
data[np.isnan(data)] = 0
print('Min and max in the image', data.min(), data.max())
print("projection time : " + str(time.time() - t))

# Get the 1units of the pB (B 10e-10) and add Pluto file info
data = data * 1e-10

#data.header[0]["D_UNITS"] = '1e-10 '+ image_header['D_UNITS']
#print(data.header[0]["D_UNIT"])

# Get the same orientation of the WL image
data = np.moveaxis(data, 0, 1)

# Apply the WL mask
if image_header["INSTRUME"] in masks:
    this_mask = masks.get(image_header["INSTRUME"].split('-',1)[1])
    if this_mask is None:
        data_mask=spuwl.apply_METIS_mask(image_header)
    else:
        data_mask = spuaia.apply_aia_mask(image_header, this_mask)
#data[data_mask] = 1e-10

# Plot.
#fig = plt.figure(num=0, clear=True)
#ax = fig.add_subplot(111)
fig, ax = plt.subplots(1, 1, num=0, clear=True)
longs = np.degrees([h["LON"] for h in data.header]) % 360
xy_values = image_header["radius"]*np.tan((np.linspace(0, im_pshape, num=6) - (im_pshape/2.)))
title = "{}".format(image_header['INSTRUME'])+ "{} ".format(image_header['DETECTOR']) +\
             "{}".format(image_header["MODELFL"])
ax.set_title(title) #  "{}".format(image_header["PLUTOFL"]))

#import astropy.visualization
#interval = astropy.visualization.MinMaxInterval()
#stretch = astropy.visualization.LogStretch()
#transform = stretch + interval
#rsun = spugen.get_rsun(image_header)
rsun1 = plt.Circle((0, 0), radius=1., color='black')
ax.add_artist(rsun1)

display = np.copy(data[:, :, 0])
#if "colmap" in image_header.keys():
#    cmap = image_header["colmap"]
#cmap=mplcm.plasma
cmap='RdYlBu_r'
#im = plt.imshow(np.log10(display), origin='lower', extent=[xy_values.min(),
im = ax.imshow(np.log10(display), origin='lower', extent=[xy_values.min(),
                xy_values.max(),  xy_values.min(), xy_values.max()],
               cmap=cmap, vmin=-0.5, vmax=2)# vmin=-1, vmax=3)
plt.xlabel('R$\odot$')
plt.ylabel('R$\odot$')
#clb = plt.colorbar(im)
#clb.set_label('log(pB) [10^-10 Bʘ]')
plt.show()

if n_steps > 1:
    ani, fig = spugen.mk_animation(fig, data, im, ax, n_steps, annot=longs)

# Filename to save 
filedata = 'f' + image_header["MODELFL"].split('.', -1)[1]
#filedata = 'f' + image_header["MODELFL"].split('hdf5')[0]

if image_header['INSTRUME'].find("LASCO") != - 1:
    dum = image_header["DATE_OBS"].split('T')[0]
    date = dum.split("/")[-1]
else:
    dum = image_header["DATE_OBS"].split('T')[0]
    date = dum.split("-")[-1]
filesav = str("{}".format(image_header['INSTRUME']) + "_D{}".format(date) +
              "_NAX{}".format(image_header["NAXIS1"]) +
              "_PSHAPE{:.1f}_".format(pshape) + 'VOX{}'.format(shape) +
              "_LON{:.1f}".format(np.degrees(image_header['LON'])) + '_' + filedata  + 'M')
filesav='test_LASCO'

if (savef):
    spugen.wr_fits(mdirs[index], data, filename=filesav)

if save:
    if n_steps > 1:
        ani.save(mdirs[index] + filesav + ".mp4", writer='ffmpeg', fps=10, 
                 dpi=100, metadata={'title': filedata})
    else:    
        fig.savefig(mdirs[index] + filesav + ".png", bbox_inches="tight")
#g    plt.close()
print(filesav)
