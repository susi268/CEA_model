#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:33:07 2021

@author: sparenti
"""

filer='r_sing_prof_PLUTO-AIA_2_193_LON218.2_Radial15_f0087.fits'
data87, h87 = spugen.rd_fitsf(dirf1, 'PLUTO-AIA_2_NAX128_PSHAPE7.0_VOX810_PSF0_LON218.2_f87.fits')
datar, hr = spugen.rd_fitsf(dirf1, filer)
fig, ax = plt.subplots(1,1, num=0, clear=True)
rsun = spugen.get_rsun(h)
ysun = rsun[:, int(h['NAXIS1']/2)]
xsun = rsun[int(h['NAXIS2']/2), :]
cmap = aia_obj.filter_cmap_dict[aia_flt]
im = ax.imshow(data87**0.25, origin='lower', extent=[-xsun.max(), xsun.max(), 
                                                     -ysun.max(), ysun.max()],cmap=cmap, vmax=5, vmin=0.8)
ax.plot(datar[:,0],datar[:,1], color='c')
ax.set_xlabel("R$_\odot$")
ax.set_ylabel("R$_\odot$")
ax.set_title('Model 2')
#clb = plt.colorbar(im)
#clb.set_label('(Emiss)$^(%s)$' % em_exp)


dirsav = '/home/sparenti/WORK/GLOBAL_MODEL/PERI2_OUT/'
dirf1='/mnt/c/Users/Susanna Parenti/Documents/LAVORO/FITS/MLSO/'
file='20210222_174916_kcor_l2_extavg.fts'
filesav = '20210215_180130_kcor_l2_extavg.png'

fig, ax = plt.subplots(1,1, num=0, clear=True)
fontsize = 11
image, h = spugen.open1img(dirf1, file)
cmap='RdYlBu_r' 
date = '22'

m1, m2 = spugen.apply_poles_mask(h, 6)
rsun = spugen.get_rsun(h)
ysun = rsun[:, int(h['NAXIS1']/2)]
xsun = rsun[int(h['NAXIS2']/2), :]

image[m1] = 1e-10
image[m2] = 1e-10

image[image<0]=1e-10
im = ax.imshow(np.log10(image), cmap=cmap, vmin=-1, vmax=3, origin='lower',
               extent=[-xsun.max(), xsun.max(), ysun.max(), -ysun.max()])

ax.set_xlabel("R$_\odot$", fontsize=fontsize)
ax.set_ylabel("R$_\odot$", fontsize=fontsize)
#ax.set_title("MLSO K-Cor" + " " + h["DATE-OBS"], fontsize=fontsize)
ax.set_title("LASCO C2" + " " + h["DATE_OBS"], fontsize=fontsize)
clb = plt.colorbar(im)
clb.set_label('log(pB) [10^{-10} Bʘ]')
fig.show()
fig.savefig(dirsav +  filesav, bbox_inches="tight")


#Plot METIS
#-----------------------------
from astropy.visualization import ImageNormalize, PercentileInterval, AsinhStretch
from scipy.ndimage import rotate

norm = ImageNormalize(data,
    interval=PercentileInterval(99.5),
    stretch=AsinhStretch())

CROTA = h["CROTA"]
rotated_data = rotate(data,-CROTA, reshape=False,order=3)

# Plot
plt.figure(figsize=(8, 8))
plt.imshow(rotated_data, cmap="inferno", origin="lower", norm=norm)

plt.colorbar(label="Intensity")
plt.title("Solar Orbiter METIS LON="+ str(h["CRLN_OBS"]))
plt.xlabel("X Pixels")
plt.ylabel("Y Pixels")
plt.tight_layout()
plt.show()
