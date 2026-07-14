#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:55:44 2020

@author: sparenti
"""

from astropy.io import fits
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage
import SPUtils as spu
import SPUtils_WL as spuwl
import SPUtils_AIA as spuaia
import scipy.interpolate as interp
import wavelets
import os
from skimage.transform import warp_polar
import matplotlib.animation as animation
import matplotlib.cm as mplcm



# To check if a dir/file exists

# def set_dir(im_dir):PLUTO-KCor_20h_NAX512_PSHAPE13.0_VOX112_LON202.1_WET_nolong
#   if not os.path.exists(im_dir+""):  
#   else:


dirf = '/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/'

# Open two data files
def open2imgs(dirf1, file1, dirf2, file2):
    print(dirf2, file2)
# Get info image1 (PLUTO)
    dataf1, h1 = rd_fitsf(dirf1, file1)
# Get info image2 (data, AIA, LASCO....)

    dataf2, h2 = rd_fitsf(dirf2, file2)
    
    print("===============================================")
    if h2["INSTRUME"] == 'AIA_2':
        print("AIA data,")
#        dataf1 *= h2["EXPTIME"]
#        print('Image 1 Header units: {}'.format(h1['D_UNIT']))
#        h1["D_UNIT"] = "Dn"
#        print('New Image1 header units: {}'.format(h1["D_UNIT"]))
        dataf2, h2 = spuaia.get_aia_prep(dataf2, h2)
    elif h2["INSTRUME"] == 'LASCO':
        dataf2, h2 = spuwl.mk_LASCO_rot_wrhead(dataf2, h2)
        print('prep LASCO data')
    elif "K-Cor" in h2["INSTRUME"]:
        print('prep K-Cor data')
        dataf2, h2 = spuwl.mk_KCor_prep(dataf2, h2, mask=True)
    elif h2["INSTRUME"] == "EUI":
        print(h2["INSTRUME"] +' data')
    else:
        raise Exception("Missing the info for this instrument")
    print("===============================================")

    # Add check on need for data binning (image2)
    dataf2_bin, h2 = image_rebin(dataf2, (h1["NAXIS1"],
                                          h1["NAXIS2"]), h=h2)

    if dataf2_bin.shape != dataf1.shape:
        raise Exception("The two images have different shape")   
    for i in range(2):
        if h1['NAXIS' + str(i + 1)] != h2['NAXIS' + str(i + 1)]:
            raise Exception("The two images have different NAXIS.")
#        elif h1['CRPIX' + str(i + 1)] != h2['CRPIX' + str(i + 1)]:
#            raise Exception("The two images have different CRPIX.")
    #    elif h1['CDELT' + str(i + 1)] != h2['CDELT' + str(i + 1)]:
    #        raise Exception("The two images have different CRDEL.")
        elif h1['CRVAL' + str(i + 1)] != h2['CRVAL' + str(i + 1)]:
            raise Exception("The two images have different CRVAL.")

    return dataf1, h1, dataf2_bin, h2


# Open and prep an image file
def open1img(dirf, file):
    image, h = rd_fitsf(dirf, file)
    if h["INSTRUME"] == 'AIA_2':
        image, h = spuaia.get_aia_prep(image, h)
        print('AIA data prep')
        
    elif h["INSTRUME"] == 'LASCO':
        print(h['filename'], h['DATE_OBS'])
        image, h = spuwl.mk_LASCO_rot_wrhead(image, h)
        print('LASCO data prep')
    elif (h["INSTRUME"].find('COSMO K-Cor') != - 1):
        print('K-Cor data prep')
        image, h = spuwl.mk_KCor_prep(image, h, mask=True)
    elif h["INSTRUME"] == 'METIS':
        print('METIS data prep')
#       image, h = spuwl.mk_metis_prep(image, h, mask=True)
    else :
        print("===============================================")    
    print("The file has been opened")
    print("===============================================")
    return image, h


def set_path(h, simu, lat=None,):
    """
    get a path to look for depending on the input header
    """ 
    path = (r'/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/'
            r'VSWMC_DATA/SIMU_UV/IMG_DIR/')
    if h["INSTRUME"] == 'PLUTO-LASCO':
        new_path = os.path.join(path, 'VICTOR_WET3D/AIA_20H/SYN_WL/LASCO_pB/')
    elif h["INSTRUME"] == 'PLUTO-K-Cor':
        new_path = os.path.join(path, 'VICTOR_WET3D/AIA_20H/SYN_WL/K-Cor/')
    elif h["INSTRUME"] == 'MULTIVP-LASCO':
        new_path = os.path.join(path, 'LASCO_MULTIVP_20181106/')
    elif h["INSTRUME"] == 'LASCO':
        new_path = '/mnt/c/Users/sparenti/Documents/LAVORO/FITS/LASCO/'
    return new_path


def plot2images(image1, image2, h1):
    """
    :param file1:
    :param file2:
    :param image1: LASCO data
    :param image2: simulation
    :return:
    """
    if (h1["INSTRUME"].find('LASCO') != - 1):
        band = 'pB'
        vmin = -1  # -1 #-3
        vmax = 2.5  # 1
        cmap = mplcm.plasma
    fig, ax = plt.subplots(1, 2, num=0, clear=True)
    ax[0].imshow(np.log10(image1), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax[1].imshow(np.log10(image2), cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    plt.show()


def plot_imgs_ratio(image1, image2, h=None, shift=None, sv=None, title='AIA-PLUTOW3',
                    nrse=True, pxbad=None):    
    """
     plot the ratio of two images and eventually shift one of them
     shift = (1,1)
     h = h1 from PLUTO  
     Assumes Image1 = PLUTO
     Ratio: AIA / PLUTO
     pxbad: to select only part of the images for the error nrse format[y1, Y2, x1, x2]
    """    
    if shift:
        image1=np.roll(image1, shift, axis=(0, 1))
        print("Immage 1 shifted")
    if h:
        lon = np.degrees(h["LON"])# - np.pi)
        lat = np.degrees(h["LAT"])
        if h["WAVELNTH"] == 'Ne':
            vmin = -3
            vmax = 2
        else:
            vmin =  0.001
#            vmax = 20
            vmax = 1.5 # 2.3
    else:
        lon = 0.
        lat = 0.
        
    ratio = np.zeros_like(image1)    
    good = (image2 > 1e-10) & (image1 != 0)
    ratio[good] = image2[good]/image1[good]

    # Plot
    rsun = get_rsun(h)
    ysun = rsun[:, int(h['NAXIS1']/2)]
    xsun = rsun[int(h['NAXIS2']/2), :]
    fig = plt.figure(num=0, clear=True)
    plt.imshow(ratio, vmin=vmin, vmax=vmax, origin='lower', cmap='coolwarm',
               extent=[-xsun.max(), xsun.max(), -ysun.max(), ysun.max()])
    fig.suptitle(title)
    plt.xlabel('R$_\odot$')
    plt.ylabel('R$_\odot$')
    
    #  fig.suptitle("{}/PLUTO".format(h["INSTRUME"]))
    clb = plt.colorbar()
    clb.set_label('Ratio')
    plt.show()

    if nrse is True:
        norm_mse(image1, image2, pxbad=pxbad)
    if sv:
        title = 'AIA_Model2'
        fln = dirf + 'ratio_'+ title + '_LAT{0:.1f}'.format(lat)+ \
            '_LON{0:.1f}'.format(lon) +'eps' #+ ".png"
        fig.savefig(fln,  format='eps', bbox_inches="tight")
        print('Saved file' + fln)
    return ratio[good]



def norm_mse(image1, image2, pxbad=None):
    """
    Calculate the mean squared  error of two images normalizedto the sigma
    image2 is the reference image (DATA)
    pxbad: pixels to be excluded for the calculation. format[y1, Y2, x1, x2]
    nmse = sum((image1 - image2)^2/image2)
    """
    if pxbad is not None:
        image1[pxbad[0]:pxbad[1], pxbad[2]:pxbad[3]] = 0
        image2[pxbad[0]:pxbad[1], pxbad[2]:pxbad[3]] = 0
    good = (image2 > 1e-10) & (image1 != 0)
#    image2[~good] = 1e5
#    image1[~good] = 1e5
    diff2 = (image1[good] - image2[good])**2 / image2[good]
    diff = (image1 - image2)**2/ image2
    err = diff2.sum()
    plt.figure(num=1, clear=True)
    plt.imshow(np.log10(diff), origin='lower')
    print("=================================================")
    print("The normalized Mean Squared Error {0:.1f}".format(err))
    print("=================================================")
    return
   


# Plot the ratio of 1D profiles build with 'plot1prof'
# DATA format = [xx, image_cut]
# files: 
def plot_prof_ratio(data1, h1, data2, h2, sv=None, plotr=None):
   dirsav ='/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/VICTOR_WET3D/AIA_20H/MULTI_PROF/'
   if len(data1[:,0]) != len(data2[:,0]):
       raise Exception("data and data2 should have the same size", 
                       len(data1[:,0]),  len(data2[:,0]))

   label = h2['inst'] + '/' + h1['inst'] + '@ ' + str(h1["theta"]) +'deg'
   titgen='ratio' + '_' + h2["INST"].strip() + '_'+ h2["INST"].strip() + \
           '_theta'+ str(h1["theta"]) 
   good = data1[:,1] > 1e-9 
   ratio = data2[good,1]/data1[good,1]                # the good have benn selected already
   if plotr is True: 
       fig, ax = plt.subplots(1,1, num=0, clear=True)
       ax.plot(data1[good,0], ratio, label=label)
       ax.set_ylabel("Ratio")
       ax.set_ylim(-5, 20)

   if sv is True:
       fig.savefig(dirsav +  titgen + ".png")
       print('Saved filename...',  titgen + ".png")
   return ratio, data1[good,0]



# Resize image1 over image2 dimensions
def images_resize(image1, image2, interpolation = cv2.INTER_LINEAR):
#    new_image1 = cv2.resize(image1, image2.shape(), CDELT21, 
#                           CDELT22, interpolation = 'inter_linear')    
    shape2 = image2.shape
    new_image1 = cv2.resize(image1, shape2,
                            interpolation=interpolation)
    return new_image1



''' Rebin an imput image to a new smaller diratio = spugen.plot_imgs_ratio(img1, img2, h=h1, title=tit1+tit2, sv=sav)mention image2 - 
    new_shape = (nx,ny)
     h = header of image
'''        
def image_rebin(image, new_shape, h=None, pxbin=None):
    if h:
        if pxbin == None:
            pxbin = h['NAXIS1'] / new_shape[0]
        if pxbin != 1:
            for i in range(2):
                if ('NAXIS' + str(i + 1)) in h:
                    h['NAXIS' + str(i + 1)] = int(h['NAXIS' + str(i + 1)] / pxbin)
                if ("CDELT" + str(i + 1)) in h:
                    h["CDELT" + str(i + 1)] = h['CDELT' + str(i + 1)] * pxbin
                if ("CRVAL" + str(i + 1)) in h:
                    h["CRVAL" + str(i + 1)] = h['CRVAL' + str(i + 1)] / pxbin
                if ('CRPIX' + str(i + 1)) in h:
                    h['CRPIX' + str(i + 1)] = (h['CRPIX' + str(i + 1)] / pxbin)
    else: 
        h=[0]
    shape = (new_shape[0], image.shape[0] // new_shape[0],
             new_shape[1], image.shape[1] // new_shape[1])
    print(shape)
    print("===============================================")
    if new_shape[0] != image.shape[0]:
        print('The image has been rebinned to the given size')
    else:
        print("No need to bin, the two images have already the same size")
    print("===============================================")
    return image.reshape(shape).mean(-1).mean(1), h


# Matrix for changing a 3D referenze frame to Carrington
# x1, y1, z1 points of array to be rorated.
# The LOS is along z1 
# Theta from yc-zc  (degrees)
# Phi from xc in the  plane xc-yc (degrees)
# (x1,y1,z1) -> (x2,y2,z2)->(xc,yc,zc)
def coords2carr(x1, y1, z1, theta, phi):
    theta = np.radians(theta)
    phi = np.radians(phi)
    # Rotaton around x1 (theta, x1=x2)
    z2 = (z1 * np.cos(theta)) - (y1 * np.sin(theta))
    y2 = (z1* np.sin(theta)) + (y1 * np.cos(theta))
    x2 = x1
    
    # Rotation around y2 (phi, y2 = y3)
    x3 = (x2 * np.cos(phi)) + (z2 * np.sin(phi)) 
    z3 = - (x2 * np.sin(phi)) + (z2 * np.cos(phi))   
    y3 = y2
    xc = z3
    yc = x3
    zc = y3
    return xc, yc, zc


def cart2polar(image, h=None, center=None, output_shape=None):
    """
    convert the image fromcartesian to polar
    if the image header is not passed, you shpuld pass 'center'
    center=(x_c,y_c)
    The polar angle 0 is at the equator and 90 at North of the image 
    Note: CRPIX -1 as in the fits the first element is 1. 
    """
    if h is None and center is None:
        raise ValueError("Either h or center should be passed")
    if h is not None:
        polar = warp_polar(image, center=(h["CRPIX1"]-1 , h["CRPIX2"]-1), 
                           output_shape=output_shape)
    else:
        polar = warp_polar(image, center=center, output_shape=output_shape)
    return polar

#def get_r_from_h(h):
    
    


# Apply a Gaussian smooting to an image
def img_gauss_flt(image, kernel, sigma, cmap = None, save = None):
    dirf = '/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/IMG_BLUR/'
    image = cv2.GaussianBlur(image, kernel, sigma, cv2.BORDER_DEFAULT)
    fig = plt.figure(num=1)
    plt.suptitle("Gaussian filter with sigma = {}".format(sigma))
    plt.imshow(image**0.5, origin='lower', cmap=cmap)
    plt.show()
    if save:
        fig.savefig(dirf + "AIAS193" + 
                    "_Gauss_sigma{}" .format(sigma) +".png",
                    bbox_inches="tight")
#    plt.close()      
    return image

# Apply a Median smooting to an image
def img_median_flt(image, sz, cmap = None, save = None): 
    dirf = '/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/IMG_BLUR/'
    image = scipy.ndimage.median_filter(image, size=sz)
    fig = plt.figure(num=1)
    plt.suptitle("Median filter with size = {}".format(sz))
    plt.imshow(image**0.5, origin='lower', cmap=cmap)
    plt.show()
    if save:
        fig.savefig(dirf + "AIA193" + 
                    "_Median_sz{}" .format(sz) +".png",
                    bbox_inches="tight")
    return image


def im_med_spike(image, sz, level, cmap = None, save = None):
    """  
    Apply a median filter to an image and replaces the bright areas with the 
    average of the image 
    """     
    dirf = '/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/IMG_BLUR/'
    im_med = img_median_flt(image, sz)
 #   im_med = image
    immean, imstd = cv2.meanStdDev(im_med[30:100, 30:100])
   # level = 1.1
    im_med[im_med > (level * float(imstd))] = float(immean)
#    plt.hist(im_med)
    fig = plt.figure(num=1, clear=True)
    plt.suptitle("Median filter = {}".format(sz) + ", spike removal lev = {}".format(level))
    plt.imshow(im_med**0.5, origin='lower', cmap=cmap, )
    plt.show()
    if save:
        fig.savefig(dirf + "AIA193" + 
                    "_Median_sz{}" .format(sz) +"_spklev{}std".format(level) +
                    ".png", bbox_inches="tight") 
    return im_med



def get_rsun(h):
    """
    Get the distanze in R_sun of an image
    h: header
    Note: CRPIX+1 as in the fits the first element is 1. 
    """
    x, y = np.ogrid[:h["NAXIS1"], :h["NAXIS2"]]
    x = (x - h["CRPIX1"]+1) * h["CDELT1"] + h["CRVAL1"]
    y = (y - h["CRPIX2"]+1) * h["CDELT2"] + h["CRVAL2"]

    if h.get("radius") != None:
        r_y = h["radius"] * np.tan(y)
        r_x = h["radius"] * np.tan(x)
    else:
        r_x = h["D"] * np.tan(x)
        r_y = h["D"] * np.tan(y)

    dist_rsun = np.sqrt((r_x - h["CRVAL1"])**2 +(r_y - h["CRVAL2"])**2)
    return dist_rsun


def get_1xy(h):
    
    """ 
    Get 1D x, yin solar radii 
    Note: CRPIX + 1 as in the fits the first element is 1. 
    """
    x = np.arange(h["NAXIS1"])
    y = np.arange(h["NAXIS2"])
    x = (x - h["CRPIX1"] + 1) * h["CDELT1"] + h["CRVAL1"]
    y = (y - h["CRPIX2"] + 1) * h["CDELT2"] + h["CRVAL2"]
    if "radius"  not in h:
#    if "radius" in h.keys() is False:
        h["radius"] = h["D"]
    r_x = h["radius"] * np.tan(x)
    r_y = h["radius"] * np.tan(y)
#    print(h["radius"])
    return r_x, r_y


def get_cart2spher(shape, CDELT1, CRPIX1, CRVAL1):
    """
    given the input (3D)shape, it creates a 3D cartesian box and converts to spherical
    Note: CRPIX -1 as in the fits the first element is 1. 
    """
    x, y, z = np.indices(shape).astype(np.float64)
    x -= CRPIX1 - 0.5  # x = (x - CRPIX1) * CDELT1 + CRVAL1
    x *= CDELT1
    x += CRVAL1  #        this_par[:,:,0:10] = 20+= CRVAL1
    y -= CRPIX1 - 0.5 # y = (y - CRPIX1) * CDELT1 + CRVAL1
    y *= CDELT1
    y += CRVAL1
    
    z -= CRPIX1 - 0.5 # z = (z - CRPIX1) * CDELT1 + CRVAL1
    z *= CDELT1
    z += CRVAL1
    
    # convert x,y,z to spherical for interpolating WL into x,y,z
    Rsph = np.sqrt(x*x + y*y + z*z)
    p = np.sqrt(x*x + y*y)
    theta = np.arctan2(p, z)
    phi = np.arctan2(y, x)                    # between -pi and pi
    return Rsph, theta, phi


def get_img_Lapgrad(image, sv=None, out=None):
      savdir ='/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/IMG_BLUR/'
      image64=image.astype(np.float64)
      laplacian = cv2.Laplacian(image64,cv2.CV_64F)
      plt.figure(num=0)
      plt.imshow(laplacian, vmin=-5e-3, vmax=1e-2)
      plt.suptitle('Laplacian filter')
      if sv is True:
        if out is None:
            tit = 'Laplacian_sharp.png'
        else:
            tit = out + '_lapl.png'
        plt.savefig(savdir + tit)
        print('Saved file...', tit)
      return laplacian


def get_img_sharp(image, weight=[1, 1, 0], sv=None, out=None):
    savdir = (r'/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/'
              r'CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/IMG_BLUR/')
    img_srp = wavelets.enhance(image, weights=[1, 0.5, 0.5, 0])
    plt.figure(num=1)
    plt.imshow(img_srp, vmin=-.3, vmax=1)
    plt.suptitle('Wavelets sharphing')
    if sv is True:
        if out is None:
            tit = 'Wavelets_sharp.png'
        else:
            tit = out + '_wvlt.png'
            plt.savefig(savdir + tit)
        print('Saved file...', tit)
    return img_srp


 

def apply_poles_mask(header, t_mask):
    """
    Note: CRPIX +-1 as in the fits the first element is 1. 
    """
    x, y = np.ogrid[:header["NAXIS1"], :header["NAXIS2"]]
    x = (x - header["CRPIX1"] + 1) * header["CDELT1"] + header["CRVAL1"]
    y = (y - header["CRPIX2"] + 1) * header["CDELT2"] + header["CRVAL2"]
    if not "radius" in header.keys():
        header["radius"] = header["D"] 
    r_x = header["radius"] * np.tan(x)
    r_y = header["radius"] * np.tan(y) 
    r = np.sqrt((r_x - header["CRVAL1"])**2 +(r_y - header["CRVAL2"])**2)
    theta = np.arctan2(x,y) + np.pi
    limits_max = np.pi/2 + np.radians(t_mask)
    limits_min = np.pi/2 - np.radians(t_mask)
    theta_mask1 = np.logical_and(np.logical_and(theta <= limits_max, theta >= limits_min), r > 1) #1.05) 
    theta_mask2 = np.logical_and(np.logical_and(theta <= np.pi+limits_max, theta >= np.pi + limits_min), r > 1) 
    print("//////////////////////////////////////////")
    print("Dark mask applied theta {}".format(t_mask)) 
    print("//////////////////////////////////////////")
    return theta_mask1, theta_mask2


def mk_animation(fig, data, im, ax, n_ima, annot=None):
    """
    given a cube of images, update the plot to create a movie
    data: 3D cube
    im: plotted image (im=plt.imshow())
    n_ima: n_im
    """
    def init():
        if annot is not None:
#            annot_1 = ax.text(0.7,0.8, 'LON: {:6.2}'.format(annot[0]))
         tit = ax.get_title()
        return tit,
            
    def updatefig(i):
        display = np.copy(data[:, :, i])
        display[display <= 0] = 0.01
#        im.set_array(np.log10(display))
        im.set_array(display**0.25)
        if annot[0] is not None:
            if i == 0:
                ax.set_title(' Perihelion LONGITUDE: {:.1f}$^\circ$'.format(annot[i]))
            else:
                if annot[i] >  360:         #(2*np.pi):
                    labs = annot[i]/360.       #% (2*np.pi)
                else:
                    labs =  annot[i]
 #              ax.set_title(' LONGITUDE: {:.1f}$^\circ$'.format(annot[i]))
                ax.set_title(' LONGITUDE: {:.1f}$^\circ$'.format(labs))
        return im, fig, ax
    ani = animation.FuncAnimation(fig, updatefig, init_func=init, frames=n_ima, 
                                  interval=1000, blit=False, repeat=False)
    return ani, fig


# Get data and header of a fits file
def rd_fitsf(dirfits, file):
     dataf = fits.getdata(dirfits + file, header=True)
#     dataf = fits.open(dirfits + file)
     
     print("Opened file: ", file)
     return dataf


# Save a fits file. Extra hader should be a list
def wr_fits(mydir, data, filename ='test', h=None, extra_h=None, overwrite=True):
    new_hdul = fits.HDUList()
    print(mydir)
    if data.ndim > 2:
        for i in range(data.shape[2]):
            new_hdul.append(fits.ImageHDU(data[:, :, i],
                                          header=fits.Header(data.header[i])))
    else:
        if (hasattr(data, 'header') is False) and h == None:
            new_hdul.append(fits.ImageHDU(data[:], header=None))
        elif (hasattr(data, 'header') is False) and h != None:
            new_hdul.append(fits.ImageHDU(data[:], header=fits.Header(h)))
#        elif (hasattr(data, 'header') is True) and h !=None):
#            raise    
        else:     
            new_hdul.append(fits.ImageHDU(data[:],
                                          header=fits.Header(data.header)))
    
# adds any other arrays passed as optional arguments as ewtra HDUs
    if extra_h is not None:
        if type(extra_h) is dict:
            extra_h = [extra_h]
        print(type(extra_h))
        for a in extra_h:
            new_hdul.append(fits.ImageHDU(header=fits.Header(a)))
#            new_hdul1=fits.HDUList(hdus=[new_hdul, fits.Header(extra_h)])
    new_hdul.writeto(mydir + filename + '.fits', overwrite=True, 
                     output_verify = 'ignore')
    new_hdul.close()
    print("Written the file: ", filename + '.fits')
    
 
    