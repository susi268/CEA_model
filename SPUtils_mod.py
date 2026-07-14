#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:26:34 2020

Create the objet with the output of the models to be used by PLUTO2Thomson

@author: sparenti
"""
from PLUTO_soft import pyPLUTO as pp
import SPUtils as spu
#import SPUtils_gen as spugen
import io
import numpy   as np
#import matplotlib.pyplot as plt
import h5py

"""
arg[0]: file, arg[1]: model, arg[3]: PLUTO 'num' 
"""
class mod_obj():

    def __init__(self, *args):
        self.file = args[0]
        self.source = args[1]
        self.x1 = None              # in Solar Radii
        self.x2 = None              # in rad
        self.x3 = None              # in rad
        self.n1_tot = None
        self.dx3 = None
        if self.source == 'PLUTO':
            if len(args) == 2:
                raise ValueError('Must specifiy file number')
            self.PLUTO_read(self.file, args[2])
        elif self.source == 'MULTIVP':
            self.MULTVP_read(self.file)
        elif self.source == 'GIBS':
            self.Gibson99_str(rdim = 256)
        elif self.source == 'LAMY':
            self.Lamy_data()
        elif self.source == 'FLOR':
            self.Florian_data()
        else:
            raise ValueError('Source not defined')


    def PLUTO_read(self, plfile, num):
        """
        Read PLUTO output
        """     
        wdir = ('/home/sparenti/WORK/GLOBAL_MODEL/WP_3D_Metis_20210115_lmax85_Murteira/')
#        wdir = ('/home/sparenti/WORK/GLOBAL_MODEL/TEST_CODE/Model2_NOV21/')
        print("Looking the data in...", wdir)
#            r'/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/'
#                r'CEA_SOLO_SIMU/VICTOR_20210521_SOLO_PER1-2/PER1_SIMU/'   #WP_3D_S olO_E1_CT/'
#                r'WP_3D_CT_ADAPT_20200614_1200_real1_lmax20/WP_3D_CT_ADAPT_20200614_1200_real1_lmax20_mhd/')
#        obj = pp.pload(num, w_dir=wdir, datatype='vtk')
        obj = pp.pload(num, w_dir=wdir, datatype='dbl')
        mod_sun = spu.synth_ini(wdir, filename=plfile)
        obj = spu.load_pluto_logTN(obj, mod_sun, GEOM="Spherical")
        self.plfile = plfile
        self.x1 = obj.x1
        self.x2 = obj.x2   #obj.x3
        self.x3 = obj.x3
        self.n1_tot = obj.n1_tot
        self.dx3 = obj.dx3[1] - obj.dx3[0]
        self.logT = obj.logT
        self.logN = obj.logN
        print('=====================================================')
        print('PLUTO_read: dx1 is assumed constant (dx1 = dx1[0])')
        print('=====================================================')


    def MULTVP_read(self, mvfile):
        """
        Read MULTI-VP output
        """
        wdir = '/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/RUI_MULTIVP/'
        infile = open(wdir + mvfile, "r")
        lines = infile.readlines()
        
        ## read and check header
        nheader        = 7
        header         = lines[0:nheader]
        [nr,nthe,nphi] = np.loadtxt(io.StringIO(header[4]),dtype=np.dtype('i4','2i4'))
        npts           = nr*nthe*nphi        
        print(":: Reading file " + mvfile)

        ## check grid dims
        npts  = nr*nthe*nphi
        print(":: Detected grid dims:", nr, nthe, nphi, "; total nr of points:", npts)

        ## decode data
        nfilecols = 6 
        nlin      = int(npts/nfilecols) ## warning: assumes all lines are fully filled
        data      = lines[nheader:]
        
        ## find data array positions in data line list
        ##(n.b: there's an empty line b/w consecutive arrays in the file)
        ipos_r    = 0
        ipos_the  = ipos_r    + nr   + 1
        ipos_phi  = ipos_the  + nthe + 1
        ipos_vr   = ipos_phi  + nphi + 1
        ipos_vphi = ipos_vr   + nlin + 1 
        ipos_n    = ipos_vphi + nlin + 1
        ipos_te   = ipos_n    + nlin + 1
        ipos_br   = ipos_te   + nlin + 1
        ipos_bphi = ipos_br   + nlin + 1 
        ipos_fexp = ipos_bphi + nlin + 1 
        ipos_mask = ipos_fexp + nlin + 1 
        
        ## read arrays
        r    = np.loadtxt(data[ipos_r :ipos_r + nr])
        the  = np.loadtxt(data[ipos_the : ipos_the + nthe])
        phi  = np.loadtxt(data[ipos_phi : ipos_phi + nphi])
        t    = np.reshape( 
                 np.reshape( 
                   np.loadtxt(data[ipos_te : ipos_te + nlin]), 
                           (npts,)), 
                         (nr, nthe, nphi), order='F')
        
        n    = np.reshape( 
                 np.reshape( 
                   np.loadtxt(data[ipos_n  :ipos_n  +nlin]), 
                           (npts,) ), 
                         (nr, nthe, nphi), order='F')
        
        ## build Carrington frame coords
        # lat holds latitudes (degrees)            , while the holds co-latitude (rad) 
        # lon holds carrington longitudes (degrees), phi holds (-pi,pi) azimuths (rad)
        lat = np.round((-the + 0.5 * np.pi) / np.pi * 180, decimals=2)
        lon = np.round((phi + np.pi) / np.pi * 180, decimals=2)
        infile.close()
        # fills the objet
        self.x1 = r
#        self.x2 = np.radians(lat)
        self.x2 = the               # because the code uses co-lat
        self.x3 = np.radians(lon)
        self.n1_tot = nr
        self.dx3 = lon[1] - lon[0]
        self.logT = np.log10(t)
        self.logN = np.log10(n) + 6       # n in SI (m^-3)



    def Gibson99_str(self, rmin=1.1, rmax=4, rdim = 256):
        """
        Produces a spherical symmetric model of density from Gibson et al. 1999 
        """
        a = 77.1
        b = 31.4
        c = 0.954
        d = 8.30
        e = 0.550
        f = 4.63
        r = np.arange(1, rmax, (rmax - rmin)/rdim)
        n = (a * r**(-b) + c * r**(-d) + e * r**(-f)) * 1e8   #cm^-3
        nang = int(rdim / 2)
        nnn = np.tile(np.expand_dims(n, axis=(1, 2)), (nang, 2 * nang))
        
        self.x1 = r
        self.x2 = np.arange(0, np.pi, 2.* (rmax - rmin)/nang)  # theta
        #linspace
        self.x3 = np.arange(0, 2*np.pi, (rmax - rmin)/nang)    # phi
        self.n1_tot = rdim
        self.dx3 = (rmax - rmin)/rdim
        self.logT = np.zeros(rdim)
        self.logN = np.log10(nnn) + 6                       # m-3
#        self.logN[:] = 7 + 6.
#        plt.plot(r, 10**(self.logN[:, 69, 0] -6))
#        plt.yscale('log')
        print('==================================================')
        print('Gibson model extends from {} to {} Rsun' .format(rmin, rmax))
        print('==================================================')        


    def Lamy_data(self):
#        file = 'sing_prof_LASCOC2_Ne_Radial50_f23731980Nes.fits'
        dirf = '/mnt/c/Users/sparenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SIMU_UV/IMG_DIR/VICTOR_WET3D/AIA_20H/SYN_WL/LASCO_Ne/'
        cut, h = spugen.rd_fitsf(dirf, self.file)
        n = cut[:,1]
        good = n > 0
        n[good] = np.log10(n[good])
        rmax = cut[:,0].max()
        rmin = cut[:,0].min()
        nang = int(len(n) / 2)
#        plt.plot(cut[:,0], n)
#        print(nang, rmin, rmax, n.shape)
        nnn = np.tile(np.expand_dims(n, axis=(1, 2)), (nang, 2 * nang))
        nnn += 6          # m-3
#        nnn[:] = 13
        self.x1 = cut[:,0]
        self.x2 = np.linspace(0, np.pi, nang)  # theta
        self.x3 = np.linspace(0, 2*np.pi, 2*nang)    # phi
        self.n1_tot = nang *2
        self.dx3 = (rmax - rmin)/(nang *2)
        self.logT = np.zeros(nnn.shape)
        self.logN = nnn
        print('==================================================')
        print('LASCO Ne extends from {} to {} Rsun' .format(rmin, rmax))
        print('==================================================')        
       
        
    def Florian_data(self): 
        dirf = '/mnt/c/Users/sparenti/Documents/LAVORO/CMEs/FLORIANR_CMEs/FL_DATA/'
# id of the case
 #       case_id = 'z10'
# your file
        path_data = dirf + self.file #'{case_id}.0000.hdf5'

# Parameter that we want to extract from the hdf5
# x1,x2 and x3 are r,theta and phi
        param_list = ['x1','x2','x3','rho']
# parameters
        n1_tot = 331 # nb points for r 
        dphi = 0.01227148 # step for phi
        n0 = 3.079e8 #1/cm^3 

# Reading the data and putting in the dictionnary A
        A = {}
        with h5py.File(path_data,'r') as f:
#            print( 'Keys in f:', [keys for keys in f.keys()])         
            for param in param_list:
                A[param] = f.get(param)[()]
    
        self.x1 = A['x1']
        self.x2 = A['x2']    # theta
        self.x3 = A['x3']    # phi
        self.n1_tot = n1_tot
        self.dx3 = dphi
        self.logN = np.log10(A['rho'] * n0 * 1e6)
        print('==================================================')
        print('FLORIAN Ne extends from {} to {} Rsun' .format(self.x1.min(), 
              self.x1.max()))
        print('==================================================')        

# The density can now be accessed with A['rho'], or r with A['x1']

 #   def Cranmer99_ch(self)
 
 
 
 