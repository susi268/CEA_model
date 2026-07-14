import sys,os
import numpy as np
import matplotlib.pyplot as plt
import astropy.visualization as av
from mpl_toolkits.axes_grid1 import make_axes_locatable
from subprocess import call
import scipy.interpolate
import pyPLUTO as pp
import PlutoUtils as pu
import sunpy.cm
#from evtk.hl import gridToVTK
import itertools
import time

class synth_AIA(object):
    """ Computes a synthetic SDO/AIA image for a given filter"""
    def __init__(self,dir,filename=None,filter=None,dtype="dbl",elevation=0,azimuth=180,box_size=2,res=63.5): #200):
        self.data_dir=dir
        self.dtype=dtype
        # Outputs in the database
        if(filename==None):
            try:
                self.outnum=pu.find_last_output(self.dtype,self.data_dir)
            except:
                print("No {0} found in {1}".format(self.dtype,self.data_dir))
                raise 
        else:
            out=filename.split('.')
            self.outnum=int(out[1])
            self.dtype=out[2]

        # Angles / point of view
        self.elevation=elevation # Should be B0 for an actual AIA image
        self.azimuth=azimuth # Should be 180 for an actual AIA image
        self.box_size=box_size
        self.res=res
        
        if(filter==None):
            self.filter="193"
        else:
            self.filter=filter

        self.filter_cmap_dict={"94":plt.get_cmap("sdoaia94"),
                              "131":plt.get_cmap("sdoaia131"),
                              "171":plt.get_cmap("sdoaia171"),
                              "193":plt.get_cmap("sdoaia193"),                              
                              "211":plt.get_cmap("sdoaia211"),
                              "304":plt.get_cmap("sdoaia304"),                              
                              "335":plt.get_cmap("sdoaia335")}

        # Useful constants (cgs) 
        self.kb=1.3807e-16
        self.mp=1.6726e-24

        # PLUTO normalization
        self.v0=4.367e7
        self.n0=1e8
        self.l0=6.9570e10

    def load_pluto_data(self,GEOM="Cartesian"):
        """load PLUTO data and create interpolators for density and temperature"""
        if not os.path.exists(self.data_dir+"KernelEm.pkl"):
            t0=time.perf_counter()
            D=pp.pload(self.outnum,w_dir=self.data_dir,datatype=self.dtype)
            t1=time.perf_counter()
            pu.get_caseparams(D)
            
            try:
                self.v0=D.UNIT_VELOCITY
                self.n0=D.UNIT_DENSITY/self.mp
                self.l0=D.UNIT_LENGTH
            except:
                pass

            D.T=self.mp*self.v0**2/(2*self.kb)*D.prs/D.rho
            self.iTemp=scipy.interpolate.RegularGridInterpolator((D.x1,D.x2,D.x3),D.T,bounds_error=False,fill_value=None)
            self.iDens=scipy.interpolate.RegularGridInterpolator((D.x1,D.x2,D.x3),D.rho,bounds_error=False,fill_value=None)
            
            self.r,self.theta,self.phi=np.mgrid[0.0:2.5:(self.res+1)*1j,0:np.pi:(self.res+1)*1j,0:2*np.pi:(2*self.res+1)*1j]
            
            if (GEOM=="Cartesian"):
                self.Tsph=self.iTemp((self.r*np.sin(self.theta)*np.cos(self.phi),self.r*np.sin(self.theta)*np.sin(self.phi),self.r*np.cos(self.theta)))
                self.Nsph=self.iDens((self.r*np.sin(self.theta)*np.cos(self.phi),self.r*np.sin(self.theta)*np.sin(self.phi),self.r*np.cos(self.theta)))*self.n0
            elif (GEOM=="Spherical"):
                self.Tsph=self.iTemp((self.r,self.theta,self.phi))
                self.Nsph=self.iDens((self.r,self.theta,self.phi))*self.n0
            pu.save_elements([self.r,self.theta,self.phi,self.Tsph,self.Nsph],self.data_dir+"KernelEm.pkl")
            t1=time.perf_counter()            
            print("Kernel created in {0} seconds".format(t1-t0))
        else:
            [self.r,self.theta,self.phi,self.Tsph,self.Nsph]=pu.read_elements(5,self.data_dir+"KernelEm.pkl")
            print("Density and temperatures grids have been loaded from KernelEm")
        
    def interp_AIA_response(self):
        """ Load AIA response from file and create interpolator"""
        rsp=aia_response('/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SOUMITRA_EX/aia_temp_resp.dat')
        wvl=[s[1:] for s in rsp.wvl]
        try:
            idx=wvl.index(self.filter)
            self.iRsp=scipy.interpolate.interp1d(rsp.Temperature,rsp.dnresp[idx,:],bounds_error=False,fill_value=0.0)
            self.cmap=self.filter_cmap_dict[self.filter]
        except KeyError:
            print("Filter response is not currently in the database")

    def integrate_los(self):
        """Integrate the volume emission along the line of sight"""
        self.xReg,self.yReg,self.zReg=np.mgrid[-self.box_size:self.box_size:(2*self.res+1)*1j,
                                               -self.box_size:self.box_size:(2*self.res+1)*1j,
                                               -self.box_size:self.box_size:(2*self.res+1)*1j]

        self.xNew=np.zeros(self.xReg.shape)
        self.yNew=np.zeros(self.yReg.shape)
        self.zNew=np.zeros(self.zReg.shape)

        t0=time.perf_counter()

        # Create the rotation matrices
        el=-self.elevation*np.pi/180
        az=self.azimuth*np.pi/180
        Mt=np.matrix([[np.cos(el),0,np.sin(el)],[0,1,0],[-np.sin(el),0,np.cos(el)]])
        #Mt=np.matrix([[1,0,0],[0,np.cos(el),-np.sin(el)],[0,np.sin(el),np.cos(el)]])
        Mp=np.matrix([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])

        cReg=np.array([self.xReg.flatten(),self.yReg.flatten(),self.zReg.flatten()])
        cRot=np.matmul(Mt,cReg)
        cRot=np.matmul(Mp,cRot)
        
        cNew=np.array(cRot)
        self.xNew[:,:,:]=cNew[0,:].reshape(self.xNew.shape[0],self.xNew.shape[1],self.xNew.shape[2])
        self.yNew[:,:,:]=cNew[1,:].reshape(self.xNew.shape[0],self.xNew.shape[1],self.xNew.shape[2])
        self.zNew[:,:,:]=cNew[2,:].reshape(self.xNew.shape[0],self.xNew.shape[1],self.xNew.shape[2])
        
        Rsph=np.sqrt(self.xNew**2+self.yNew**2+self.zNew**2)
        Rcyl=np.sqrt(self.xNew**2+self.yNew**2)
        Theta=np.pi/2.-np.arctan(self.zNew/Rcyl)
        Phi=(np.arctan2(self.yNew,self.xNew))%(2*np.pi)

        t1=time.perf_counter()
        #print("step 1 {}s".format(t1-t0))
        
        m2=(self.r > 1.0)
        self.volume_response=np.zeros(np.shape(self.r))
        self.volume_response[m2]=self.iRsp((np.log10(self.Tsph)))[m2]*self.Nsph[m2]**2
        t2=time.perf_counter()
        #print("step 2 {}s".format(t2-t1))
        # New interpolation
        iVolRsp=scipy.interpolate.RegularGridInterpolator((self.r[:,0,0],self.theta[0,:,0],self.phi[0,0,:]),self.volume_response,bounds_error=False,fill_value=0.0)

        t3=time.perf_counter()
        #print("step 3 {}s".format(t3-t2))
        self.NewRspCart=iVolRsp((Rsph,Theta,Phi))
        
        t4=time.perf_counter()
        #print("step 4 {}s".format(t4-t3))


        # Line of sight integration
        dl=(self.xReg[-1,:,:]-self.xReg[0,:,:])/(self.xReg.shape[0])*self.l0        
        m1=(self.xReg < 0) & (np.sqrt(self.yReg**2+self.zReg**2) < 1) # Remove what's behind the Sun
        self.NewRspCart[m1]=0.0
        self.Im=self.NewRspCart.sum(axis=0)*dl

        t5=time.perf_counter()
        #print("step 5 {}s".format(t5-t4))

    def cmp_synoptic_map(self,save=False):

        mask=(self.r > 1.0)        
        self.volume_resp_syn=np.zeros(np.shape(self.r))
        self.volume_resp_syn[mask]=self.iRsp((np.log10(self.Tsph)))[mask]*self.Nsph[mask]**2
        iVolRsp=scipy.interpolate.RegularGridInterpolator((self.r[:,0,0],self.theta[0,:,0],self.phi[0,0,:]),self.volume_response,bounds_error=False,fill_value=0.0)


        rcyl=np.linspace(0,2,self.res)
        z=np.linspace(-1.,1.,self.res)

        self.Rcyl,self.Z,self.Phi=np.meshgrid(rcyl,z,self.phi[0,0,:],indexing='ij')
        Rsph=np.sqrt(self.Rcyl**2+self.Z**2)
        Theta=np.pi/2.-np.arctan(self.Z/self.Rcyl)
        self.NewRspSyn=iVolRsp((Rsph,Theta,self.Phi))

        # Line of sight integration
        dl=(self.Rcyl[1]-self.Rcyl[0])*self.l0
        self.ImSyn=self.NewRspSyn.sum(axis=0)*dl
        
        if save:
            pu.save_elements([z,self.phi[0,0,:],self.ImSyn],self.data_dir+"Syn"+self.filter+".pkl")
        
    def vtk_volume_emission(self):
        """Save the volumetric emission into a vtk file for analysis"""
        rmin=self.r[0,0,0]
        rmax=self.r[-1,0,0]
        thetamin=self.thetaEdge[0,0,0]
        thetamax=self.thetaEdge[0,-1,0]
        phimin=self.phiEdge[0,0,0]
        phimax=self.phiEdge[0,0,-1]
        gridToVTK(self.data_dir+"VolEM{:.1f}".format(self.filter)+"A",
                  np.linspace(rmin,rmax,self.res),np.linspace(thetamin,thetamax,self.res),np.linspace(phimin,phimax,2*self.res),cellData={"VolEmission":self.volume_response})
   
    def plot_AIA_synthetic_image(self,save=True,vmax=None,vmin=None,im_dir=None,cbar=True):
        fig=plt.figure()
        ax=fig.add_subplot(111)
        if(vmax==None):
            self.vmax=self.Im.max()
            self.vmin=self.Im.min()
        else:
            self.vmax=vmax
            self.vmin=vmin

        im=ax.pcolormesh(self.yReg[0,:,0],self.zReg[0,0,:],self.Im.T,cmap=self.cmap,vmin=self.vmin,vmax=self.vmax)#,norm=av.mpl_normalize.ImageNormalize(self.Im,stretch=av.LogStretch()))
        if cbar:
            divider=make_axes_locatable(ax)
            cax=divider.append_axes("right",size="10%",pad=0.1)
            cbar=plt.colorbar(im,cax=cax)
            ax.set_aspect("equal")

        ax.set_aspect("equal")
        ax.set_title("{} Angstroms".format(self.filter))
        if(save):
            if(im_dir==None):
                im_dir=self.data_dir+"AIAMovie/"
            call("mkdir -p "+im_dir,shell=True)
            fig.savefig(im_dir+"AIASynth{}".format(self.filter)+"A_"+"{:.2f}".format(self.elevation)+"_"+"{:.2f}".format(self.azimuth)+".png",bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
    def produce_image(self,save=True,dir=None,vmax=None,cbar=True):
        self.load_pluto_data()
        self.interp_AIA_response()
        self.integrate_ios()
        self.plot_AIA_synthetic_image(save=save,im_dir=dir,vmax=vmax,cbar=cbar)

class aia_response(object):
    def __init__(self,filename):
        self.filename=filename
        self.read_response()

    def read_response(self):        
        Phot_response=[]
        DN_response=[]

        f=open(self.filename,'r')
        #### Photons response ####
        tmp=f.readline()
        wvl=[]
        for band in range(0,7):
            Temperature=[]
            head=f.readline()
            wvl.append((head.split())[1])
            h1=(head.split())[6:]
            l=[float(x) for x in h1]
            for ll in l:
                Temperature.append(ll)

            for ii in range(0,17):
                tmp=f.readline()
                l=[float(x) for x in tmp.split()]
                for ll in l:
                    Temperature.append(ll)

            resp=[]
            for ii in range(0,21):
                tmp=f.readline()
                l=[float(x) for x in tmp.split()]
                for ll in l:
                    resp.append(ll)
    
            Phot_response.append(resp)
            tmp=f.readline()

        phresp=np.array(Phot_response)

        ### DN resp ####
        tmp=f.readline()
        wvl=[]
        for band in range(0,7):
            Temperature=[]
            head=f.readline()
            wvl.append((head.split())[1])
            h1=(head.split())[6:]
            l=[float(x) for x in h1]
            for ll in l:
                Temperature.append(ll)

            for ii in range(0,17):
                tmp=f.readline()
                l=[float(x) for x in tmp.split()]
                for ll in l:
                    Temperature.append(ll)

            resp=[]
            for ii in range(0,21):
                tmp=f.readline()
                l=[float(x) for x in tmp.split()]
                for ll in l:
                    resp.append(ll)
    
            DN_response.append(resp)
            tmp=f.readline()

        dnresp=np.array(DN_response)
        f.close()

        self.Temperature=np.array(Temperature)
        self.wvl=wvl
        self.phresp=phresp
        self.dnresp=dnresp
