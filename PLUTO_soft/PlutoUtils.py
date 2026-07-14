import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# from PLUTO_soft import pyPLUTO as pp
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset,BboxPatch,inset_axes
from matplotlib.transforms import Bbox, TransformedBbox
import pickle as pickle
import scipy.interpolate
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
if (hasattr(scipy.interpolate,'RegularGridInterpolator')):
    from scipy.interpolate import RegularGridInterpolator

def xyz2rtp(x,y,z):
    r=np.sqrt(x**2+y**2+z**2)
    t=np.arccos(z/r)
    p=np.arctan2(y,x)
    return (r,t,p)

def compute_conserved_quantities(D):
    if (not hasattr(D,'xx')):
        compute_grids(D)

    if (not hasattr(D,'Vr')):
        getcyl(D)

    if (not hasattr(D,'GAMMA')):
        get_caseparams(D)
        if (not hasattr(D,'GAMMA')):
            print (':: Warning, using hardcoded gamma = 1.05')
            D.GAMMA = 1.05

    D.S = D.prs/(D.rho**D.GAMMA)
    if (D.n3_tot != 1):
        D.K = D.rho*(D.Vr*D.Br + D.vx3*D.bx3)/(D.Br**2+D.bx3**2)
        D.E = (D.Vr**2 + D.vx3**2 - D.Vphi**2)/2.0 + D.GAMMA*D.rho**(D.GAMMA-1.0)*D.S/(D.GAMMA-1.0) + \
              D.Vphi*D.Bphi*D.K/D.rho - 1.0/D.Rsph
    else:
        D.K = D.rho*(D.Vr*D.Br + D.vx2*D.bx2)/(D.Br**2+D.bx2**2)
        D.E = (D.Vr**2 + D.vx2**2 - D.Vphi**2)/2.0 + D.GAMMA*D.rho**(D.GAMMA-1.0)*D.S/(D.GAMMA-1.0) + \
              D.Vphi*D.Bphi*D.K/D.rho - 1.0/D.Rsph
    D.O = (D.Vphi - D.K*D.Bphi/D.rho)/D.Rcyl
    D.L = D.Rcyl*(D.Vphi - D.Bphi/D.K)

def cmpOmegaEff(D,rotframe=True):
    D.Omstar=D.VphiStar_VESC*np.sqrt(2.)
    D.xx = np.tile( np.tile(D.x1,(D.n2,1))  , (D.n3,1,1)).T.squeeze()
    D.yy = np.tile( np.tile(D.x2,(D.n1,1)).T, (D.n3,1,1)).T.squeeze()
    D.zz = np.tile( np.tile(D.x3,(D.n2,1))  , (D.n1,1,1)).squeeze()

    D.Rcyl=np.sqrt(D.xx**2+D.yy**2)
    D.Rsph=np.sqrt(D.xx**2+D.yy**2+D.zz**2)
    D.cosp = D.xx/D.Rcyl ; D.sinp = D.yy/D.Rcyl
    D.cosp[np.where(D.Rcyl == 0.0)] = 0.0
    D.sinp[np.where(D.Rcyl == 0.0)] = 0.0

    for var in D.vars:
        exec("D.%s = D.%s.squeeze()" % (var,var))
    
    D.Vr   =  D.cosp * D.vx1 + D.sinp * D.vx2
    D.Vphi = -D.sinp * D.vx1 + D.cosp * D.vx2
    D.Br   =  D.cosp * D.bx1 + D.sinp * D.bx2
    D.Bphi = -D.sinp * D.bx1 + D.cosp * D.bx2

    if(rotframe):
        D.Vphi=D.Vphi+D.Rcyl*D.VphiStar_VESC*np.sqrt(2.)
        if (D.n3_tot != 1):
            D.vx1  = D.cosp * D.Vr - D.sinp * D.Vphi
            D.vx2  = D.sinp * D.Vr + D.cosp * D.Vphi
        else:
            D.vx3 = D.Vphi

    D.K = D.rho*(D.Vr*D.Br + D.vx3*D.bx3)/(D.Br**2+D.bx3**2)
    D.K[np.isnan(D.K)]=0
    D.O = (D.Vphi - D.K*D.Bphi/D.rho)/D.Rcyl
    D.O[np.isnan(D.O)]=0
    D.O=D.O/D.Omstar

def cmp_va_cs(D):
    if (not hasattr(D,'GAMMA')):
        get_caseparams(D)
        if (not hasattr(D,'GAMMA')):
            print(':: Warning, using hardcoded gamma = 1.05')
            D.GAMMA = 1.05
    D.cs  = np.sqrt(D.GAMMA*D.prs/D.rho)
    D.vv  = np.sqrt(D.vx1**2+D.vx2**2+D.vx3**2)
    D.Mach= D.vv/D.cs
    D.bb  = np.sqrt(D.bx1**2+D.bx2**2+D.bx3**2)
    D.va  = D.bb/np.sqrt(D.rho)
    D.Alf = D.vv/D.va
    D.beta = 2*D.prs/D.bb**2/4/np.pi
    if (D.n3_tot == 1):
        D.va_pol = np.sqrt(D.bx1**2+D.bx2**2)/np.sqrt(D.rho)
        D.fast_Alf = np.sqrt(2*(D.vx1**2+D.vx2**2)/(D.cs**2+D.va**2+np.sqrt((D.cs**2+D.va**2)**2-4*(D.cs**2)*(D.va_pol**2))))


def remove_rotframe(D):
    if (not hasattr(D,'xx')):
        compute_grids(D)
    if (not hasattr(D,'Vr')):
        getcyl(D)
    if (not hasattr(D,'R_ORBIT_PLANET')):
        get_caseparams(D)
    if (not hasattr(D,'rot_rem')):
        D.rot_rem = False
    if (not D.rot_rem):
        if (D.R_ORBIT_PLANET != 0.):
            print(('Case: '+D.wdir))
            print(('Rotating frame with rorb= '+str(D.R_ORBIT_PLANET)+' removed!'))
            D.Vphi = D.Vphi + D.Rcyl/(D.R_ORBIT_PLANET**1.5)
            #print 'Rotating frame with Om= '+str(D.VphiStar_VESC*sqrt(2.))+' removed!'
            #D.Vphi = D.Vphi + D.Rcyl*D.VphiStar_VESC*sqrt(2.)
            if (D.n3_tot != 1):
                D.vx1  = D.cosp * D.Vr - D.sinp * D.Vphi
                D.vx2  = D.sinp * D.Vr + D.cosp * D.Vphi
            else:
                D.vx3 = D.Vphi
        else:
            print(('Case: '+D.wdir))
            print(('Rotating frame with Om= '+str(D.VphiStar_VESC*np.sqrt(2.))+' removed!'))
            D.Vphi = D.Vphi + D.Rcyl*D.VphiStar_VESC*np.sqrt(2.)
            if (D.n3_tot != 1):
                D.vx1  = D.cosp * D.Vr - D.sinp * D.Vphi
                D.vx2  = D.sinp * D.Vr + D.cosp * D.Vphi
            else:
                D.vx3 = D.Vphi
        D.rot_rem=True
    # Update things that have already been computed
    if (hasattr(D,'Alf')):
        print('Recomputing the characteristic velocities as well...')
        cmp_va_cs(D)
    if (hasattr(D,'K')):
        print('Recomputing the conserved quantities as well...')
        compute_conserved_quantities(D)

def save_elements(list_e,name):
    f=open(name,'wb')
    for el in list_e:
        pickle.dump(el,f)
    f.close()

def read_elements(nb_el,name):
    f=open(name,'rb')
    list_el = []
    for el in range(nb_el):
        list_el.append(pickle.load(f))
    f.close()
    return list_el

def find_last_output(dtype,dir):
    list_file=[]
    ext=len(dtype)
    for fn in os.listdir(dir):
        if fn[-ext:]==dtype:
            list_file.append(fn)
    if(len(list_file)==0):
        raise IndexError("No file of the requested format found in this directory")
    list_file.sort()
    print(("Last output is {0}".format(list_file[-1])))
    return int(list_file[-1].split('.')[1])

def find_outlist(dir,dtype="dbl"):
    outlist=[]
    ext=len(dtype)
    for ff in os.listdir(dir):
        if ff[-ext:]==dtype:
            outlist.append(ff)
    outlist.sort()
    return outlist

                    
def get_caseparams(D):    
    fname = D.wdir+'/pluto.ini'
    if (os.path.exists(fname)):
        lines=[line.strip() for line in open(fname)]
        for ll in lines[lines.index('[Parameters]')+2:]:
            lll = ll.split()
            object.__setattr__(D,lll[0],float(lll[1]))
    else:
        print('Err: I did not find a pluto.ini file in the working dir')

    fname2 = D.wdir+'/definitions.h'
    if (os.path.exists(fname2)):
        lines=[line.strip() for line in open(fname2)]
        for ll in lines[lines.index('/* [Beg] user-defined constants (do not change this line) */')+2:]:
            lll = ll.split()
            try:
                object.__setattr__(D,lll[1],float(lll[2]))
            except:
                break            
    else:
        print('Err: I did not find a definitions.h file in the working dir')

#
# def get_userdefconst(D):
#     fname = D.wdir+'/definitions.h'
#     if (os.path.exists(fname)):
#         lines=[line.strip() for line in open(fname)]
#         for ll in lines[lines.index('/* [Beg] user-defined constants (do not change this line) */')+2:]:
#             lll = ll.split()
#             try:
#                 object.__setattr__(D,lll[1],float(lll[2]))
#             except:
#                 break
#     else:
#         print('Err: I did not find a definitions.h file in the working dir')



def get_userdefconst(D):
    fname = D.wdir+'/definitions.h'
    if (os.path.exists(fname)):
        lines=[line.strip() for line in open(fname)]
        for ll in lines[lines.index('/* [Beg] user-defined constants (do not change this line) */')+2:]:
            lll = ll.split()
            try:
                if ll == '':
                    break
                elif lll[2] == "YES":
                    object.__setattr__(D, lll[1], True)
                elif lll[2] == "NO":
                    object.__setattr__(D, lll[1], False)

                else:
                    object.__setattr__(D,lll[1],float(lll[2]))
            except:
                object.__setattr__(D,lll[1],str(lll[2]))

    else:
        print('Err: I did not find a definitions.h file in the working dir')



def get_lims(x,dx):
    xx = np.zeros(len(x)+1)
    xx[1:] = x + dx/2.0
    xx[0]  = xx[1] - dx[0]
    return xx

def getcyl(D):
    # Get cylindrical coordinate components of the vectors
    if (not hasattr(D,'xx')):
        compute_grids(D)
        
    if (D.n3_tot != 1):
        D.Vr   =  D.cosp * D.vx1 + D.sinp * D.vx2
        D.Vphi = -D.sinp * D.vx1 + D.cosp * D.vx2
        if (hasattr(D,'bx1')):
            D.Br   =  D.cosp * D.bx1 + D.sinp * D.bx2
            D.Bphi = -D.sinp * D.bx1 + D.cosp * D.bx2
    else:
        D.Vr   =  D.vx1 
        D.Vphi =  D.vx3
        if (hasattr(D,'bx1')):
            D.Br   =  D.bx1
            D.Bphi =  D.bx3

def getsph(D):
    # Get spherical coordinate components of the vectors
    if (not hasattr(D,'xx')):
        compute_grids(D)
        
    D.TT = -np.arctan(D.zz/D.Rcyl)+np.pi/2.
    D.PP = np.arctan2(D.yy,D.xx)
    if (D.n3_tot != 1):
        D.VR = np.sin(D.TT)*np.cos(D.PP)*D.vx1 + np.sin(D.TT)*np.sin(D.PP)*D.vx2 + np.cos(D.TT)*D.vx3
        D.VT = np.cos(D.TT)*np.cos(D.PP)*D.vx1 + np.cos(D.TT)*np.sin(D.PP)*D.vx2 - np.sin(D.TT)*D.vx3
        D.VP =             -np.sin(D.PP)*D.vx1 +              np.cos(D.PP)*D.vx2 
        if (hasattr(D,'bx1')):
            D.BR = np.sin(D.TT)*np.cos(D.PP)*D.bx1 + np.sin(D.TT)*np.sin(D.PP)*D.bx2 + np.cos(D.TT)*D.bx3
            D.BT = np.cos(D.TT)*np.cos(D.PP)*D.bx1 + np.cos(D.TT)*np.sin(D.PP)*D.bx2 - np.sin(D.TT)*D.bx3
            D.BP =             -np.sin(D.PP)*D.bx1 +              np.cos(D.PP)*D.bx2 

def getForcesSph(D):
    if not hasattr(D,'TT'):
        getsph(D)
    if not hasattr(D,'Jx1'):
        compute_derivs(D)

    D.JR = np.sin(D.TT)*np.cos(D.PP)*D.Jx1 + np.sin(D.TT)*np.sin(D.PP)*D.Jx2 + np.cos(D.TT)*D.Jx3
    D.JT = np.cos(D.TT)*np.cos(D.PP)*D.Jx1 + np.cos(D.TT)*np.sin(D.PP)*D.Jx2 - np.sin(D.TT)*D.Jx3
    D.JP =             -np.sin(D.PP)*D.Jx1 +              np.cos(D.PP)*D.Jx2 

    #Lorentz
    D.LR = D.JT*D.BP - D.JP*D.BT
    D.LT = D.JP*D.BR - D.JR*D.BP
    D.LP = D.JR*D.BT - D.JT*D.BR

    #Pressure Gradient
    D.gradP_R = np.sin(D.TT)*np.cos(D.PP)*D.gradP_x1 + np.sin(D.TT)*np.sin(D.PP)*D.gradP_x2 + np.cos(D.TT)*D.gradP_x3
    D.gradP_T = np.cos(D.TT)*np.cos(D.PP)*D.gradP_x1 + np.cos(D.TT)*np.sin(D.PP)*D.gradP_x2 - np.sin(D.TT)*D.gradP_x3
    D.gradP_P =             -np.sin(D.PP)*D.gradP_x1 +              np.cos(D.PP)*D.gradP_x2 

    # Momentum

    D.VGV_x1=D.vx1*D.dx1_vx1+D.vx2*D.dx2_vx1+D.vx3*D.dx3_vx1
    D.VGV_x2=D.vx1*D.dx1_vx2+D.vx2*D.dx2_vx2+D.vx3*D.dx3_vx2
    D.VGV_x3=D.vx1*D.dx1_vx3+D.vx2*D.dx2_vx3+D.vx3*D.dx3_vx3

    D.VGV_R = np.sin(D.TT)*np.cos(D.PP)*D.VGV_x1 + np.sin(D.TT)*np.sin(D.PP)*D.VGV_x2 + np.cos(D.TT)*D.VGV_x3
    D.VGV_T = np.cos(D.TT)*np.cos(D.PP)*D.VGV_x1 + np.cos(D.TT)*np.sin(D.PP)*D.VGV_x2 - np.sin(D.TT)*D.VGV_x3
    D.VGV_P =             -np.sin(D.PP)*D.VGV_x1 +              np.cos(D.PP)*D.VGV_x2 

def compute_grids(D):

    D.xx = np.tile( np.tile(D.x1,(D.n2,1))  , (D.n3,1,1)).T.squeeze()
    D.yy = np.tile( np.tile(D.x2,(D.n1,1)).T, (D.n3,1,1)).T.squeeze()
    D.zz = np.tile( np.tile(D.x3,(D.n2,1))  , (D.n1,1,1)).squeeze()

    if (D.n3_tot != 1):
        D.Rcyl = np.sqrt(D.xx**2 + D.yy**2)
        D.Rsph = np.sqrt(D.xx**2 + D.yy**2 + D.zz**2)
    else:
        D.Rcyl = D.xx
        D.Rsph = np.sqrt(D.xx**2+D.yy**2)

    D.cosp = D.xx/D.Rcyl ; D.sinp = D.yy/D.Rcyl
    D.cosp[np.where(D.Rcyl == 0.0)] = 0.0
    D.sinp[np.where(D.Rcyl == 0.0)] = 0.0

    D.x1r = get_lims(D.x1,D.dx1)
    D.x2r = get_lims(D.x2,D.dx2)
    if (D.n3_tot != 1):
        D.x3r = get_lims(D.x3,D.dx3)
    else:
        D.x3r = D.x3

    D.xxr = np.tile( np.tile(D.x1r,(D.n2+1,1))  , (D.n3+1,1,1)).T.squeeze()
    D.yyr = np.tile( np.tile(D.x2r,(D.n1+1,1)).T, (D.n3+1,1,1)).T.squeeze()
    D.zzr = np.tile( np.tile(D.x3r,(D.n2+1,1))  , (D.n1+1,1,1)).squeeze()


def shell_int(qs,D,rad):
    """Integrate a scalar field in a spherical shell"""
    if(not(hasattr(rad,'__iter__'))):
        rad=np.asarray([rad])
    
    ntt = 40; npp = 40
    tts = np.linspace(0.,np.pi,num=ntt)
    pps = np.linspace(0.,2.*np.pi,num=npp)

    if (hasattr (scipy.interpolate, 'RegularGridInterpolator')):
        if(D.x1.dtype==np.float32):
            x1double=D.x1.astype(float)
            x2double=D.x2.astype(float)
            x3double=D.x3.astype(float)

            qsdouble=qs.astype(float)
            qs_i = RegularGridInterpolator((x1double,x2double,x3double),qsdouble)
        else:
            qs_i = RegularGridInterpolator((D.x1,D.x2,D.x3),qs)
    else:
        print("I need a more recent version of scipy for the interpolation")
        return 0
        
    sh_int=np.array([])
    for rr in rad:
        tmp_q = np.zeros((ntt,npp))
        for it,tt in enumerate(tts):
            for ip,pp in enumerate(pps):
                x = rr*np.sin(tt)*np.cos(pp) ; y = rr*np.sin(tt)*np.sin(pp) ; z =rr*np.cos(tt)
                qs_l = qs_i((x,y,z))
                tmp_q[it,ip] = qs_l * rr**2 * np.sin(tt)

        # Now perform the integration
        sh_int=np.append(sh_int,np.trapz(np.trapz(tmp_q,x=tts,axis=0),x=pps))

    dr=rad[1]-rad[0]
    return np.sum(sh_int)*dr

def compute_flux(qx,qy,qz,D,rad,take_abs=False):
    """Computes the flux of vector q through a sphere of length r centered on the origin"""

    if(not(hasattr(rad,'__iter__'))):
        rad=np.asarray([rad])

    # Integrate over sphere with an interpolation
    if (D.n3_tot == 1):
        ntt = 40 ;
        tts = np.linspace(0.01,np.pi-0.01,num=ntt)
        if (hasattr (scipy.interpolate, 'RegularGridInterpolator')):
            qx_i = RegularGridInterpolator((D.x1,D.x2),qx)
            qy_i = RegularGridInterpolator((D.x1,D.x2),qy)
        else:
            print('I need a more recent version of scipy to trace field lines in 3D')
            return 0

        tmp_q = np.zeros((ntt))
        for it,tt in enumerate(tts):
            x = r*np.sin(tt) ; y = r*np.cos(tt)
            qx_l = qx_i((x,y)) ; qy_l = qy_i((x,y))
            tmp_q[it] = np.sin(tt)*qx_l + np.cos(tt)*qy_l
            tmp_q[it] = tmp_q[it] * r**2 * np.sin(tt) * 2.*np.pi

            # Now perform the integration
        if(take_abs):
            tmp_q = np.abs(tmp_q)
        flux=np.trapz(tmp_q,x=tts,axis=0)
        return flux

    else:
        ntt = 40; npp = 40
        tts = np.linspace(0.,np.pi,num=ntt)
        pps = np.linspace(0.,2.*np.pi,num=npp)
        if (hasattr (scipy.interpolate, 'RegularGridInterpolator')):
                
            if(D.x1.dtype==np.float32):
                x1double=D.x1.astype(float)
                x2double=D.x2.astype(float)
                x3double=D.x3.astype(float)
                qxdouble=qx.astype(float)
                qydouble=qy.astype(float)
                qzdouble=qz.astype(float)

                qx_i = RegularGridInterpolator((x1double,x2double,x3double),qxdouble)
                qy_i = RegularGridInterpolator((x1double,x2double,x3double),qydouble)
                qz_i = RegularGridInterpolator((x1double,x2double,x3double),qzdouble)
            else:
                qx_i = RegularGridInterpolator((D.x1,D.x2,D.x3),qx)
                qy_i = RegularGridInterpolator((D.x1,D.x2,D.x3),qy)
                qz_i = RegularGridInterpolator((D.x1,D.x2,D.x3),qz)            
        else:
            print("I need a more recent version of scipy for the interpolation")
            return 0
        
        flux=np.array([])
        for rr in rad:
            tmp_q = np.zeros((ntt,npp))
            for it,tt in enumerate(tts):
                for ip,pp in enumerate(pps):
                    x = rr*np.sin(tt)*np.cos(pp) ; y = rr*np.sin(tt)*np.sin(pp) ; z =rr*np.cos(tt)
                    qx_l = qx_i((x,y,z)) ; qy_l = qy_i((x,y,z)) ; qz_l = qz_i((x,y,z)) ; 
                    tmp_q[it,ip] = np.sin(tt)*np.cos(pp)*qx_l + np.sin(tt)*np.sin(pp)*qy_l + np.cos(tt)*qz_l
                    tmp_q[it,ip] = tmp_q[it,ip] * rr**2 * np.sin(tt)

        # Now perform the integration
            if(take_abs):
                tmp_q = np.abs(tmp_q)
        
            
            flux=np.append(flux,np.trapz(np.trapz(tmp_q,x=tts,axis=0),x=pps))

        return flux


def nufd1(x):
    n = len(x)
    if (n == 1):
        return 1.
    h = x[1:]-x[:n-1]
    a0 = -(2*h[0]+h[1])/(h[0]*(h[0]+h[1]))
    ak = -h[1:]/(h[:n-2]*(h[:n-2]+h[1:]))
    an = h[-1]/(h[-2]*(h[-1]+h[-2]))
    b0 = (h[0]+h[1])/(h[0]*h[1])
    bk = (h[1:] - h[:n-2])/(h[:n-2]*h[1:])
    bn = -(h[-1]+h[-2])/(h[-1]*h[-2])
    c0 = -h[0]/(h[1]*(h[0]+h[1]))
    ck = h[:n-2]/(h[1:]*(h[:n-2]+h[1:]))
    cn = (2*h[-1]+h[-2])/(h[-1]*(h[-2]+h[-1]))
    val  = np.hstack((a0,ak,an,b0,bk,bn,c0,ck,cn))
    row = np.tile(np.arange(n),3)
    dex = np.hstack((0,np.arange(n-2),n-3))
    col = np.hstack((dex,dex+1,dex+2))
    D = coo_matrix((val,(row,col)),shape=(n,n))
    return D

def nufd2(x):
    n = len(x)
    h = x[1:]-x[:n-1]
    a = 2/(h[:n-2]*(h[1:]+h[:n-2]))
    b = -2/(h[1:]*h[:n-2])
    c = 2/(h[1:]*(h[1:]+h[:n-2]))
    val  = np.hstack((a[0],a,a[-1],b[0],b,b[-1],c[0],c,c[-1]))
    row = np.tile(np.arange(n),3)
    dex = np.hstack((0,np.arange(n-2),n-3))
    col = np.hstack((dex,dex+1,dex+2))
    D2 = csr_matrix((val,(row,col)),shape=(n,n))
    return D2

def deriv(r,f,order=1):
    if (order == 1):
        DD = nufd1(r)
    elif (order == 2):
        DD = nufd2(r)
    else:
        print ('Derivative not coded yet')
        return f
    return DD*f

def compute_derivs(D):
    if (not hasattr(D,'xx')):
        compute_grids(D)
    if (not hasattr(D,'R_PLANET')):
        get_caseparams(D)

    # Derivatives matrices
    d1 = nufd1(D.x1) ; d2 = nufd1(D.x2) ; d3 = nufd1(D.x3)

    # Derivatives for advection
    D.dx1_vx1 = 0.*D.vx1 ; D.dx2_vx1 = 0.*D.vx1 ; D.dx3_vx1 = 0.*D.vx1
    D.dx1_vx2 = 0.*D.vx1 ; D.dx2_vx2 = 0.*D.vx1 ; D.dx3_vx2 = 0.*D.vx1
    D.dx1_vx3 = 0.*D.vx1 ; D.dx2_vx3 = 0.*D.vx1 ; D.dx3_vx3 = 0.*D.vx1
    for i in range(D.n1):
        for j in range(D.n2):
            D.dx3_vx1[i,j,:] = d3*D.vx1[i,j,:]
            D.dx3_vx2[i,j,:] = d3*D.vx2[i,j,:]
            D.dx3_vx3[i,j,:] = d3*D.vx3[i,j,:]
    for i in range(D.n1):
        for k in range(D.n3):
            D.dx2_vx1[i,:,k] = d2*D.vx1[i,:,k]
            D.dx2_vx2[i,:,k] = d2*D.vx2[i,:,k]
            D.dx2_vx3[i,:,k] = d2*D.vx3[i,:,k]
    for j in range(D.n2):
        for k in range(D.n3):
            D.dx1_vx1[:,j,k] = d1*D.vx1[:,j,k]
            D.dx1_vx2[:,j,k] = d1*D.vx2[:,j,k]
            D.dx1_vx3[:,j,k] = d1*D.vx3[:,j,k]

    # Pressure gradient
    D.gradP_x1 =  0.*D.prs ; D.gradP_x2 =  0.*D.prs ; D.gradP_x3 =  0.*D.prs
    for i in range(D.n1):
        for j in range(D.n2):
            D.gradP_x3[i,j,:] = -d3*D.prs[i,j,:]
    for i in range(D.n1):
        for k in range(D.n3):
            D.gradP_x2[i,:,k] = -d2*D.prs[i,:,k]
    for j in range(D.n2):
        for k in range(D.n3):
            D.gradP_x1[:,j,k] = -d1*D.prs[:,j,k]

    # Gravity
    D.pot_g = -1./D.Rsph
    if (D.MASSRATIO != 0):
        D.Rsph_p = np.sqrt((D.xx-D.R_ORBIT_PLANET)**2+D.yy**2+D.zz**2)
        D.pot_g = D.pot_g - D.MASSRATIO/D.Rsph_p
    D.g_x3 = 0.*D.rho ; D.g_x2 = 0.*D.rho ; D.g_x1 = 0*D.rho
    for i in range(D.n1):
        for j in range(D.n2):
            D.g_x3[i,j,:] = (d3*D.pot_g[i,j,:])
    D.g_x3 = -D.rho*D.g_x3
    for i in range(D.n1):
        for k in range(D.n3):
            D.g_x2[i,:,k] = (d2*D.pot_g[i,:,k])
    D.g_x2 = -D.rho*D.g_x2
    for j in range(D.n2):
        for k in range(D.n3):
            D.g_x1[:,j,k] = (d1*D.pot_g[:,j,k])
    D.g_x1 = -D.rho*D.g_x1

    # Currents
    D.dx2_bx1 = 0.*D.vx1 ; D.dx3_bx1 = 0.*D.vx1
    D.dx1_bx2 = 0.*D.vx1 ; D.dx3_bx2 = 0.*D.vx1
    D.dx1_bx3 = 0.*D.vx1 ; D.dx2_bx3 = 0.*D.vx1 
    for i in range(D.n1):
        for j in range(D.n2):
            D.dx3_bx1[i,j,:] = d3*D.bx1[i,j,:]
            D.dx3_bx2[i,j,:] = d3*D.bx2[i,j,:]
    for i in range(D.n1):
        for k in range(D.n3):
            D.dx2_bx1[i,:,k] = d2*D.bx1[i,:,k]
            D.dx2_bx3[i,:,k] = d2*D.bx3[i,:,k]
    for j in range(D.n2):
        for k in range(D.n3):
            D.dx1_bx2[:,j,k] = d1*D.bx2[:,j,k]
            D.dx1_bx3[:,j,k] = d1*D.bx3[:,j,k]
    D.Jx1 = D.dx2_bx3 - D.dx3_bx2
    D.Jx2 = D.dx3_bx1 - D.dx1_bx3
    D.Jx3 = D.dx1_bx2 - D.dx2_bx1


def compute_jdot(D,rad,v2=False,recompute=False,rotF=False):
    "Compute the angular momentum loss rate"

    if (not hasattr(D,'xx')):
        compute_grids(D)
    
    if os.path.exists(D.wdir+'jdot_saved') and (not recompute):
        [D.rad_j,D.jdot_r] = read_elements(2,D.wdir+'jdot_saved')
    else:
        D.rad_j = rad
        D.jdot_r = np.zeros(len(rad))

        if (v2):
            D.OmS = 0. 
            if (rotF):
                if (D.R_ORBIT_PLANET != 0.):
                    D.OmS = 1./(D.R_ORBIT_PLANET**1.5)
                else:
                    D.OmS = D.VphiStar_VESC*np.sqrt(2.)
            D.BB  = np.sqrt(D.bx1**2 + D.bx2**2 + D.bx3**2)
            qx = (-D.xx*D.bx2 + D.yy*D.bx1)*D.bx1 - D.yy*(D.prs+(D.BB**2)/2.) + ( D.xx*D.vx2 - D.yy*D.vx1 + (D.xx**2+D.yy**2)*D.OmS)*D.rho*D.vx1
            qy = (-D.xx*D.bx2 + D.yy*D.bx1)*D.bx2 + D.xx*(D.prs+(D.BB**2)/2.) + ( D.xx*D.vx2 - D.yy*D.vx1 + (D.xx**2+D.yy**2)*D.OmS)*D.rho*D.vx2
            qz = (-D.xx*D.bx2 + D.yy*D.bx1)*D.bx3                             + ( D.xx*D.vx2 - D.yy*D.vx1 + (D.xx**2+D.yy**2)*D.OmS)*D.rho*D.vx3 
        else:
            if (not hasattr(D,'L')):
                compute_conserved_quantities(D)
            D.BB  = np.sqrt(D.bx1**2 + D.bx2**2 + D.bx3**2)
            qx = D.L*D.rho*D.vx1 #- D.yy*(D.prs+(D.BB**2)/2.)
            qy = D.L*D.rho*D.vx2 #+ D.xx*(D.prs+(D.BB**2)/2.)
            qz = D.L*D.rho*D.vx3



        D.jdot_r = compute_flux(qx,qy,qz,D,rad)
        #save_elements([D.rad_j,D.jdot_r],D.wdir+'jdot_saved')

def compute_mdot(D,rad,recompute=False,mypath=None):
    "Compute the mass loss rate"
    
    if (mypath == None):
        mypath = D.wdir

    if os.path.exists(mypath+'mdot_saved') and (not recompute):
        [D.rad_m,D.mdot_r] = read_elements(2,mypath+'mdot_saved')
    else:
        qx = D.rho*D.vx1
        qy = D.rho*D.vx2
        qz = D.rho*D.vx3
        
        D.rad_m = rad
        D.mdot_r = np.zeros(len(rad))

        D.mdot_r = compute_flux(qx,qy,qz,D,rad)
        #save_elements([D.rad_m,D.mdot_r],mypath+'mdot_saved')


def spherical_interp(D,qty,theta,phi,rad):
    iqty=scipy.interpolate.RegularGridInterpolator((D.x1,D.x2,D.x3),qty)    
    list_spqty=[]
    for rr in rad:
        spqty=iqty((rr*np.sin(theta)*np.cos(phi),rr*np.sin(theta)*np.sin(phi),rr*np.cos(theta)))
        list_spqty.append(spqty)

    return list_spqty
