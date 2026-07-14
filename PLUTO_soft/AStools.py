from __future__ import print_function
from platform import python_version
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
from pylab import *
import pyPLUTO as pp
import glob
from mpl_toolkits.axes_grid.inset_locator import BboxPatch, inset_axes
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes, mark_inset
import scipy.interpolate
from scipy.interpolate import RectBivariateSpline
if (hasattr(scipy.interpolate, 'RegularGridInterpolator')):
    from scipy.interpolate import RegularGridInterpolator
try:
    import cPickle as pickle
except ImportError:
    import pickle as pickle
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    print("<AS> Missing basemap")
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import sph_harm
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.optimize import curve_fit
from matplotlib.image import NonUniformImage
import copy as cp
from matplotlib.collections import Collection 
from matplotlib.artist import allow_rasterization 
from subprocess import call

def GetProcs(D):
    # Get log files in directory (pluto 4.3)
    files=glob.glob(D.wdir+'pluto.*.log')
    D.ProcCoords=[]
    for fname in files:
        iproc=int(fname.split('.')[-2])
        lines=[line.strip() for line in open(fname)]
        indxl=lines.index('Local grid:')+1
        Coords={}
        l1=lines[indxl].split('[')[1].split(']')[0].split(',')
        Coords["x1beg"] = float(l1[0])
        Coords["x1end"] = float(l1[1])
        if D.n2 != 1:
            l1=lines[indxl+1].split('[')[1].split(']')[0].split(',')
            Coords["x2beg"] = float(l1[0])
            Coords["x2end"] = float(l1[1])
        if D.n3 != 1:
            l1=lines[indxl+2].split('[')[1].split(']')[0].split(',')
            Coords["x3beg"] = float(l1[0])
            Coords["x3end"] = float(l1[1])
        D.ProcCoords.append(Coords)

def PlotProcGrid(D,geom='cartesian',color='k',lw=0.5):
    if not hasattr(D,'ProcCoords'):
        GetProcs(D)
    for Coords in D.ProcCoords:
        x1=np.linspace(Coords["x1beg"],Coords["x1end"])
        x2=np.linspace(Coords["x2beg"],Coords["x2end"])
        PlotUnitGrid(x1,x2,mode=geom,color=color,lw=lw)
    
def PlotUnitGrid(x1,x2,mode='spherical',color='k',alpha=0.5,ls=':',lw=0.5):
    ones=np.ones_like(x1)
    if mode == 'spherical':
        plot(x1*np.sin(x2[0]),x1*np.cos(x2[0]),color=color,alpha=alpha,ls=ls,lw=lw)
        plot(x1[-1]*np.sin(x2),x1[-1]*np.cos(x2),color=color,alpha=alpha,ls=ls,lw=lw)
        plot(x1*np.sin(x2[-1]),x1*np.cos(x2[-1]),color=color,alpha=alpha,ls=ls,lw=lw)
        plot(x1[0]*np.sin(x2),x1[0]*np.cos(x2),color=color,alpha=alpha,ls=ls,lw=lw)
    elif mode == 'cartesian':
        plot(x1,x2[0]*ones,color=color,alpha=alpha,ls=ls,lw=lw)
        plot(x1[-1]*ones,x2,color=color,alpha=alpha,ls=ls,lw=lw)
        plot(x1,x2[-1]*ones,color=color,alpha=alpha,ls=ls,lw=lw)
        plot(x1[0]*ones,x2,color=color,alpha=alpha,ls=ls,lw=lw)


def mkdir(mydir):
    call("mkdir -p "+mydir,shell=True)

def IterateCmap(cmap,num_colors):
    cm=get_cmap(cmap)
    color=[cm(1.*i/num_colors) for i in range(num_colors)]
    return color

def ReadCol(filename,nheader=1):
    f=open(filename,'r')
    lines=f.readlines()
    nr = len(lines)-nheader
    nf = len(lines[nr/2].strip().split())
    l_fields = [np.zeros(nr) for il in range(nf)]
    for ir,ll in enumerate(lines[nheader:]):
        col = ll.strip().split()
        for ifi in range(nf):
            l_fields[ifi][ir] = col[ifi].replace('D','e')
    return l_fields

def plotslice(D,var,cut,**kwargs):

    toplot=getarr(D,var)

    if cut==0:
        ix=np.abs(D.x1-0.5*(D.x1[0]+D.x1[-1])).argmin()
        x1=D.x2r ; x2 = D.x3r ; tp = toplot[ix,:,:].T
    elif cut==1:
        iy=np.abs(D.x2-0.5*(D.x2[0]+D.x2[-1])).argmin()
        x1=D.x1r ; x2 = D.x3r ; tp = toplot[:,iy,:].T
    elif cut==2:
        iz=np.abs(D.x3-0.5*(D.x3[0]+D.x3[-1])).argmin()
        x1=D.x1r ; x2 = D.x2r ; tp = toplot[:,:,iz].T

    ax=gca()
    pc=ax.pcolorfast(x1,x2,tp,**kwargs)

def plot_three_slices(D,var,ax1,ax2,ax3,**kwargs):
    sca(ax1)
    plotslice(D,var,0,**kwargs)
    sca(ax2)
    plotslice(D,var,1,**kwargs)
    sca(ax3)
    plotslice(D,var,2,**kwargs)

def unload_j():
    rcdefaults()        

def format1(x):
    return '%.1f' % x
def format2(x):
    return '%.2f' % x

def area(vs):
    a = 0
    x0, y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return np.abs(a)

def SciNot(num, decimal_digits=1, precision=None, exponent=None):
    """                                                                                                                                                                                          
    Returns a string representation of the scientific                                                                                                                                                      
    notation of the given number formatted for use with                                                                                                                                                       
    LaTeX or Mathtext, with specified number of significant                                                                                                                                                     
    decimal digits and precision (number of decimal digits                                                                                                                                                        
    to show). The exponent to be used can also be specified                                                                                                                                                     
    explicitly.                                                                                                                                                                                 
    """
    if not exponent:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if not precision:
        precision = decimal_digits

    return "${0:.{2}f} \cdot 10^{{{1:d}}}$".format(coeff, exponent, precision)

def BaseNot(num):
    if type(num) is dict:
        if num.has_key('decimals'):
            return "${0:.{1}f}".format(num['val'],num['decimals'])
        else:
            return "${0:.2f}".format(num['val'])            
    elif type(num) is list:
        return "${0:.{1}f}".format(num[0],num[1])            
    else:
        return "${0:.2f}".format(num)            
    
def FloatNot(num, dd1=2,dd2=-1):
    if dd2 == -1:
        dd2 = dd1
    if type(num) is list:
        if len(num) <= 2:
            if num[0] == 0. and num[1] == 0.:
                return '-'
            elif num[1] == 0.:
                return("${0:.{1}f}$".format(num[0],dd1))
            else:
                return "${0:.{1}f} \pm {2:.{3}f}$".format(num[0],dd1,num[1],dd2)
        else:
            return "${0:.{1}f} \pm {2:.{3}f} {4}$".format(num[0],dd1,num[1],dd2,num[2])
    elif type(num) is dict:
        if num.has_key('err'):
            return BaseNot(num['val'])+" \pm "+BaseNot(num['err']) #"${0:.{1}f} \pm {2:.{3}f} {4}$".format(num[0],dd1,num[1],dd2,num[2])
        else:
            return BaseNot(num['val'])
    else:
        if num == 0.:
            return '-'
        else:
            return "${0:.{1}f}$".format(num,dd1)

##
def get_pluto_iters(wdir='',suff=''):
    files=glob.glob(wdir+'data.*'+suff+'.dbl')
    iters=[int(v.split('/')[-1].split('.')[-2]) for v in files]
    iters.sort()
    return iters

## Do a mean over circle
def get_sphmean(x1,x2,v,rorb,nsample=10):
    new_v = RectBivariateSpline(x1,x2,v)
    Cpoints = [(rorb*np.cos(np.pi/2.+np.pi*ii/nsample),rorb*np.sin(np.pi/2.+np.pi*ii/nsample)) for ii in range(nsample)]
    v_intp = np.zeros(nsample)
    for iis,xyc in enumerate(Cpoints):
        v_intp[iis] = new_v(xyc[0],xyc[1])
    return np.mean(v_intp)

## Do a mean over circle
def slice_sphere(x1,x2,x3,v,rorb,nsample=10):
    new_v = RegularGridInterpolator((x1,x2,x3),v)
    th = np.linspace(0.,pi,num=nsample)[1:-1]
    ph = np.linspace(0.,2.*pi,num=nsample)
    v_intp = np.zeros((nsample-2,nsample))
    for ii,tt in enumerate(th):
        for jj,pp in enumerate(ph):
            v_intp[ii,jj] = new_v((rorb*np.sin(tt)*np.cos(pp),\
                                       rorb*np.sin(tt)*np.sin(pp),\
                                       rorb*np.cos(tt)))
    return v_intp,th,ph

## Do a slice
def slice_plane(xo,n,q,x1,x2,x3,nx=50,ny=50,xxlim=[-1,1],yylim=[-1,1]):
    x1_sl = np.linspace(xxlim[0],xxlim[1],num=nx)
    x2_sl = np.linspace(yylim[0],yylim[1],num=ny)
    if (hasattr (scipy.interpolate, 'RegularGridInterpolator')):
        q_i = RegularGridInterpolator((x1,x2,x3),q)
    else:
        print('I need a more recent version of scipy')
        return 0
    q_sl = np.zeros((nx,ny))
    for ix,xx in enumerate(x1_sl):
        for iy,yy in enumerate(x2_sl):
            x=xo[0]+xx ; y = xo[1]+n[2]*yy ; z = xo[2]-n[1]*yy
            q_sl[ix,iy] = q_i((x,y,z))
    return [x1_sl,x2_sl,q_sl]

## Reduce range of python structure
def reduce_range(D,x1range,x2range,x3range):
    ## Take care of grids
    x1b,x1e = np.abs(D.x1-x1range[0]).argmin(),np.abs(D.x1-x1range[1]).argmin()
    x2b,x2e = np.abs(D.x2-x2range[0]).argmin(),np.abs(D.x2-x2range[1]).argmin()
    x3b,x3e = np.abs(D.x3-x3range[0]).argmin(),np.abs(D.x3-x3range[1]).argmin()
    Do = cp.copy(D)
    ## Take care of 3D fields
    for kk in D.__dict__.keys():
        sk=getattr(D,kk).shape
        if (sk == np.shape(D.vx1)):
            setattr(Do,kk,getattr(D,kk)[x1b:x1e,x2b:x2e,x3b:x3e])
            #exec("Do.%s = D.%s[x1b:x1e,x2b:x2e,x3b:x3e]" % (kk,kk))
        else:
            setattr(Do,kk,getattr(D,kk))
            #exec("Do.%s = D.%s" % (kk,kk))
    ## Redefine the geometrical arrays
    xs={"x1b":x1b,"x1e":x1e,"x2b":x2b,"x2e":x2e,"x3b":x3b,"x3e":x3e}
    for ii in [1,2,3]:
        xloc="x{}".format(i)
        nloc="n{}".format(i)
        setattr(Do,xloc,getattr(D,xloc)[xs[xloc+'b'],xs[xloc+'e']])
        setattr(Do,xloc+'r',getattr(D,xloc+'r')[xs[xloc+'b'],xs[xloc+'e']])
        setattr(Do,'d'+xloc,getattr(D,'d'+xloc)[xs[xloc+'b'],xs[xloc+'e']])
        setattr(Do,nloc,getattr(D,nloc))
        #exec ("Do.x%i = D.x%i[x%ib:x%ie]" % (ii,ii,ii,ii)  )
        #exec ("Do.x%ir = D.x%ir[x%ib:x%ie]" % (ii,ii,ii,ii))
        #exec ("Do.dx%i = D.dx%i[x%ib:x%ie]" % (ii,ii,ii,ii))
        #exec ("Do.n%i = len(Do.x%i)" % (ii,ii)             )
    return Do

## Reduce resolution of the simulation to make things faster 
def reduce_resolution(D,dec=2):
    Do = cp.copy(D)
    for kk in D.__dict__.keys():
        #exec("sk = np.shape(D.%s)" % (kk))
        sk=getattr(D,kk).shape
        if (sk == np.shape(D.vx1)):
            setattr(Do,kk,getattr(D,kk)[::dec,::dec,::dec])
            #exec("Do.%s = D.%s[::dec,::dec,::dec]" % (kk,kk))
        else:
            setattr(Do,kk,getattr(D,kk))
            #exec("Do.%s = D.%s" % (kk,kk))
    ## Redefine the geometrical arrays
    for ii in [1,2,3]:
        xloc="x{}".format(i)
        nloc="n{}".format(i)
        setattr(Do,xloc,getattr(D,xloc)[::dec])
        setattr(Do,xloc+'r',getattr(D,xloc+'r')[::dec])
        setattr(Do,'d'+xloc,getattr(D,'d'+xloc)[::dec]*dec)
        setattr(Do,nloc,len(getattr(D,xloc)))
        #exec ("Do.x%i  = D.x%i[::dec]"      % (ii,ii))
        #exec ("Do.x%ir = D.x%ir[::dec]"     % (ii,ii))
        #exec ("Do.dx%i = D.dx%i[::dec]*dec" % (ii,ii))
        #exec ("Do.n%i  = len(Do.x%i)"       % (ii,ii))
    Do.dec_astools = dec
    return Do

## Fitting routines
def fct(x,*a):
    f=0.*x
    for i in range(len(a)):
        f = f + a[i]*(x**i)
    return f

def fit_poly(x,f,a0):
    r0,pcov = curve_fit(fct,x,f,p0=a0)
    #print (r0)
    res = fct(x,*r0)
    return res,r0

def fit_generic(x,f,a0,ff=fct,verbose=False):
    r0,pcov = curve_fit(ff,x,f,p0=a0)
    # Calculate reduced chi_square
    if (verbose):
        print("Variance of the fit:",np.sqrt(np.diag(pcov)))
    res = ff(x,*r0)
    return res,r0

def fct_cos(x,*a):
    f = a[0] + a[1]*np.cos(a[2]*x)
    return f

def fct_cos2(x,*a):
    f = a[0] + a[1]*np.cos(0.5*x)
    return f

def fct_m(x,*a):
    aa = a[1]*x[0] + a[0] + 1./x[0]
    return (aa*x)/(1.+a[0]*x+a[1]*x**2)

def fct_m2(x,*a):
    #aa = a[1]*x[0] + a[0] + 1./x[0]
    return a[0]*(x**a[2])/(1.+a[1]*x)**(a[2]+1)

def fct_npar(X,*a):
    res = a[0]
    for ix,x in enumerate(X):
        res *= x**a[ix]
    return res
def fct_3par(X,*a):
    x,y,z = X
    return a[0]*(x)**a[1]*(y)**a[2]*(z)**a[3]
def fct_2par(X,*a):
    x,y = X
    return a[0]*(x)**a[1]*(y)**a[2]
def fct_1par(X,*a):
    return a[0]*(X)**a[1]
def fct_log_2par(X,*a):
    x,y=X
    return a[0] + a[1]*x + a[2]*y
def fct_log_1p(X,a1,a2):
    return a1 + a2*X 
def fct_log_2p(X,a1,a2,a3):
    x,y=X
    return a1 + a2*x + a3*y
def fct_logCos_3p(X,a1,a2,a3,a4):
    x,y,z=X
    return np.log(a1) + a2*x + a3*y + np.log(1.+(a4-1.)*np.cos(z))
def fct_logCos2_3p(X,a1,a2,a3,a4,a5):
    x,y,z=X
    return np.log(a1) + a2*x + a3*(1+a5*np.cos(z))*y + np.log(1.+(a4-1.)*np.cos(z))
def fct_log_3p(X,a1,a2,a3,a4):
    x,y,z=X
    return a1 + a2*x + a3*y + a4*z
def fct_2p(X,a1,a2,a3):
    x,y=X
    return a1 * x**a2 * y**a3
def fct_line(x,a1,a2):
    return a1*x + a2
def fct_sq(X,a1,a2,a3):
    return a3*X**2 + a2*X + a1
def fct_cu(X,a1,a2,a3,a4):
    return a4*X**3 + a3*X**2 + a2*X + a1
def fct_sq2(X,a1,a2,a3,a4,a5,a6,a7):
    return a7*X**6 + a6*X**5 + a5*X**4 + a4*X**3 + a3*X**2 + a2*X + a1
def fct_sq3(X,a1,a2,a3,a4,a5,a6):
    return a2*X**a4 + a1*X**a5 + a3*X**a6
def fct_cu2(X,a1,a2,a3,a4,a5,a6):
    return a6*X**5 + a5*X**4 + a4*X**3 + a3*X**2 + a2*X + a1

def fct_step(x,*a):
    step = 0.5*(1.+tanh((x-a[0])/a[1]))
    return (x/x[0])*(1.-step) + a[2]*step/x

def fct_step2(x,*a):
    step = 0.5*(1.+tanh((x-a[0])/a[1]))
    return (a[2]*x+a[3])*(1.-step) + a[4]*step

def fit_m(x,f,a0):
    r0,pcov = curve_fit(fct_m,x,f,p0=a0)
    print (r0)
    res = fct_m(x,*r0)
    return res,r0

def ylm(m,n,phi,theta):
    ll = double(n) ; mm = double(m)
    return sph_harm(m,n,phi,theta)
def xlm(m,n,phi,theta):
    ll = double(n) ; mm = double(m)
    return sph_harm(m,n,phi,theta)*1j*mm/(sin(theta)*(ll+1))
def zlm(m,n,phi,theta):
    ll = double(n) ; mm = double(m)
    cc = ((ll+1.-mm) / (ll+1.) )/sqrt( ((2.*(ll+1.)+1.0)*(ll+1-mm)) / ((2.*ll+1.0)*(ll+1.+mm)) )
    return (-sph_harm(m,n,phi,theta)*cos(theta) + cc*sph_harm(m,n+1,phi,theta))/sin(theta)


def read_Bfield(myfile,dir="/home/astrugar/PYTHON_PLUTO/",version=1):

    filename = dir+myfile
    f = open(filename,'r')
    tmp = f.readline()
    params = f.readline().split()
    nharms = int(params[0]) ; ncomps = params[1]
    nl = int((-3+np.sqrt(9+8*nharms))/2.) # 5 #nharms    
    alpha = np.zeros(nharms,dtype=complex)
    ii = 0
    for n in r_[1:nl+1]:
        for m in range(n+1):
            vals = f.readline().split()
            alpha[ii] = complex(float(vals[2]),-float(vals[3]))
            ii = ii + 1
    tmp=f.readline()
    beta = np.zeros(nharms,dtype=complex) 
    ii = 0
    for n in r_[1:nl+1]:
        for m in range(n+1):
            vals = f.readline().split()
            beta[ii] = -complex(float(vals[2]),-float(vals[3]))
            if (version == 2):
                beta[ii] -= alpha[ii]
            ii = ii + 1
    tmp=f.readline()
    gamma = np.zeros(nharms,dtype=complex) 
    ii = 0
    for n in r_[1:nl+1]:
        for m in range(n+1):
            vals = f.readline().split()
            gamma[ii] = complex(float(vals[2]),-float(vals[3]))
            ii = ii + 1
    f.close()
    return alpha,beta,gamma

def extrapol_B(myfile,theta,phi,dir="/home/astrugar/PYTHON_PLUTO/",psfile='None',R=1.5,Rb=0.7,\
                   cclim=40.,liml=-1):

    ###############################
    ##### Potential extrapolation #
    ###############################
    pi = np.pi ; cos = np.cos ; sin = np.sin

    Rss = 50.;
    alpha,beta,gamma = read_Bfield(myfile,dir=dir)
    if (liml < 0):
        nl = int((-3+np.sqrt(9+8*len(alpha)))/2.)
    else:
        nl = liml

    Alm = np.zeros(np.shape(alpha),dtype=complex)
    Almm = np.zeros(np.shape(alpha),dtype=complex)
    Almp = np.zeros(np.shape(alpha),dtype=complex)
    Blm = np.zeros(np.shape(alpha),dtype=complex)
    Blmm = np.zeros(np.shape(alpha),dtype=complex)
    Blmp = np.zeros(np.shape(alpha),dtype=complex)
    ii =0 
    for n in r_[1:nl+1]:
        for m in range(n+1):
            Blm[ii] = alpha[ii]/( (1.+n) * (Rb**(-(n+2.))) + n * (Rss**(-(2.*n+1.))) * (Rb**(n-1.)) )
            Alm[ii] = - (Rss**(-(2.*n+1))) * Blm[ii]
            ii=ii+1
    ii =0 
    for n in r_[1:nl+1]:
        for m in range(n+1):
            if ((n == 1) or (n == m)):
                Blmm[ii] = 0.  ; Almm[ii] = 0.
            else:
                Blmm[ii] = Blm[ii-n] ; Almm[ii] = Alm[ii-n]
            if (n == 5):
                Blmp[ii] =0. ; Almp[ii] = 0.
            else:
                Blmp[ii] = Blm[ii+n+1] ; Almp[ii] = Alm[ii+n+1]
            ii=ii+1
            
    br = np.zeros(np.shape(theta))
    bt = np.zeros(np.shape(theta))
    bp = np.zeros(np.shape(theta))
    ii=0
    I = complex(0.,1.)
    for n in r_[1:nl+1]:
        for m in r_[0:n+1]:
            yy = ylm(m,n,phi,theta)
            if ((n == 1) and (m == 0)):
                yym = sqrt(1./(4.*pi))
            if (n==m):
                yym = 0.
            else:
                yym = ylm(m,n-1,phi,theta)
            if (m == 0):
                coeff = 1.0
            else:
                coeff = 1.
            yyp = ylm(m,n+1,phi,theta)
            l = 1.0*n ; mm = 1.0*m ; lp = 1.0*(n+1) ; lm = 1.0*(n-1)
            
            rml  = sqrt((l*l-mm*mm)/(4.*l*l-1.))
            rmlp = sqrt((lp*lp-mm*mm)/(4.*lp*lp-1.))
            
            tmp = - yy * coeff * ( Alm[ii]*n*(R**lm) - Blm[ii]*lp*(R**(-(l+2.))) )
            br = br + tmp.real
            
            tmp = - coeff * ( Alm[ii]*(R**l ) + Blm[ii]*(R**(-lp))) * ( l*rmlp*yyp - lp*rml*yym)
            bt = bt + tmp.real/(R*sin(theta))
            
            tmp = - yy * coeff * mm * I * ( Alm[ii]*(R**l) + Blm[ii]*(R**(-lp)) )
            bp = bp + tmp.real/(R*sin(theta))
            ii = ii + 1

    if (psfile != 'None'):
        plot_fields(br,bt,bp,theta,phi,psfile=psfile,myfile=myfile)

    return br,bt,bp

def reconstruct_B(myfile,theta,phi,dir="/home/astrugar/PYTHON_PLUTO/",psfile='None',cclim=40.,version=1):
    ###########################################################################################
    ##### Reconstruct the components of the magnetic field on the given grid of point theta,phi 
    ###########################################################################################
    pi = np.pi ; cos = np.cos ; sin = np.sin

    #nl = 5 #nharms    
    alpha,beta,gamma = read_Bfield(myfile,dir=dir,version=version)
    nl = int((-3+np.sqrt(9+8*len(alpha)))/2.)    
    
    br = np.zeros(np.shape(theta))
    bt = np.zeros(np.shape(theta))
    bp = np.zeros(np.shape(theta))
    ii=0
    for n in r_[1:nl+1]:
        for m in range(n+1):
            xx = xlm(m,n,phi,theta)
            yy = ylm(m,n,phi,theta)
            zz = zlm(m,n,phi,theta)
            tmp = alpha[ii]*yy
            br = br + tmp.real
            tmp = beta[ii]*zz - gamma[ii]*xx
            bt = bt + tmp.real
            tmp = beta[ii]*xx + gamma[ii]*zz
            bp = bp + tmp.real
            ii = ii + 1 
                        
    if (psfile != 'None'):
        plot_fields([br,bp,-bt],theta,phi,psfile=psfile,cclim=cclim,tits=[r'$B_r$',r'$B_\varphi$',r'$-B_\theta$'],mycmap='bwr')
    else:
        return br,bt,bp

def plot_fields(list_bs,theta,phi,psfile=None,cclim=40.,list_lats=[90,0,-90],tits=None,mycmap='RdBu_r'):
    fig=figure(figsize=(1+2*len(list_bs),2*len(list_lats)))
    subplots_adjust(hspace=0.1,wspace=0.05)
    titles = ['Lat ='+str(v) for v in list_lats]
    nf = len(list_bs)
    nl = len(list_lats)
    nlev=10
    for ii,ilat in enumerate(list_lats):
        for iff,ff in enumerate(list_bs):
            subplot(nf,nl,1+iff*nl+ii)
            m = Basemap(resolution='c',projection='ortho',lon_0=0.,lat_0=ilat)
            im1 = m.contourf(phi*180./pi,-theta*180/pi+90.,ff,list(np.linspace(-cclim,cclim,nlev)),\
                                  latlon=True,cmap=mycmap,extend='both')
            m.drawmapboundary()
            parallels = np.arange(-80.,80,20.)
            m.drawparallels(parallels)
            if (iff == 0):
                title(titles[ii])
            if (tits != None) and (ii==0):
                annotate(tits[iff],xycoords='axes fraction',xy=(0.01,0.01))
            if (ii == nl-1):
                pos1=gca().get_position()
                xx0 = pos1.x0+pos1.width
                cax=gcf().add_axes([xx0,pos1.y0,0.02,pos1.height])
                norm = mpl.colors.Normalize(vmin=-cclim,vmax=cclim)
                cb1=mpl.colorbar.ColorbarBase(cax,cmap=mycmap,norm=norm,\
                                                  orientation='vertical',ticks=[-cclim,0,cclim],\
                                                  extend='both')
                cb1.set_label('[G]')
    
    if (psfile != None):
        fig.savefig(psfile,bbox_inches='tight')


def get_sphSlice(D,rsp,theta,phi,lqty=['rho']):
    """ Slices cartesian box on a sphere of given radius """
    if (not hasattr(D,'Rsph')):
        compute_grids(D)
    
    for qty in lqty:
        if (not hasattr(D,qty)):
            print ('Please make sure qty '+qty+' is presen')
        else:
            toslice=getattr(D,qty)
            #exec ("toslice=D.%s" % (qty))
    
        if (hasattr (scipy.interpolate, 'RegularGridInterpolator')):
            new_qty = RegularGridInterpolator((D.x1,D.x2,D.x3),toslice)
        else:
            print ('I need a more recent version of scipy to slice on a sphere')
            sys.exit(0)
        
        SphSlice = np.zeros(np.shape(theta))
        for (i,j) in np.ndindex(np.shape(theta)):
            tt = theta[i,j]
            pp = phi[i,j]
            x  = rsp * np.sin(tt) * np.cos(pp)
            y  = rsp * np.sin(tt) * np.sin(pp)
            z  = rsp * np.cos(tt)
            SphSlice[i,j] = new_qty([x,y,z])

        setattr(D,"{}_SphSlice".format(qty),SphSlice)
        #exec ("D.%s_SphSlice = SphSlice" % (qty))

def my_log_pdf(T_input,sub_bin=5,lw=1,color='k',normed=False):
    # filter NaN's
    T_av = 1.*T_input[~np.isnan(T_input)]
    # filter Inf's
    T_av = 1.*T_av[~np.isinf(T_av)]
    # Filter nan's and inf's
    T_av1 = T_av[~np.isnan(T_av)]
    T_av2 = T_av1[~np.isinf(T_av1)]
    T_av3 = T_av2[T_av2 != 0] 
    min_bin = floor(amin(log10(T_av3)))
    max_bin = ceil(amax(log10(T_av3)))
    mybins_T = exp(linspace(min_bin,max_bin,num=(max_bin-min_bin+1)*sub_bin)*log(10))
    pdf_T,mybins_T,patches=hist(T_av3,bins=mybins_T,\
                                histtype='step',lw=lw,color=color)
    if (normed):
        pdf_T=pdf_T/(size(T_av3)*diff(mybins_T))
        
    return pdf_T,mybins_T

def azavg_fromcart(q,x1,x2,nsample=10):
    """ Return the azimutal average of q
    The output is an array of dimension
    len(x1)-len(x1)/2  """
    
    iavg = np.abs(x1).argmin()#len(x1)/2
    xx = x1[iavg:]
    qavg  = np.zeros(len(xx))
    # Prepare interpolatation
    new_q = RectBivariateSpline(x1,x2,q)
    # Perform interpolation
    for ii,xx in enumerate(xx):
        Cpoints = [(xx*cos(2*pi*v/nsample),xx*sin(2*pi*v/nsample)) for v in range(nsample)]
        tmp = np.zeros(nsample)
        for iis,xyc in enumerate(Cpoints):
            tmp[iis] = new_q(xyc[0],xyc[1])
        qavg[ii] = np.mean(tmp)

    return qavg

def sphavg_from2D(q,x1,x2,nsample=10):
    """ Return the azimutal average of q
    The output is an array of dimension
    len(x1)-len(x1)/2  """
    
    iavg = 0
    xx = x1[0:]
    qavg  = np.zeros(len(xx))
    # Prepare interpolatation
    new_q = RectBivariateSpline(x1,x2,q)
    # Perform interpolation
    for ii,xx in enumerate(xx):
        Cpoints = [(xx*cos(-pi/2.+pi*v/nsample),xx*sin(-pi/2.+pi*v/nsample)) for v in range(nsample)]
        tmp = np.zeros(nsample)
        for iis,xyc in enumerate(Cpoints):
            tmp[iis] = new_q(xyc[0],xyc[1])
        qavg[ii] = np.mean(tmp)

    return qavg

def getazav(D,nsample=10):
    """ Compute the azimutal avgs of all the quantities """

    i_a = np.abs(D.x1).argmin()
    D.n1_a = len(D.x1[i_a:])#D.n1-D.n1/2
    D.rho_a = np.zeros((D.n1_a,D.n3))
    D.prs_a = np.zeros((D.n1_a,D.n3))
    D.vx1_a = np.zeros((D.n1_a,D.n3))
    D.vx2_a = np.zeros((D.n1_a,D.n3))
    D.vx3_a = np.zeros((D.n1_a,D.n3))
    D.bx1_a = np.zeros((D.n1_a,D.n3))
    D.bx2_a = np.zeros((D.n1_a,D.n3))
    D.bx3_a = np.zeros((D.n1_a,D.n3))

    if (not hasattr(D,'Vr')):
        getcyl(D)
        
    D.Vr_a   = np.zeros((D.n1_a,D.n3))
    D.Vphi_a = np.zeros((D.n1_a,D.n3))
    D.Br_a   = np.zeros((D.n1_a,D.n3))
    D.Bphi_a = np.zeros((D.n1_a,D.n3))

    D.va_a = np.zeros((D.n1_a,D.n3))
    D.Alf_a = np.zeros((D.n1_a,D.n3))
        
    # Compute quantities
    for i3,x3 in enumerate(D.x3):
        D.rho_a[:,i3] = azavg_fromcart(D.rho[:,:,i3],D.x1,D.x2,nsample=nsample)
        D.prs_a[:,i3] = azavg_fromcart(D.prs[:,:,i3],D.x1,D.x2,nsample=nsample)
        D.vx1_a[:,i3] = azavg_fromcart(D.vx1[:,:,i3],D.x1,D.x2,nsample=nsample)
        D.vx2_a[:,i3] = azavg_fromcart(D.vx2[:,:,i3],D.x1,D.x2,nsample=nsample)
        D.vx3_a[:,i3] = azavg_fromcart(D.vx3[:,:,i3],D.x1,D.x2,nsample=nsample)
        D.bx1_a[:,i3] = azavg_fromcart(D.bx1[:,:,i3],D.x1,D.x2,nsample=nsample)
        D.bx2_a[:,i3] = azavg_fromcart(D.bx2[:,:,i3],D.x1,D.x2,nsample=nsample)
        D.bx3_a[:,i3] = azavg_fromcart(D.bx3[:,:,i3],D.x1,D.x2,nsample=nsample)

        D.Vr_a[:,i3]   = azavg_fromcart(D.Vr[:,:,i3],D.x1,D.x2,nsample=nsample)
        D.Vphi_a[:,i3] = azavg_fromcart(D.Vphi[:,:,i3],D.x1,D.x2,nsample=nsample)
        D.Br_a[:,i3]   = azavg_fromcart(D.Br[:,:,i3],D.x1,D.x2,nsample=nsample)
        D.Bphi_a[:,i3] = azavg_fromcart(D.Bphi[:,:,i3],D.x1,D.x2,nsample=nsample)

        if not hasattr(D,'va'):
            comp_va_cs(D)
        D.va_a[:,i3]   = azavg_fromcart(D.va[:,:,i3],D.x1,D.x2,nsample=nsample)
        D.Alf_a[:,i3]  = azavg_fromcart(D.Alf[:,:,i3],D.x1,D.x2,nsample=nsample)
    
    # Compute the other quantities
    D.x1_a   = D.x1[i_a:]
    D.dx1_a  = D.dx1[i_a:]
    D.x1r_a  = D.x1r[i_a:]
    D.xx_a,D.yy_a = np.meshgrid(D.x1_a,D.x3) ; D.xx_a = D.xx_a.T ; D.yy_a = D.yy_a.T
    D.Rcyl_a = D.xx_a
    D.Rsph_a = sqrt(D.xx_a**2+D.yy_a**2)

    D.S_a = D.prs_a/(D.rho_a**D.GAMMA)
    D.K_a = D.rho_a*(D.Vr_a*D.Br_a + D.vx3_a*D.bx3_a)/(D.Br_a**2+D.bx3_a**2)
    D.E_a = (D.Vr_a**2 + D.vx3_a**2 - D.Vphi_a**2)/2.0 + \
            D.GAMMA*D.rho_a**(D.GAMMA-1.0)*D.S_a/(D.GAMMA-1.0) + \
            D.Vphi_a*D.Bphi_a*D.K_a/D.rho_a - 1.0/D.Rsph_a
    D.O_a = (D.Vphi_a - D.K_a*D.Bphi_a/D.rho_a)/D.Rcyl_a
    D.L_a = D.Rcyl_a*(D.Vphi_a - D.Bphi_a/D.K_a)

    D.cs_a  = sqrt(D.GAMMA*D.prs_a/D.rho_a)
    D.Mach_a= sqrt(D.vx1_a**2+D.vx2_a**2+D.vx3_a**2)/D.cs_a
    D.va_a  = sqrt(D.bx1_a**2+D.bx2_a**2+D.bx3_a**2)/sqrt(D.rho_a)
    #D.Alf_a = sqrt(D.vx1_a**2+D.vx2_a**2+D.vx3_a**2)/D.va_a

def save_elements(list_e,name):
    #f=file(name,'w')
    f=open(name,'wb')
    for el in list_e:
        pickle.dump(el,f,1)
    f.close()

def read_elements(nb_el,name):
    #f=file(name,'r')
    f=open(name,'rb')
    list_el = []
    for el in range(nb_el):
        try:
            list_el.append(pickle.load(f,fix_imports=True,encoding='latin1'))
        except:
            list_el.append(pickle.load(f))
    f.close()
    return list_el

def temporal_mean(itmin,itmax,w_dir='./',x1range=None,x2range=None,x3range=None,dec=1,save_f=False,dir_save='',level=0,datatype=None,silent=False,suff=''):

    D = pp.pload(itmin,w_dir=w_dir,x1range=x1range,x2range=x2range,x3range=x3range,dec=dec,level=level,datatype=datatype,silent=silent,noload=True)
    varinf = D.vars
    if (dir_save == ''):
        dsave=D.wdir
    else:
        dsave=dir_save
    fname = 'temporal_mean_'+str(itmin)+'_'+str(itmax)+suff
    allvars = D.vars

    if (not os.path.isdir(dsave)) and save_f:
        call('mkdir -p '+dsave,shell=True)

    if (os.path.exists(dsave+'/'+fname) and save_f):
        if not silent:
            print ('Temporal mean already calculated, reading it...')
        listD = read_elements(len(allvars),dsave+'/'+fname)
        for ivar,var in enumerate(allvars):
            #print(var,np.amax(listD[ivar]))
            #exec ("print(%s,%i)",(var,ivar))
            #exec ('D.%s = listD[%i]' % (var,ivar))
            setattr(D,var,listD[ivar])
    else:
        norm = 1.0
        D = pp.pload(itmin,w_dir=w_dir,x1range=x1range,x2range=x2range,x3range=x3range,dec=dec,level=level,datatype=datatype,silent=silent)
        for it in range(itmin+1,itmax+1):
            Dtmp = pp.pload(it,w_dir=D.wdir,x1range=x1range,x2range=x2range,x3range=x3range,level=level,datatype=datatype,silent=silent,dec=dec)
            allvars = Dtmp.vars
            norm = norm + 1.0
            for var in allvars:
                setattr(D,var,getattr(D,var)+getattr(Dtmp,var))
                #exec ('D.%s = D.%s + Dtmp.%s' % (var,var,var))
        for var in allvars:
            setattr(D,var,getattr(D,var)/norm)
            #exec ('D.%s = D.%s/norm' % (var,var))
        if (save_f):
            listD=[]
            for var in allvars:
                listD.append(getattr(D,var))
                #exec ('listD.append(D.%s)' % (var))
            save_elements(listD,dsave+'/'+fname)

    # Cope with older versions
    if hasattr(D,'v1'):
        print ('Warning, old version detected')
        D.vx1 = D.v1
        D.vx2 = D.v2
        D.vx3 = D.v3
        D.prs = D.pr
        if hasattr(D,'b1'):
            D.bx1 = D.b1
            D.bx2 = D.b2
            D.bx3 = D.b3
            if hasattr(D,'B1_bckg'):
                D.bx1 = D.bx1 + D.B1_bckg
                D.bx2 = D.bx2 + D.B2_bckg
                D.bx3 = D.bx3 + D.B3_bckg 

    D.mean_Steps='%i_%i' % (itmin,itmax)
    return D

def get_caseparams(D):
    
    fname = D.wdir+'/pluto.ini'
    if (os.path.exists(fname)):
        lines=[line.strip() for line in open(fname)]
        for ll in lines[lines.index('[Parameters]')+2:]:
            lll = ll.split()
            object.__setattr__(D,lll[0],float(lll[1]))
        # Compute adimensional paramaters
        if hasattr(D,'R_STAR'):
            D.Urho = 1.e-16
            D.Ulen = D.R_STAR*6.9599e10
            GG = 6.6728e-8 ; Msun = 1.98855e33 
            if not hasattr(D,'M_STAR'):
                D.M_STAR=1.0
            D.Uvel = np.sqrt(GG*D.M_STAR*Msun/D.Ulen)
    else:
        print ('Err: I did not find '+D.wdir+'/pluto.ini')
        
    # Get the correct adimentional numbers and whether there is a rotating frame or not
    fname = D.wdir+'/definitions.h'
    if (os.path.exists(fname)):
        lines=[line.strip() for line in open(fname)]
        indxline=lines.index('/* -- physics dependent declarations -- */')+2
        indyline=lines.index('/* -- user-defined parameters (labels) -- */')
        for ll in lines[indxline:indyline-1]:
            lll = ll.split()
            object.__setattr__(D,lll[1],lll[2])        
        indxline=lines.index('/* [Beg] user-defined constants (do not change this line) */')+2
        for ll in lines[indxline:indxline+3]:
            lll = ll.split()
            #print(lll)
            try:
                object.__setattr__(D,lll[1],float(lll[2]))
            #except ValueError:
            except:
                object.__setattr__(D,lll[1],lll[2])
        D.Urho = D.UNIT_DENSITY
        D.Ulen = D.UNIT_LENGTH
        D.Uvel = D.UNIT_VELOCITY

def compute_grids(D,mode='cartesian'):


    if D.n3==1:
        x3=0.
    else:
        x3=D.x3


    D.xx = np.tile( np.tile(D.x1,(D.n2,1))  , (D.n3,1,1)).T.squeeze()
    D.yy = np.tile( np.tile(D.x2,(D.n1,1)).T, (D.n3,1,1)).T.squeeze()
    D.zz = np.tile( np.tile(  x3,(D.n2,1))  , (D.n1,1,1)).squeeze()

    D.dxx = np.tile( np.tile(D.dx1,(D.n2,1))  , (D.n3,1,1)).T.squeeze()
    D.dyy = np.tile( np.tile(D.dx2,(D.n1,1)).T, (D.n3,1,1)).T.squeeze()
    D.dzz = np.tile( np.tile(D.dx3,(D.n2,1))  , (D.n1,1,1)).squeeze()

    if mode == 'cartesian':
        D.Rcyl = sqrt(D.xx**2 + D.yy**2)
        D.Rsph = sqrt(D.xx**2 + D.yy**2 + D.zz**2)
    elif mode == 'cylindrical':
        D.Rcyl = D.xx
        D.Rsph = sqrt(D.xx**2+D.yy**2)
    elif mode == 'spherical':
        D.Rcyl = D.xx*np.sin(D.yy)
        D.Rsph = D.xx
        D.xx1  = D.xx * np.sin(D.yy) * np.cos(D.zz)
        D.xx2  = D.xx * np.sin(D.yy) * np.sin(D.zz)
        D.xx3  = D.xx * np.cos(D.yy) 

    D.cosp = D.xx/D.Rcyl ; D.sinp = D.yy/D.Rcyl
    D.cosp[np.where(D.Rcyl == 0.0)] = 0.0
    D.sinp[np.where(D.Rcyl == 0.0)] = 0.0

    D.x1r = get_lims(D.x1,D.dx1)
    D.x2r = get_lims(D.x2,D.dx2)
    if (D.n3_tot != 1):
        D.x3r = get_lims(D.x3,D.dx3)
    else:
        D.x3r = D.x3

    n1r = D.n1+1 if D.n1 > 1 else 1
    n2r = D.n2+1 if D.n2 > 1 else 1
    n3r = D.n3+1 if D.n3 > 1 else 1
    x1r = D.x1r if D.n1 > 1 else D.x1
    x2r = D.x2r if D.n2 > 1 else D.x2
    x3r = D.x3r if D.n3 > 1 else D.x3

    if D.n3 == 1:
        x3r=np.array([0])

    D.xxr = np.tile( np.tile(x1r,(n2r,1))  , (n3r,1,1)).T.squeeze()
    D.yyr = np.tile( np.tile(x2r,(n1r,1)).T, (n3r,1,1)).T.squeeze()
    D.zzr = np.tile( np.tile(x3r,(n2r,1))  , (n1r,1,1)).squeeze()

    if mode == 'spherical':
        D.xXr  = D.xxr * np.sin(D.yyr) * np.cos(D.zzr)
        D.xYr  = D.xxr * np.sin(D.yyr) * np.sin(D.zzr)
        D.xZr  = D.xxr * np.cos(D.yyr) 
        D.xX  =  D.xx * np.sin(D.yy) * np.cos(D.zz)
        D.xY  =  D.xx * np.sin(D.yy) * np.sin(D.zz)
        D.xZ  =  D.xx * np.cos(D.yy) 
        D.dxX =  D.dxx* np.sin(D.dyy)* np.cos(D.dzz)
        D.dxY =  D.dxx* np.sin(D.dyy)* np.sin(D.dzz)
        D.dxZ =  D.dxx* np.cos(D.dyy) 

def SphericalToCartesian(D,vec=""):
    if not hasattr(D,"xx1"):
        compute_grids(D,mode='spherical')

    tx1=getattr(D,"{}x1".format(vec))
    tx2=getattr(D,"{}x2".format(vec))
    tx3=getattr(D,"{}x3".format(vec))
        
    setattr(D,vec+'X',np.sin(D.yy)*np.cos(D.zz)*tx1 + np.cos(D.yy)*np.cos(D.zz)*tx2 - np.sin(D.zz)*tx3)
    setattr(D,vec+'Y',np.sin(D.yy)*np.sin(D.zz)*tx1 + np.cos(D.yy)*np.sin(D.zz)*tx2 + np.cos(D.zz)*tx3)
    setattr(D,vec+'Z',             np.cos(D.yy)*tx1 -              np.sin(D.yy)*tx2)

    #exec("D.{}X = np.sin(D.yy)*np.cos(D.zz)*D.{}x1 + np.cos(D.yy)*np.cos(D.zz)*D.{}x2 - np.sin(D.zz)*D.{}x3".format(vec,vec,vec,vec))
    #exec("D.{}Y = np.sin(D.yy)*np.sin(D.zz)*D.{}x1 + np.cos(D.yy)*np.sin(D.zz)*D.{}x2 + np.cos(D.zz)*D.{}x3".format(vec,vec,vec,vec))
    #exec("D.{}Z =              np.cos(D.yy)*D.{}x1 -              np.sin(D.yy)*D.{}x2                      ".format(vec,vec,vec))


def compute_grids_p(D):

    if not hasattr(D,'R_ORBIT_PLANET'):
        get_caseparams(D)
    if D.R_ORBIT_PLANET == 0:
        print('Nothing to be done here...')
        return

    D.xx_p = np.tile( np.tile(D.x1-D.R_ORBIT_PLANET,(D.n2,1))  , (D.n3,1,1)).T#.squeeze()

    if (D.n3_tot != 1):
        D.Rcyl_p = sqrt(D.xx_p**2 + D.yy**2)
        D.Rsph_p = sqrt(D.xx_p**2 + D.yy**2 + D.zz**2)
    else:
        D.Rcyl_p = D.xx_p
        D.Rsph_p = sqrt(D.xx_p**2+D.yy**2)

    D.cosp_p = D.xx_p/D.Rcyl_p ; D.sinp_p = D.yy/D.Rcyl_p
    D.cosp_p[np.where(D.Rcyl_p == 0.0)] = 0.0
    D.sinp_p[np.where(D.Rcyl_p == 0.0)] = 0.0

    D.x1r_p = get_lims(D.x1-D.R_ORBIT_PLANET,D.dx1)
    if (D.n3_tot != 1):
        D.x3r = get_lims(D.x3,D.dx3)
    else:
        D.x3r = D.x3

    D.xxr_p = np.tile( np.tile(D.x1r_p,(D.n2+1,1))  , (D.n3+1,1,1)).T#.squeeze()

def derivAS(A,lx,order=1):
    """ This routines intends to perform the derivatives of A along each of its dimension using any
    grid specified in lx. lx is a list of 1d grids, of length ndim(A). Only first derivative coded for now  """
    if (np.ndim(A) != len(lx)):
        print(">> Problem in derivAS: lx should have as many items than ndim(A)")
        return
    for ii,xx in enumerate(lx):
        if (np.shape(A)[ii] != len(xx)):
            print(">> Problem in derivAS: dimension %i does not match" % (ii))
            return

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

def derivPyPl(Y,X=None):
    """
    Calculates the derivative of Y with respect to X.
    
    **Inputs:**
    
    Y : 1-D array to be differentiated.\n
    X : 1-D array with len(X) = len(Y).\n
    
    If X is not specified then by default X is chosen to be an equally spaced array having same number of elements
    as Y.
    
    **Outputs:**
    
    This returns an 1-D array having the same no. of elements as Y (or X) and contains the values of dY/dX.
    
    """
    n = len(Y)
    n2 = n-2
    if X==None : X = np.arange(n)
    Xarr = np.asarray(X,dtype='float')
    Yarr = np.asarray(Y,dtype='float')
    x12 = Xarr - np.roll(Xarr,-1)   #x1 - x2
    x01 = np.roll(Xarr,1) - Xarr    #x0 - x1
    x02 = np.roll(Xarr,1) - np.roll(Xarr,-1) #x0 - x2
    DfDx = np.roll(Yarr,1) * (x12 / (x01*x02)) + Yarr * (1./x12 - 1./x01) - np.roll(Yarr,-1) * (x01 / (x02 * x12))
    # Formulae for the first and last points:
    DfDx[0] = Yarr[0] * (x01[1]+x02[1])/(x01[1]*x02[1]) - Yarr[1] * x02[1]/(x01[1]*x12[1]) + Yarr[2] * x01[1]/(x02[1]*x12[1])
    DfDx[n-1] = -Yarr[n-3] * x12[n2]/(x01[n2]*x02[n2]) + Yarr[n-2]*x02[n2]/(x01[n2]*x12[n2]) - Yarr[n-1]*(x02[n2]+x12[n2])/(x02[n2]*x12[n2])
    
    return DfDx

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


def plot_grid(D,r1=32,r2=32,psf='None'):
    x1 = D.x1r
    x2 = D.x2r
    load_j('None')
    fig=figure(figsize=(5,5))
    for i in range(0,len(x1)):
        if ((i % r1) == 0):
            axvline(x1[i],color='k',alpha=0.5)
        else:
            axvline(x1[i],color='k',alpha=0.1)
    for i in range(0,len(x2)):
        if ((i % r2) == 0):
            axhline(x2[i],color='k',alpha=0.5)
        else:
            axhline(x2[i],color='k',alpha=0.1)

    #title(r'Grid (highlighted every %i points)' % (r1))
    title(r'Grid')
    xlabel(r'$\varpi/r_\star$')
    ylabel(r'$z/r_\star$')
    xlim(x1[0],x1[-1])
    ylim(x2[0],x2[-1])

    # zoom 1
    ax=gca()
    axins = zoomed_inset_axes(ax,40,loc=1)
    for i in range(0,len(x1)):
        if ((i % 8) == 0):
            axins.axvline(x1[i],color='k',alpha=1.0)
        else:
            axins.axvline(x1[i],color='k',alpha=0.1)
    for i in range(0,len(x2)):
        if ((i % 8) == 0):
            axins.axhline(x2[i],color='k',alpha=1.0)
        else:
            axins.axhline(x2[i],color='k',alpha=0.1)
    axins.set_xlim([2.8,3.2])
    axins.set_ylim([-0.2,0.2])
    mark_inset(ax,axins,loc1=2,loc2=4,fc='w',ec='b',lw=2,alpha=0.7)
    rect=TransformedBbox(axins.viewLim,axins.transData)
    circle=Circle((3.,0),0.1,fc='None',ec='b',zorder=4)
    axins.add_artist(circle)
    rex=Rectangle((2.8,-0.2),0.4,0.4,fc='None',ec='b',lw=2,zorder=4,alpha=0.7);ax.add_artist(rex)
    p = BboxPatch(rect,fc='w')
    axins.add_patch(p)
    axins.get_xaxis().set_visible(False)
    axins.get_yaxis().set_visible(False)
    axins.annotate('Planet boundary',xy=(0.5,0.9),xycoords='axes fraction',\
                   color='b',zorder=5,ha='center',va='center',\
                   bbox=dict(boxstyle='round',fc='white',ec='None'))
    draw()

    axins = zoomed_inset_axes(ax,15,loc=8)
    for i in range(0,len(x1)):
        if ((i % 8) == 0):
            axins.axvline(x1[i],color='k',alpha=1.0)
        else:
            axins.axvline(x1[i],color='k',alpha=0.2)
    for i in range(0,len(x2)):
        if ((i % 8) == 2):
            axins.axhline(x2[i],color='k',alpha=1.0)
        else:
            axins.axhline(x2[i],color='k',alpha=0.2)
    axins.set_xlim([0,1.1])
    axins.set_ylim([-1.1,0])
    mark_inset(ax,axins,loc1=3,loc2=1,fc='w',ec='r',lw=2,alpha=0.7)
    rect=TransformedBbox(axins.viewLim,axins.transData)
    circle=Circle((0.,0),1.0,fc='None',ec='r')
    axins.add_artist(circle)
    rex=Rectangle((0.,-1.1),1.1,1.1,fc='None',ec='r',lw=2,zorder=4,alpha=0.7);ax.add_artist(rex)
    p = BboxPatch(rect,fc='w')
    axins.add_patch(p)
    axins.get_xaxis().set_visible(False)
    axins.get_yaxis().set_visible(False)
    axins.annotate('Stellar boundary',xy=(0.42,0.8),xycoords='axes fraction',\
                   color='r',zorder=5,ha='center',va='center',\
                   bbox=dict(boxstyle='round',fc='white',ec='None'))
    draw()

    if (psf != 'None'):
        fig.savefig(psf,transparent=True,bbox_inches='tight')

def plot_grid2(D,r1=32,r2=32,psf='None'):
    x1 = D.x1r
    x2 = D.x2r
    load_j('None')
    fig=figure(figsize=(8,8))
    subplot(222)
    for i in range(0,len(x1)):
        if ((i % r1) == 0):
            axvline(x1[i],color='k',alpha=0.5)
        else:
            axvline(x1[i],color='k',alpha=0.1)
    for i in range(0,len(x2)):
        if ((i % r2) == 0):
            axhline(x2[i],color='k',alpha=0.5)
        else:
            axhline(x2[i],color='k',alpha=0.1)

    #title(r'Grid (highlighted every %i points)' % (r1))
    #suptitle(r'Grid')
    xlabel(r'$\varpi/r_\star$')
    ylabel(r'$z/r_\star$')
    xlim(x1[0],x1[-1])
    ylim(x2[0],x2[-1])

    # zoom 1
    ax=gca()
    axins = zoomed_inset_axes(ax,100,loc=4,bbox_to_anchor=(0.92,0.1),bbox_transform=ax.figure.transFigure)
    for i in range(0,len(x1)):
        if ((i % 8) == 0):
            axins.axvline(x1[i],color='k',alpha=1.0)
        else:
            axins.axvline(x1[i],color='k',alpha=0.1)
    for i in range(0,len(x2)):
        if ((i % 8) == 0):
            axins.axhline(x2[i],color='k',alpha=1.0)
        else:
            axins.axhline(x2[i],color='k',alpha=0.1)
    axins.set_xlim([2.8,3.2])
    axins.set_ylim([-0.2,0.2])
    mark_inset(ax,axins,loc1=2,loc2=1,fc='none',ec='b',lw=2,alpha=0.7)
    rect=TransformedBbox(axins.viewLim,axins.transData)
    circle=Circle((3.,0),0.1,fc='None',ec='b',zorder=4);axins.add_artist(circle)
    rex=Rectangle((2.8,-0.2),0.4,0.4,fc='None',ec='b',lw=2,zorder=4,alpha=0.7);ax.add_artist(rex)
    p = BboxPatch(rect,fc='w',ec='b')
    axins.add_patch(p)
    axins.set_xlabel(r'$\varpi/r_\star$')
    axins.set_ylabel(r'$z/r_\star$')
    #axins.get_xaxis().set_visible(False)
    #axins.get_yaxis().set_visible(False)
    axins.annotate('Planet boundary',xy=(0.5,0.9),xycoords='axes fraction',\
                   color='b',zorder=5,ha='center',va='center',\
                   bbox=dict(boxstyle='round',fc='white',ec='None'))
    draw()

    axins = zoomed_inset_axes(ax,42,loc=4,bbox_to_anchor=(0.5,0.1),bbox_transform=ax.figure.transFigure)
    for i in range(0,len(x1)):
        if ((i % 8) == 0):
            axins.axvline(x1[i],color='k',alpha=1.0)
        else:
            axins.axvline(x1[i],color='k',alpha=0.2)
    for i in range(0,len(x2)):
        if ((i % 8) == 2):
            axins.axhline(x2[i],color='k',alpha=1.0)
        else:
            axins.axhline(x2[i],color='k',alpha=0.2)
    axins.set_xlim([0,1.1])
    axins.set_ylim([-1.1,1.1])
    mark_inset(ax,axins,loc1=4,loc2=1,fc='none',ec='r',lw=2,alpha=0.7)
    rect=TransformedBbox(axins.viewLim,axins.transData)
    circle=Circle((0.,0),1.0,fc='None',ec='r',zorder=4);axins.add_artist(circle)
    rex=Rectangle((0.,-1.1),1.1,2.2,fc='None',ec='r',lw=2,zorder=4,alpha=0.7);ax.add_artist(rex)
    #p = BboxPatch(rect,fc='w',ec='r')
    #axins.add_patch(p)
    axins.set_xlabel(r'$\varpi/r_\star$')
    axins.set_ylabel(r'$z/r_\star$')
    #axins.get_xaxis().set_visible(False)
    #axins.get_yaxis().set_visible(False)
    axins.annotate('Stellar boundary',xy=(0.42,0.8),xycoords='axes fraction',\
                   color='r',zorder=5,ha='center',va='center',\
                   bbox=dict(boxstyle='round',fc='white',ec='None'))
    draw()

    if (psf != 'None'):
        fig.savefig(psf,transparent=True,bbox_inches='tight')

def getcyl(D):
    # Get cylindrical coordinate components of the vectors
    if (not hasattr(D,'xx')):
        compute_grids(D)
        
    if (D.n3_tot != 1):
        D.Vr   =  (D.cosp * D.vx1.squeeze() + D.sinp * D.vx2.squeeze()).reshape(D.vx2.shape)
        D.Vphi = (-D.sinp * D.vx1.squeeze() + D.cosp * D.vx2.squeeze()).reshape(D.vx2.shape)
        if (hasattr(D,'bx1')):
            D.Br   = ( D.cosp * D.bx1.squeeze() + D.sinp * D.bx2.squeeze()).reshape(D.bx2.shape)
            D.Bphi = (-D.sinp * D.bx1.squeeze() + D.cosp * D.bx2.squeeze()).reshape(D.bx2.shape)
    else:
        D.Vr   =  D.vx1 
        D.Vphi =  D.vx3
        if (hasattr(D,'bx1')):
            D.Br   =  D.bx1
            D.Bphi =  D.bx3

def ComputeGrad(x1,x2,x3,var,geom='cartesian'):
    if geom == 'cartesian':
        deriv_x1 = nufd1(x1) ; deriv_x2 = nufd1(x2) ; deriv_x3 = nufd1(x3)
        dx1V = np.zeros_like(var) ; dx2V = np.zeros_like(var) ; dx3V = np.zeros_like(var)
        for ix,xx in enumerate(x1):
            for iz,zz in enumerate(x3):
                dx2V[ix,:,iz] = deriv_x2*var[ix,:,iz]
            for iy,yy in enumerate(x2):
                dx3V[ix,iy,:] = deriv_x3*var[ix,iy,:]
        for iy,yy in enumerate(x2):
            for iz,zz in enumerate(x3):
                dx1V[:,iy,iz] = deriv_x1*var[:,iy,iz]
    elif geom == 'spherical':
        deriv_r = nufd1(x1) ; deriv_t = nufd1(x2) ; deriv_p = nufd1(x3)
        dx1V=np.zeros_like(var); dx2V=np.zeros_like(var); dx3V=np.zeros_like(var);
        if len(x3) == 1:
            for ix,xx in enumerate(x1):
                dx2V[ix,:] = (deriv_t*var[ix,:])/xx
            for iy,yy in enumerate(x2):
                dx1V[:,iy] = (deriv_r*var[:,iy])/(x1*x1)
        else:
            for ix,xx in enumerate(x1):
                for iz,zz in enumerate(x3):
                    dx2V[ix,:,iz] = (deriv_t*var[ix,:,iz])/xx
                for iy,yy in enumerate(x2):
                    dx3V[ix,iy,:] = (deriv_p*var[ix,iy,:])/(xx*np.sin(yy))
            for iy,yy in enumerate(x2):
                for iz,zz in enumerate(x3):
                    dx1V[:,iy,iz] = (deriv_r*var[:,iy,iz])/(x1*x1)
    else:
        print('Gradient for '+geom+' not coded yet')

    return [dx1V,dx2V,dx3V]

def ComputeDiv(x1,x2,x3,var1,var2,var3,geom='cartesian'):
    if geom == 'cartesian':
        deriv_x1 = nufd1(x1) ; deriv_x2 = nufd1(x2) ; deriv_x3 = nufd1(x3)
        div=np.zeros_like(var1)
        if len(x3) == 1:
            for ix,xx in enumerate(x1):
                div[ix,:] += deriv_x2*var2[ix,:]
            for iy,yy in enumerate(x2):
                div[:,iy] += deriv_x1*var1[:,iy]
        else:
            for ix,xx in enumerate(x1):
                for iz,zz in enumerate(x3):
                    div[ix,:,iz] += deriv_x2*var2[ix,:,iz]
                for iy,yy in enumerate(x2):
                    div[ix,iy,:] += deriv_x3*var3[ix,iy,:]
            for iy,yy in enumerate(x2):
                for iz,zz in enumerate(x3):
                    div[:,iy,iz] += deriv_x1*var1[:,iy,iz]
    elif geom == 'spherical':
        deriv_r = nufd1(x1) ; deriv_t = nufd1(x2) ; deriv_p = nufd1(x3)
        div=np.zeros_like(var1)
        if len(x3) == 1:
            for ix,xx in enumerate(x1):
                tmp=np.sin(x2)*var2[ix,:]
                div[ix,:] += (deriv_t*(tmp))/(xx*np.sin(x2))
            for iy,yy in enumerate(x2):
                tmp=x1*x1*var1[:,iy]
                div[:,iy] += (deriv_r*(tmp))/(x1*x1)
        else:
            for ix,xx in enumerate(x1):
                for iz,zz in enumerate(x3):
                    div[ix,:,iz] += (deriv_t*(np.sin(x2)*var2[ix,:,iz]))/(xx*np.sin(x2))
                for iy,yy in enumerate(x2):
                    div[ix,iy,:] += (deriv_p*(var3[ix,iy,:]))/(xx*np.sin(yy))
            for iy,yy in enumerate(x2):
                for iz,zz in enumerate(x3):
                    div[:,iy,iz] += (deriv_r*(x1*x1*var1[:,iy,iz]))/(x1*x1)
    else:
        print('Divergence for '+geom+' not coded yet')
    return div

def compute_Current(D,version=0):
    D.Jx1 = np.zeros_like(D.bx1)
    D.Jx2 = np.zeros_like(D.bx2)
    D.Jx3 = np.zeros_like(D.bx3)
    if version == 0:
        # Preload derivatives to accelerate the process
        deriv_x1 = nufd1(D.x1) ; deriv_x2 = nufd1(D.x2) ; deriv_x3 = nufd1(D.x3)
        for ix,xx in enumerate(D.x1):
            for iy,yy in enumerate(D.x2):
                D.Jx1[ix,iy,:] -= deriv_x3*D.bx2[ix,iy,:]
                D.Jx2[ix,iy,:] += deriv_x3*D.bx1[ix,iy,:]
        for ix,xx in enumerate(D.x1):
            for iz,zz in enumerate(D.x3):
                D.Jx1[ix,:,iz] += deriv_x2*D.bx3[ix,:,iz]
                D.Jx3[ix,:,iz] -= deriv_x2*D.bx1[ix,:,iz]
        for iy,yy in enumerate(D.x2):
            for iz,zz in enumerate(D.x3):
                D.Jx2[:,iy,iz] -= deriv_x1*D.bx3[:,iy,iz]
                D.Jx3[:,iy,iz] += deriv_x1*D.bx2[:,iy,iz]
    else:
        # Requires numpy > 1.13.1
        D.Jx1 = np.gradient(D.bx3,D.x2,axis=1) - np.gradient(D.bx2,D.x3,axis=2)
        D.Jx2 = np.gradient(D.bx1,D.x3,axis=2) - np.gradient(D.bx3,D.x1,axis=0)
        D.Jx3 = np.gradient(D.bx2,D.x1,axis=0) - np.gradient(D.bx1,D.x2,axis=1)

def see_currents(D,version=0,recompute=False,cut=0,mini=0.,maxi=1.,psfile='None'):

    suffplots=['x1','x2','x3','J']
    if not hasattr(D,'Jx1') or recompute:
        compute_Current(D,version=version)
    if not hasattr(D,'JJ'):
        D.JJ = np.sqrt(D.Jx1**2+D.Jx2**2+D.Jx3**2)
    
    if cut==0:
        ix=np.abs(D.x1-0.5*(D.x1[0]+D.x1[-1])).argmin()
        x1=D.x1 ; x2=D.x3
    elif cut==1:
        iy=np.abs(D.x2-0.5*(D.x2[0]+D.x2[-1])).argmin()
        x1=D.x1 ; x2=D.x3
    elif cut==2:
        iz=np.abs(D.x3-0.5*(D.x3[0]+D.x3[-1])).argmin()
        x1=D.x1 ; x2=D.x2

    fig=figure(figsize=(5,5))
    for isuf,sufp in enumerate(suffplots):
        subplot(2,2,1+isuf)
        ax=gca()
        title(sufp)
        tmpplot=getattr(D,"J{}".format(sufp))
        if cut==0:
            toplot=tmpplot[ix,:,:]
        elif cut==1:
            toplot=tmpplot[:,iy,:]
        elif cut==2:
            toplot=tmpplot[:,:,iz]
        ax.pcolorfast(x1,x2,np.log10(np.abs(toplot)).T,cmap='viridis',vmin=mini,vmax=maxi)
        #exec("ax.pcolorfast(x1,x2,np.log10(np.abs(D.J{}{})).T,cmap='viridis',vmin=mini,vmax=maxi)".format(sufp,cuts))
        colorbarAS('viridis',mini,maxi)
    fig.tight_layout()
    if psfile != 'None':
        fig.savefig(psfile,bbox_inches='tight')

    fig=figure(figsize=(4,3))
    pdfs=[]
    for isuf,sufp in enumerate(suffplots):
        tmpplot=getattr(D,"J{}".format(sufp))
        if cut==0:
            toplot=tmpplot[ix,:,:]
        elif cut==1:
            toplot=tmpplot[:,iy,:]
        elif cut==2:
            toplot=tmpplot[:,:,iz]        
        pdf,bins = my_log_pdf(np.abs(toplot).ravel())
        #exec("pdf,bins = my_log_pdf(np.abs(D.J{}{}).ravel())".format(sufp,cuts))
        cbins=(bins[0:-1]+bins[1:])/2.0
        pdfs.append([cbins,pdf])
    clf()
    for isuf,sufp in enumerate(suffplots):
        step(pdfs[isuf][0],pdfs[isuf][1]/pdfs[isuf][1].max(),label='J'+sufp)
    #axvline((D.x1[1]-D.x1[0]),color='k',ls=':')
    xscale('log')
    legend()
    ofig='.'.join(psfile.split('.')[:-1])+'_PDF.'+psfile.split('.')[-1]
    fig.savefig(ofig,bbox_inches='tight')

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

def colorbarAS(mycmap,mini,maxi,size="5%",height="100%",pad=0.1,orientation='vertical',format="%.2f",position="right",ticks=[],label='',nticks=0):
    ax=gca() 
    locbounds0=ax._position.bounds
    xmin = locbounds0[0]+locbounds0[2]+pad
    xd   = float(size.strip('%'))*locbounds0[2]/100.
    yd   = float(height.strip('%'))*locbounds0[3]/100.
    ymin = locbounds0[1] + (locbounds0[3]-yd)*0.5
    cax = gcf().add_axes([xmin,ymin,xd,yd])
    norm = mpl.colors.Normalize(vmin=mini,vmax=maxi)
    if len(ticks)==0:
        if nticks != 0:
            ticks=list(np.linspace(mini,maxi,num=nticks))
        else:
            ticks=[mini,0.,maxi]
    cb1=mpl.colorbar.ColorbarBase(cax,cmap=mycmap,norm=norm,orientation=orientation,format=format,ticks=ticks)
    cb1.set_label(label)
    return cb1
    
def remove_rotframe(D,silent=True):
    if (not hasattr(D,'xx')):
        compute_grids(D)
    if (not hasattr(D,'Vr')):
        getcyl(D)
    if (not hasattr(D,'R_ORBIT_PLANET')):
        get_caseparams(D)
        if not hasattr(D,'R_ORBIT_PLANET'):
            D.R_ORBIT_PLANET=0
    if (not hasattr(D,'rot_rem')):
        D.rot_rem = False
    if (not D.rot_rem):
        if (D.R_ORBIT_PLANET != 0.):
            if (not silent):
                print('Case: '+D.wdir)
                print('Rotating frame with rorb= '+str(D.R_ORBIT_PLANET)+' removed!')
            D.Vphi = D.Vphi + D.Rcyl/(D.R_ORBIT_PLANET**1.5)
            if (D.n3_tot != 1):
                D.vx1  = D.cosp * D.Vr - D.sinp * D.Vphi
                D.vx2  = D.sinp * D.Vr + D.cosp * D.Vphi
            else:
                D.vx3 = D.Vphi
        else:
            if hasattr(D,'VphiStar_VESC'):
                OmStar=D.VphiStar_VESC*sqrt(2.)
            else:
                OmStar=D.VROT_VESC*sqrt(2.)
            if (not silent):
                print('Case: '+D.wdir)
                print('Rotating frame with Om={} removed!'.format(OmStar))
            D.Vphi = D.Vphi + D.Rcyl.squeeze()*OmStar
            if (D.n3_tot != 1):
                D.vx1  = D.cosp * D.Vr - D.sinp * D.Vphi
                D.vx2  = D.sinp * D.Vr + D.cosp * D.Vphi
            else:
                D.vx3 = D.Vphi
        D.rot_rem=True
        # Update things that have already been computed
        if (hasattr(D,'Alf')):
            if (not silent):
                print('Recomputing the characteristic velocities as well...')
            comp_va_cs(D)
        if (hasattr(D,'K')):
            if (not silent):
                print('Recomputing the conserved quantities as well...')
            compute_conserved_quantities(D)
    else:
        print('Nothing to do here')

def add_colorbar(sca,cmap='Spectral_r',vmin=0,vmax=1,label='',pos='right',size_cb='5%',format="%.0f",pad_cb=0.1,orientation='vertical'):
    # Add a colorbar next to the plot
    dd = make_axes_locatable(sca)
    cax = dd.append_axes(pos,size=size_cb,pad=pad_cb)
    norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    cb1=mpl.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm,orientation=orientation,format=format)
    tloc = mpl.ticker.MaxNLocator(nbins=8) ; cb1.locator=tloc ; cb1.update_ticks()
    cb1.set_label(label)

def add_rotframe(D,silent=True):
    if (not hasattr(D,'xx')):
        compute_grids(D)
    if (not hasattr(D,'Vr')):
        getcyl(D)
    if (not hasattr(D,'R_ORBIT_PLANET')):
        get_caseparams(D)
    if (not hasattr(D,'rot_rem')):
        print('The rotating frame was not previously removed')
        return
    if (D.rot_rem):
        if (D.R_ORBIT_PLANET != 0.):
            if (not silent):
                print('Case: '+D.wdir)
                print('Rotating frame with rorb= '+str(D.R_ORBIT_PLANET)+' added!')
            D.Vphi = D.Vphi - D.Rcyl/(D.R_ORBIT_PLANET**1.5)
        else:
            if (not silent):
                print('Case: '+D.wdir)
                print('Rotating frame with Om= '+str(D.VphiStar_VESC*sqrt(2.))+' added!')
            D.Vphi = D.Vphi - D.Rcyl*D.VphiStar_VESC*sqrt(2.)
        if (D.n3_tot != 1):
            D.vx1  = D.cosp * D.Vr - D.sinp * D.Vphi
            D.vx2  = D.sinp * D.Vr + D.cosp * D.Vphi
        else:
            D.vx3 = D.Vphi
        D.rot_rem=False
        # Update things that have already been computed
        if (hasattr(D,'Alf')):
            if (not silent):
                print('Recomputing the characteristic velocities as well...')
            comp_va_cs(D)
        if (hasattr(D,'K')):
            if (not silent):
                print('Recomputing the conserved quantities as well...')
            compute_conserved_quantities(D)
    else:
        if (not silent):
            print('Nothing to do here')

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
    
def comp_va_cs(D):
    if (not hasattr(D,'GAMMA')):
        get_caseparams(D)
        if (not hasattr(D,'GAMMA')):
            print (':: Warning, using hardcoded gamma = 1.05')
            D.GAMMA = 1.05
    D.cs  = sqrt(D.GAMMA*D.prs/D.rho)
    D.Mach= sqrt(D.vx1**2+D.vx2**2+D.vx3**2)/D.cs
    D.va  = sqrt(D.bx1**2+D.bx2**2+D.bx3**2)/sqrt(D.rho)
    D.Alf = sqrt(D.vx1**2+D.vx2**2+D.vx3**2)/D.va
    if (D.n3_tot == 1):
        D.va_pol = sqrt(D.bx1**2+D.bx2**2)/sqrt(D.rho)
        D.fast_Alf = sqrt(2*(D.vx1**2+D.vx2**2)/(D.cs**2+D.va**2+sqrt((D.cs**2+D.va**2)**2-4*(D.cs**2)*(D.va_pol**2)) ) )


def addstar(color='k',rs=1,fc='None',zorder=2):
    circle=plt.Circle((0,0),rs,fc=fc,ec=color,zorder=zorder)
    fig=plt.gcf()
    fig.gca().add_artist(circle)

def hidestar(color='w',rs=1):
    circle=plt.Circle((0,0),rs,fc=color,ec='None')
    fig=plt.gcf()
    fig.gca().add_artist(circle)

def addplanet(color='k',fc='None',rorb=3,rp=0.1,ls='solid',lw=1,transpose=False,zorder=2):
    if (transpose):
        circle=plt.Circle((0,rorb),rp,fc=fc,ec=color,linestyle=ls,linewidth=lw,zorder=zorder)
    else:
        circle=plt.Circle((rorb,0),rp,fc=fc,ec=color,linestyle=ls,linewidth=lw,zorder=zorder)
    fig=plt.gcf()
    fig.gca().add_artist(circle)

def addcircle(color='k',center=(0,0),rp=1.,ls='solid',lw=1):
    circle=plt.Circle(center,rp,fc='None',ec=color,linestyle=ls,linewidth=lw)
    fig=plt.gcf()
    fig.gca().add_artist(circle)

def hideplanet(color='w',rorb=3,rp=0.1):
    circle=plt.Circle((rorb,0),rp,fc=color,ec='None')
    fig=plt.gcf()
    fig.gca().add_artist(circle)

def ind_nearest(array,value):
    return (np.abs(array-value)).argmin()

def compute_flux(qx,qy,qz,D,r,take_abs=False,int_Sph=False):
    "Computes the flux of vector q through a cube of length 2*r centered on the origin"
    if (int_Sph):
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
                x = r*sin(tt) ; y = r*cos(tt)
                qx_l = qx_i((x,y)) ; qy_l = qy_i((x,y))
                tmp_q[it] = sin(tt)*qx_l + cos(tt)*qy_l
                tmp_q[it] = tmp_q[it] * r**2 * sin(tt) * 2.*np.pi

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
                qx_i = RegularGridInterpolator((D.x1,D.x2,D.x3),qx)
                qy_i = RegularGridInterpolator((D.x1,D.x2,D.x3),qy)
                qz_i = RegularGridInterpolator((D.x1,D.x2,D.x3),qz)
            else:
                print('I need a more recent version of scipy to trace field lines in 3D')
                return 0
        
            tmp_q = np.zeros((ntt,npp))
            for it,tt in enumerate(tts):
                for ip,pp in enumerate(pps):
                    x = r*sin(tt)*cos(pp) ; y = r*sin(tt)*sin(pp) ; z =r*cos(tt)
                    qx_l = qx_i((x,y,z)) ; qy_l = qy_i((x,y,z)) ; qz_l = qz_i((x,y,z)) ; 
                    tmp_q[it,ip] = sin(tt)*cos(pp)*qx_l + sin(tt)*sin(pp)*qy_l + cos(tt)*qz_l
                    tmp_q[it,ip] = tmp_q[it,ip] * r**2 * sin(tt)

            # Now perform the integration
            if(take_abs):
                tmp_q = np.abs(tmp_q)
            flux=np.trapz(np.trapz(tmp_q,x=tts,axis=0),x=pps)
            return flux
    else:

        coeff = -1.0
        if (D.n3_tot != 1):
            # Integrate over cubes to avoid interpolation
            indx = r_[ind_nearest(D.x1,-r):ind_nearest(D.x1,r)+1]
            indy = r_[ind_nearest(D.x2,-r):ind_nearest(D.x2,r)+1]
            indz = r_[ind_nearest(D.x3,-r):ind_nearest(D.x3,r)+1]

            qxa = coeff*qx[indx[ 0],:,:].squeeze() ; qxb = qx[indx[-1],:,:].squeeze()
            qya = coeff*qy[:,indy[ 0],:].squeeze() ; qyb = qy[:,indy[-1],:].squeeze()
            qza = coeff*qz[:,:,indz[ 0]].squeeze() ; qzb = qz[:,:,indz[-1]].squeeze()
        
            if (take_abs):
                qxa = np.abs(qxa) ; qxb = np.abs(qxb)
                qya = np.abs(qxa) ; qyb = np.abs(qxb)
                qza = np.abs(qxa) ; qzb = np.abs(qxb)

            flux = np.trapz(np.trapz(qxa[indy,:],x=D.x2[indy],axis=0)[indz],x=D.x3[indz]) + \
                np.trapz(np.trapz(qxb[indy,:],x=D.x2[indy],axis=0)[indz],x=D.x3[indz]) + \
                np.trapz(np.trapz(qya[indx,:],x=D.x1[indx],axis=0)[indz],x=D.x3[indz]) + \
                np.trapz(np.trapz(qyb[indx,:],x=D.x1[indx],axis=0)[indz],x=D.x3[indz]) + \
                np.trapz(np.trapz(qza[indx,:],x=D.x1[indx],axis=0)[indy],x=D.x2[indy]) + \
                np.trapz(np.trapz(qzb[indx,:],x=D.x1[indx],axis=0)[indy],x=D.x2[indy])
        else:
            # Integrate over cylinders to avoid interpolation
            indx = r_[0:ind_nearest(D.x1,r)+1]
            indy = r_[ind_nearest(D.x2,-r):ind_nearest(D.x2,r)+1]

            qxb = qx[indx[-1],:]*D.x1[indx[-1]]
            qya = coeff*D.x1*qy[:,indy[ 0]] ; qyb = D.x1*qy[:,indy[-1]]

            if (take_abs):
                qxb = np.abs(qxb)
                qya = np.abs(qxa) ; qyb = np.abs(qxb)

            flux = ( np.trapz(qxb[indy],x=D.x2[indy],axis=0) + \
                         np.trapz(qya[indx],x=D.x1[indx],axis=0) + \
                         np.trapz(qyb[indx],x=D.x1[indx],axis=0))*2.0*pi
            
        return flux

def compute_jdot_sph(D,rad,recompute=False,suff='',silent=True):
    "Compute the angular momentum loss rate"
    if not os.path.isdir(D.wdir+'/Data'):
        call("mkdir -p "+D.wdir+'/Data',shell=True)

    filename=D.wdir+'/Data/jdot_saved'+suff
    if os.path.exists(filename) and (not recompute):
        if not silent:
            print ('Warning, reading previously computed Jdot')
        [D.rad_j,D.jdot_r] = read_elements(2,filename)
    else:
        D.rad_j = rad
        D.jdot_r = np.zeros_like(rad)

        if not hasattr(D,'Rcyl'):
            compute_grids(D)

        D.K = D.rho*(D.vx1*D.bx1 + D.vx2*D.bx2)/(D.bx1**2+D.bx2**2)
        D.L = D.Rcyl*(D.vx3 - D.bx3/D.K)
        q   = D.L*D.rho*D.vx1 
            
        for ir,r in enumerate(rad):
            indr=np.abs(r-D.x1).argmin()
            D.jdot_r[ir] = 2.*np.pi*r**2*np.trapz(np.sin(D.x2)*q[indr,:],x=D.x2)
        save_elements([D.rad_j,D.jdot_r],filename)

def compute_mdot_sph(D,rad,recompute=False,suff='',silent=True):
    "Compute the mass loss rate"
    if not os.path.isdir(D.wdir+'/Data'):
        call("mkdir -p "+D.wdir+'/Data',shell=True)

    filename=D.wdir+'/Data/mdot_saved'+suff
    if os.path.exists(filename) and (not recompute):
        if not silent:
            print ('Warning, reading previously computed Jdot')
        [D.rad_m,D.mdot_r] = read_elements(2,filename)
    else:
        D.rad_m = rad
        D.mdot_r = np.zeros_like(rad)

        q = D.rho*D.vx1 

        for ir,r in enumerate(rad):
            indr=np.abs(r-D.x1).argmin()
            D.mdot_r[ir] = 2.*np.pi*(r**2)*np.trapz(np.sin(D.x2)*q[indr,:],x=D.x2)
        save_elements([D.rad_m,D.mdot_r],filename)


def compute_jdot(D,rad,v2=False,int_Sph=False,recompute=False,rotF=False,suff=''):
    "Compute the angular momentum loss rate"

    if (not hasattr(D,'xx')):
        compute_grids(D)
    if not os.path.isdir(D.wdir+'data'):
        call("mkdir -p "+D.wdir+'data',shell=True)

    filename=D.wdir+'data/jdot_saved'+suff
    
    if os.path.exists(filename) and (not recompute):
        print ('Warning, reading previously computed Jdot')
        [D.rad_j,D.jdot_r] = read_elements(2,filename)
    else:
        D.rad_j = rad
        D.jdot_r = np.zeros(len(rad))

        if (v2):
            D.OmS = 0. 
            if (rotF):
                if (D.R_ORBIT_PLANET != 0.):
                    D.OmS = 1./(D.R_ORBIT_PLANET**1.5)
                else:
                    D.OmS = D.VphiStar_VESC*sqrt(2.)
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

        for ir,r in enumerate(rad):        
            D.jdot_r[ir] = compute_flux(qx,qy,qz,D,r,int_Sph=int_Sph)
        save_elements([D.rad_j,D.jdot_r],filename)

def compute_mdot(D,rad,int_Sph=False,recompute=False,suff=''):
    "Compute the mass loss rate"
    
    if not os.path.isdir(D.wdir+'data'):
        call("mkdir -p "+D.wdir+'data',shell=True)

    filename=D.wdir+'data/mdot_saved'+suff
    if os.path.exists(filename) and (not recompute):
        print ('Warning, reading previously computed Mdot')
        [D.rad_m,D.mdot_r] = read_elements(2,filename)
    else:
        qx = D.rho*D.vx1
        qy = D.rho*D.vx2
        qz = D.rho*D.vx3
        
        D.rad_m = rad
        D.mdot_r = np.zeros(len(rad))
        for ir,r in enumerate(rad):        
            D.mdot_r[ir] = compute_flux(qx,qy,qz,D,r,int_Sph=int_Sph)
        save_elements([D.rad_m,D.mdot_r],filename)

def compute_amom_contrib_p(D):
    if (not hasattr(D,'R_PLANET')):
        get_caseparams(D)
    if (not hasattr(D,'xx')):
        compute_grids(D)
    rem_rot = False
    if hasattr(D,'rot_rem'):
        if D.rot_rem:
            add_rotframe(D)
            rem_rot = True

    # Constuct the various terms 
    D.xxo = D.xx-D.R_ORBIT_PLANET
    D.Rsph_p = np.sqrt(D.xxo**2+D.yy**2+D.zz**2)
    
    D.T_ram = -(D.yy*D.vx1-D.xx*D.vx2)*D.rho*(D.vx1*D.xxo+D.vx2*D.yy+D.vx3*D.zz)/D.Rsph_p
    D.T_cor = (1./(D.R_ORBIT_PLANET**1.5)*(D.Rcyl**2))*D.rho*(D.vx1*D.xxo+D.vx2*D.yy+D.vx3*D.zz)/D.Rsph_p
    #D.T_ram = -(D.yy*D.vx1-D.xx*D.vx2-D.VphiStar_VESC*sqrt(2.)*(D.Rcyl**2))*D.rho*(D.vx1*D.xxo+D.vx2*D.yy+D.vx3*D.zz)/D.Rsph_p
    D.T_mt  =  (D.yy*D.bx1-D.xx*D.bx2)*(D.bx1*D.xxo+D.bx2*D.yy+D.bx3*D.zz)/D.Rsph_p
    D.T_p   = D.yy*D.R_ORBIT_PLANET*D.prs/D.Rsph_p
    D.T_mp  = D.yy*D.R_ORBIT_PLANET*(D.bx1**2+D.bx2**2+D.bx3**2)/(2.*D.Rsph_p)

    if rem_rot:
        remove_rotframe(D)

def plot_amom_contrib_p_rad(D,rad,recompute=False,norm=1.,laby='',leg=True,rnorm=1.,\
                                xb=0.15,xe=0.35,doplot=True,ff=None):
    if (not hasattr(D,'R_PLANET')):
        get_caseparams(D)
    if (not hasattr(D,'xx')):
        compute_grids(D)
    if (not hasattr(D,'T_ram')):
        compute_amom_contrib_p(D)

    #D.phi_g = -1./D.Rsph - D.MASS_RATIO/D.Rsph_p
    if (ff == None):
        ff = D.wdir+'/save_torqs'

    list_torques = ['T_ram','T_cor','T_mt','T_p','T_mp']
    
    list_contribs=[] 
    if (not os.path.exists(ff)) or recompute:
        for iT,TT in enumerate(list_torques):
            myT=getattr(D,TT)
            #exec('myT = D.%s' % (TT))
            list_contribs.append(0.*rad)
            for ir,rt in enumerate(rad):
                [tts,pps,q,dint]=interpolate_on_sphere(myT,D.x1,D.x2,D.x3,rt,dr=D.R_ORBIT_PLANET)
                list_contribs[iT][ir] = np.trapz(np.trapz(q*dint,x=tts,axis=0),x=pps)
        save_elements([rad]+list_contribs,ff)
    else:
        print('Reading saved torques')
        list_contribs = read_elements(len(list_torques)+1,ff)
        list_contribs=list_contribs[1:]

    tot=0.*rad
    for iT,TT in enumerate(list_torques):
        tot = tot+list_contribs[iT]

    list_torq_plot = ['0+1','2','3','4']
    list_labels  = ['Ram + Coriolis','Magnetic tension','Pressure','Magnetic pressure']
    
    #figure()
    if (doplot):
    	toplot = np.zeros(len(list_contribs[0]))
    	for iT,labT in enumerate(list_labels):
    	    toplot[:] = 0.
    	    for ttp in list_torq_plot[iT].split('+'):
    	        toplot += list_contribs[int(ttp)]
    	    plot(rad/rnorm,toplot/norm,label=labT,lw=2)
    	plot(rad/rnorm,tot/norm,color='k',label='Total',lw=3,alpha=0.7)
    	xlabel(r'Integration radius [R'+r'$_P$'+']')
    	ylabel(laby)
    	#ylabel(r'$T/\tau_w$')
    	grid(True)
    	if (leg):
    	    legend(bbox_to_anchor=(1.5,0.5),loc='center')
    #plot(rad,tot-2*list_contribs[0]-2*list_contribs[2],color='k',alpha=0.4,lw=2)

    # Save torq
    ibb = np.abs(rad-xb).argmin() ; iee = np.abs(rad-xe).argmin()
    D.saved_torq_p = np.mean(tot[ibb:iee])

def plot_amom_contrib_p(D,rt=None,twosigma=None,suff_title='',verbose=False,mycmap='RdYlBu_r',norm=1.,suptit=None,annot=True):
    if (not hasattr(D,'R_PLANET')):
        get_caseparams(D)
    if (not hasattr(D,'xx')):
        compute_grids(D)
    if (not hasattr(D,'T_ram')):
        compute_amom_contrib_p(D)
    if (rt == None):
        rt = 1.5*D.R_PLANET
    
    list_torques = ['T_ram','T_cor','T_mt','T_p','T_mp']
    list_labels  = ['Ram','Coriolis','Magnetic tension','Pressure','Magnetic pressure']
    list_pos     = [1,2,3,4,5]
    list_tot = []

    tot = 0.*D.T_ram ; tot_int = 0.
    str_summary = ''
    I=pp.Image()
    for iT in range(5):
        subplot(2,3,list_pos[iT])
        myT=getattr(D,list_torques[iT])
        #exec ("myT = D.%s" % (list_torques[iT]))
        tot = tot+myT
        [tts,pps,q,dint]=interpolate_on_sphere(myT,D.x1,D.x2,D.x3,rt,dr=D.R_ORBIT_PLANET)
        plot_mollweide(q/norm,pps,tts,tit=list_labels[iT],mycmap=mycmap,twosigma=twosigma,annot=annot)
        if (annot):
            annotate(r'$\oplus$ Star',xy=(1.,1.),xycoords='axes fraction',ha='right',va='top')
            annotate(r'$\rightarrow V_{\rm orb}$',xy=(0.,1.),xycoords='axes fraction',ha='left',va='top')
        list_tot.append(np.trapz(np.trapz(q*dint/norm,x=tts,axis=0),x=pps))
        if (annot):
            annotate(r'$%.3e$' % (list_tot[iT]),xy=(0,0),xycoords='axes fraction',ha='left')
        tot_int = tot_int+list_tot[iT]
        str_summary = str_summary+list_labels[iT]+': '+str(list_tot[iT])+'\n'
    str_summary = str_summary+'Tot: '+str(tot_int)+'\n'
    subplot(2,3,6)
    [tts,pps,q,dint]=interpolate_on_sphere(tot,D.x1,D.x2,D.x3,rt,dr=D.R_ORBIT_PLANET)
    plot_mollweide(q/norm,pps,tts,tit='Total',mycmap=mycmap,twosigma=twosigma,annot=annot)
    #subplots_adjust(left=0.05,right=0.95,wspace=0.1,hspace=0.1)
    cax=gcf().add_axes([0.86,0.2,0.01,0.55])
    norm = mpl.colors.Normalize(vmin=-twosigma,vmax=twosigma)
    cb1=mpl.colorbar.ColorbarBase(cax,cmap=mycmap,norm=norm,orientation='vertical',ticks=[-twosigma,0,twosigma])
    cb1.set_label(r'$\mathcal{T}/\mathcal{T}_w$')
    subplots_adjust(left=0.05,right=0.85,top=0.95,bottom=0.,wspace=0.1,hspace=-0.5)
    # Add Colortable on the right
    if (verbose):
        print(str_summary)
    if suptit != None:
        suptitle(suptit)
    else:
        suptitle('Summary done on a sphere for r='+str(rt)+suff_title)

def plot_amom_contrib_rs(D,rt=None,twosigma=None,suff_title='',verbose=False,mycmap='RdYlBu_r'):
    if (not hasattr(D,'R_PLANET')):
        get_caseparams(D)
    if (not hasattr(D,'xx')):
        compute_grids(D)
    if (not hasattr(D,'T_ram')):
        compute_amom_contrib_p(D)
    if (rt == None):
        rt = [1.5*D.R_PLANET]
    
    r_ref = rt[-1]

    list_torques = ['T_ram+T_cor','T_mt','T_p','T_mp']
    list_labels  = ['Ram','Magnetic tension','Pressure','Magnetic pressure']
    list_tot = []

    tot = 0.*D.T_ram ; tot_int = 0.
    str_summary = ''
    I=pp.Image()
    #fig=figure(figsize=(16,6))
    for ir,rr in enumerate(rt):
        for iT,tt in enumerate(list_torques):
            subplot(len(rt),5,ir*(len(list_torques)+1)+iT+1)
            strexec = 'myT = '
            myT=0.
            for tmpstr in tt.split('+'):
                myT += getattr(D,tmpstr)
                #strexec = strexec + ' + D.'+tmpstr
            #exec(strexec)
            tot = tot+myT
            if (ir == 0):
                mytit=list_labels[iT]
            else:
                mytit=''
            [tts,pps,q,dint]=interpolate_on_sphere(myT,D.x1,D.x2,D.x3,rr,dr=D.R_ORBIT_PLANET,ntt=10,npp=20)
            plot_mollweide(q,pps,tts,tit=mytit,mycmap=mycmap,twosigma=twosigma,rr=rr/r_ref,draw_par=False)
            annotate(r'$\oplus$ Star',xy=(1.,1.),xycoords='axes fraction',ha='right',va='top')
            annotate(r'$\rightarrow V_{\rm orb}$',xy=(0.,1.),xycoords='axes fraction',ha='left',va='top')
            list_tot.append(np.trapz(np.trapz(q*dint,x=tts,axis=0),x=pps))
            annotate(r'$%.3e$' % (list_tot[iT]),xy=(0,0),xycoords='axes fraction',ha='left')
            tot_int = tot_int+list_tot[iT]
            str_summary = str_summary+list_labels[iT]+': '+str(list_tot[iT])+'\n'
            str_summary = str_summary+'Tot: '+str(tot_int)+'\n'
        subplot(len(rt),5,ir*(len(list_torques)+1)+5)
        [tts,pps,q,dint]=interpolate_on_sphere(tot,D.x1,D.x2,D.x3,rr,dr=D.R_ORBIT_PLANET,ntt=10,npp=20)
        if (ir == 0):
            mytit='Total'
        else:
            mytit=''
        plot_mollweide(q,pps,tts,tit=mytit,mycmap=mycmap,twosigma=twosigma,rr=rr/r_ref,draw_par=False)
        if (verbose):
            print(str_summary)
    #suptitle('Summary done on a sphere for r='+str(rt)+suff_title)
    subplots_adjust(left=0.05,right=0.95,wspace=0.1,hspace=-0.1)

def plot_mollweide(qty,pps,tts,tit='',lon_0=0,lat_0=0,mycmap='RdYlBu_r',twosigma=None,rr=1.,draw_par=True,draw_daynight=True,annot=True):
    lat = -(tts*180./np.pi-90.) ; lon = pps*180./np.pi
    q = qty[:,:]
    x,y = np.meshgrid(lon,lat)
    if (twosigma == None):
        twosigma = 2*np.std(q)
    # Draw boundary
    ax = gca()
    m1 = Basemap(projection="moll",lon_0=lon_0,lat_0=lat_0,ax=ax)
    m1.drawmapboundary(color='w')
    dd = str(rr*100)[0:3]
    #axz = inset_axes(ax,width=dd+'%',height=dd+'%',loc=10)
    axz = inset_axes(ax,width=dd+'%',height=dd+'70%',loc=10)
    m = Basemap(projection="moll",lon_0=lon_0,lat_0=lat_0,ax=axz)
    #image1=m.pcolormesh(x,y,q,latlon=True,\
    #                        vmin=-twosigma,vmax=twosigma)
    image1=m.contourf(x,y,q,latlon=True,levels=np.linspace(-twosigma,twosigma,num=10),extend='both')
    image1.set_cmap(mycmap)
    insert(image1)
    m.drawmapboundary()
    if draw_par:
        parallels = np.arange(-90.,90.,30.) ; m.drawparallels(parallels)
    if draw_daynight:
        daynight = [-90.,90.]; m.drawmeridians(daynight,dashes=[1,0])
    #ax.annotate(u'\u00B1'+('%.2e' % twosigma),xy=(1.,0.),xycoords='axes fraction',ha='right')
    if (annot):
        axz.annotate(r'$\pm$ %.2e' % (twosigma),xy=(1.,0.),xycoords='axes fraction',ha='right')
    axz.set_title(tit,fontsize=18)
    
def interpolate_on_sphere(q,x1,x2,x3,rt,dr=0.,ntt=30,npp=60):
    tts = np.linspace(0.,np.pi,num=ntt)
    pps = np.linspace(0.,2.*np.pi,num=npp)
    if (hasattr (scipy.interpolate, 'RegularGridInterpolator')):
        q_i = RegularGridInterpolator((x1,x2,x3),q)
    else:
        print ('I need a more recent version of scipy')
        return 0
        
    tmp_q = np.zeros((ntt,npp))
    dint = np.zeros((ntt,npp))
    for it,tt in enumerate(tts):
        for ip,pp in enumerate(pps):
            x = dr+rt*sin(tt)*cos(pp) ; y = rt*sin(tt)*sin(pp) ; z =rt*cos(tt)
            tmp_q[it,ip] = q_i((x,y,z)) 
            dint[it,ip] = (rt**2)*sin(tt)

    return [tts,pps,tmp_q,dint]

def find_close_loops(D):
    D.idLoops = 0.*D.vx1 -1
    D.idLoops[np.where(D.Rsph < 1.)] = 0.
    D.idLoops[np.where(D.Rsph > 15.)] = 0.
    I=pp.Image()
    for ix in range(D.n1):
        for iy in range(D.n2):
            for iz in range(D.n3):
                R = np.sqrt(D.x1[ix]**2+D.x2[iy]**2+D.x3[iz]**2)
                if (R > 1.0) and (R < 15.):
                    [qx,qy,qz]=I.ASfl3D(D.bx1,D.bx2,D.bx3,D.x1,D.x2,D.x3,D.dx1,D.dx2,D.dx3,\
                                        D.x1[ix],D.x2[iy],D.x3[iz])
                    R0=np.sqrt(qx[0]**2+qy[0]**2+qz[0]**2)
                    R1=np.sqrt(qx[-1]**2+qy[-1]**2+qz[-1]**2)
                    if ((R0 < 1.5) and (R1 < 1.5)):
                        D.idLoops[ix,iy,iz] = 1.
                    else:
                        D.idLoops[ix,iy,iz] = 0.
def get_lims(x,dx):
    xx = np.zeros(len(x)+1)
    xx[1:] = x + dx/2.0
    xx[0]  = xx[1] - dx[0]
    return xx

def my_movie(D,q,shift=0,vmin=0,vmax=0,addS=True,addP=True,xl=[0,0],yl=[0,0],addAlf=False,BS=False,BP=False,logq=False,opb=False):
    "My wrapper to plot a lot of different stuff"

    if (not hasattr(D,'x1r')):
        print ('Recomputing cylindrical vectors...')
        compute_grids(D)
    if (addAlf and (not hasattr(D,'Alf'))):
        comp_va_cs(D)
        print ('Recomputing characteristic velocities...')

    i0 = np.abs(D.x1).argmin()
    j0 = np.abs(D.x2).argmin()
    k0 = np.abs(D.x3).argmin()

    # Various defs
    if hasattr(D,q):
        toplot=getattr(D,q)
        #exec ("toplot = D.%s" % q)
    else:
        print ('Err: Quantity not coded yet')
        return

    if (logq):
        toplot=np.log10(toplot)
        if (vmin == vmax):
            vmin = np.amin(toplot)
            vmax = np.amax(toplot)
    else:
        if (vmin == vmax):
            twosigma = 2*np.std(toplot)
            vmin = -twosigma ; vmax = twosigma

    # Plot the three cuts
    I = pp.Image()
    clf()
    if (shift != 0):
        print('I shifted the plots from:'     )
        print('-------> in x1:',D.x1[i0+shift])
        print('-------> in x2:',D.x2[j0+shift])
        print('-------> in x3:',D.x3[k0+shift])

    subplot(1,3,1)
    I.pldisplay(toplot[:,j0+shift,:],x1=D.x1r,x2=D.x3r,vmin=vmin,vmax=vmax,\
                label1 = 'X',label2='Z',cbar='horizontal',loc_cbar='top')
    if (addS):
        addstar(rs=sqrt(1-D.x2[j0+shift]**2))
    if (addP):
        if not hasattr(D,'R_ORBIT_PLANET'):
            get_caseparams(D)
        addplanet(rorb=D.R_ORBIT_PLANET,rp=sqrt(D.R_PLANET**2-D.x2[j0+shift]**2))
    if (BS):
        angles=linspace(-0.9*pi/2.0,0.9*pi/2.0,20)
        [flx1,fly1]=I.ASfieldlines(D.bx1[:,j0+shift,:],\
                                   D.bx3[:,j0+shift,:],\
                                   D.x1,D.x3,D.dx1,D.dx3,\
                                   1.2*cos(angles),1.2*sin(angles),step_min=1.e-2)
        for ilx,lx in enumerate(flx1):
            plot(lx,fly1[ilx],color='k',ls='-')
        angles=linspace(-0.9*pi/2.0+pi,0.9*pi/2.0+pi,20)
        [flx1,fly1]=I.ASfieldlines(D.bx1[:,j0+shift,:],\
                                   D.bx3[:,j0+shift,:],\
                                   D.x1,D.x3,D.dx1,D.dx3,\
                                   1.2*cos(angles),1.2*sin(angles),step_min=1.e-2)
        for ilx,lx in enumerate(flx1):
            plot(lx,fly1[ilx],color='k',ls='-')
    if (BP):
        angles=linspace(-0.9*pi,0.9*pi,20)
        [flxP,flyP]=I.ASfieldlines(D.bx1[:,j0+shift,:],\
                                   D.bx3[:,j0+shift,:],\
                                   D.x1,D.x3,D.dx1,D.dx3,\
                                   D.R_ORBIT_PLANET+1.2*D.R_PLANET*cos(angles),\
                                   1.2*D.R_PLANET*sin(angles),step_min=1.e-2)
        for ilx,lx in enumerate(flxP):
            plot(lx,flyP[ilx],color='k',ls='-')
    if (addAlf):
        contour(D.x1,D.x3,D.Alf[:,j0+shift,:].T,levels=[1.0],color='b')
    if (opb):
        I.oplotbox(D.AMRLevel,lrange=[0,len(D.AMRLevel)-1],jslice=j0+shift)

    if (xl[0]!=xl[1]):
        xlim(xl)
    if (yl[0]!=yl[1]):
        ylim(yl)
    
    subplot(1,3,2)
    I.pldisplay(toplot[i0+shift,:,:],x1=D.x2r,x2=D.x3r,vmin=vmin,vmax=vmax,\
                label1 = 'Y',label2='Z',cbar='horizontal',loc_cbar='top')
    if (addS):
        addstar(rs=sqrt(1-D.x1[i0+shift]**2))
    if (addAlf):
        contour(D.x2,D.x3,D.Alf[i0+shift,:,:].T,levels=[1.0],color='b')
    if (opb):
        I.oplotbox(D.AMRLevel,lrange=[0,len(D.AMRLevel)-1],islice=i0+shift)
    if (xl[0]!=xl[1]):
        xlim(xl)
    if (yl[0]!=yl[1]):
        ylim(yl)

    subplot(1,3,3)
    I.pldisplay(toplot[:,:,k0+shift],x1=D.x1r,x2=D.x2r,vmin=vmin,vmax=vmax,\
                label1 = 'X',label2='Y',cbar='horizontal',loc_cbar='top')
    if (addS):
        addstar(rs=sqrt(1-D.x3[k0+shift]**2))
    if (addP):
        if (not hasattr(D,'R_ORBIT_PLANET')):
            get_caseparams(D)
        addplanet(rorb=D.R_ORBIT_PLANET,rp=sqrt(D.R_PLANET**2-D.x3[k0+shift]**2))
    if (addAlf):
        contour(D.x1,D.x2,D.Alf[:,:,k0+shift].T,levels=[1.0],color='b')
    if (opb):
        I.oplotbox(D.AMRLevel,lrange=[0,len(D.AMRLevel)-1],kslice=k0+shift)
    if (xl[0]!=xl[1]):
        xlim(xl)
    if (yl[0]!=yl[1]):
        ylim(yl)
        
    suptitle(q)
    
def load_j(journal='',verbose=False):

    # First, use latex because we want it to be pretty
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text',usetex=True)

    ## Depending on the journal to publish in
    ## add the correct preamble here
    if (journal == 'solar physics'):
        rcParams['text.latex.preamble']=\
            [r"\DeclareMathVersion{bold}",\
                 r"\usepackage[psamsfonts]{amsfonts}",\
                 r"\usepackage{amssymb}",\
                 r"\usepackage{mathptmx}"]
    else:
        rcParams['text.latex.preamble']=\
            [r"\usepackage{wasysym}"]
        if verbose:
            print("Using defaults for LaTeX in figures (generic journal)")

def unload_j():
    rcdefaults()

def running_mean(x,N):
    return smooth(x,N,do_fit_edges=True,window='hanning')

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2-1):-(window_len/2)-1]

def load_mycmaps(cmap='bwr'):
    
    cdictAS1 = {'red':  ((   0., 1.0, 1.0),
                         (1./6., 0.0, 0.0),
                         (1./2., 0.0, 0.0),
                         (4./6., 1.0, 1.0),
                         (  1.0, 1.0, 1.0)),
                'green': ((  0.0,1.0, 1.0),
                          (1./6.,1.0, 1.0),
                          (2./6.,0.0, 0.0),
                          (4./6.,0.0, 0.0),
                          (5./6.,1.0, 1.0),
                          (  1.0,1.0, 1.0)),
                'blue':  ((  0.0,1.0, 1.0),
                          (2./6.,1.0, 1.0),
                          (1./2.,0.0, 0.0),
                          (5./6.,0.0, 0.0),
                          (  1.0,1.0, 1.0)),
                }
    AS1 = LinearSegmentedColormap('AS1', cdictAS1)
    plt.register_cmap(cmap=AS1)

def nonuniform_imshow(x, y, C, **kwargs):
    """Plot image with nonuniform pixel spacing.
    This function is a convenience method for calling image.NonUniformImage.
    """
    ax = plt.gca()
    im = NonUniformImage(ax,extent=(x[0],x[-1],y[0],y[-1]), **kwargs)
    im.set_data(x, y, C)
    ax.images.append(im)
    ax.set_xlim([x[0],x[-1]])
    ax.set_ylim([y[0],y[-1]])
    return im

class ListCollection(Collection): 
     def __init__(self, collections, **kwargs): 
         Collection.__init__(self, **kwargs) 
         self.set_collections(collections) 
     def set_collections(self, collections): 
         self._collections = collections 
     def get_collections(self): 
         return self._collections 
     @allow_rasterization 
     def draw(self, renderer): 
         for _c in self._collections: 
             _c.draw(renderer) 

def insert(c,zorder=0): 
     collections = c.collections 
     for _c in collections: 
         _c.remove() 
     cc = ListCollection(collections, rasterized=True,zorder=zorder) 
     ax = plt.gca() 
     ax.add_artist(cc) 
     return cc 

class Params():
    def __init__(self):
        self.Ms = 1.98855e33
        self.Rs = 6.9599e10
        self.Os = 2.6e-6
        self.GG = 6.6728e-8
        self.kb = 1.38065e-16
        self.mh = 1.6605e-24
        self.Mj = 1.89813e30
        self.Me = 5.9721e27
        self.Rj = 6.9911e9
        self.Re = 6.371e8
        self.AU = 1.495979e13

def calc_star_params(M=1.,R=1.,P=1.,ns=1.e8,Ts=0.,Fx=0.,Btypical=0.,\
                         RP=1.,MP=1.,PO=1.,mu=0.5,gamma=1.05,rho_ref=1.e-16,p_unit='J'):
    PP = Params()
    ## few defs
    #vescsun = (2.*PP.GG*Msun/Rsun)**(0.5)
    #fsun = Osun * Rsun**1.5 / (GG*Msun)**0.5
    #Mdotsun = 2e-14 * Msun / (3600.*24.*365)
    ##
    # Calibrated values for gamma=1.05
    if (gamma == 1.05):
        Tsun     = 1.5e6                         # [K]
        rhosun   = 1.e8                          # [cm^{-3}]
        mdot_sun = 2.e-14*PP.Ms/(365.*24.*3600.) # [g s^{-1}]
        Fx_sun   = 3.e4                          # [erg cm^{-2} s{-1} ]
        A1 = 0.28
        A2 = 0.07
        p1 = 0.19
        p2 = 1.6
        vesc_sun = (2.*PP.GG*PP.Ms/PP.Rs)**0.5
        mdot_sun_vr = 1.e-3 # A1 * (2./(np.sqrt(4.*np.pi)*vesc_sun))**(-p1) * (1.+(0.004/A2)**2)**(p2)
        #mdot_sun_vr = A1 * (2./(np.sqrt(4.*np.pi*2.9e-15)*vesc_sun))**(-p1) * (1.+(0.004/A2)**2)**(p2)
        print (mdot_sun_vr)
    ##
    Ms = M*PP.Ms
    Rs = R*PP.Rs
    Os = 2.*np.pi/(P*24.*3600.)
    if (p_unit == 'J'):
        Rp = RP*PP.Rj
        Mp = MP*PP.Mj
    elif (p_unit == 'E'):
        Rp = RP*PP.Re
        Mp = MP*PP.Me
    else:
        print("Unknown units for planet")
        return
    Rorb = (PP.GG*Ms)**(1./3.)*(PO*24.*3600./(2.*np.pi))**(2./3.)
    ## 
    vrot = Rs * Os
    vesc = (2.*PP.GG*Ms/Rs)**0.5
    vrot_vesc = vrot/vesc
    fs = vrot_vesc*np.sqrt(2.)
    cs_vesc = (gamma*PP.kb*Ts/(mu*PP.mh*(vesc**2)))**(0.5)
    ## units
    rho_ref = 1.e-16
    rho_star = ns * mu * PP.mh/rho_ref
    unit_l = Rs
    unit_v = sqrt(PP.GG*Ms/unit_l)
    unit_B = unit_v * sqrt(4.*np.pi*rho_ref)
    #print(unit_v)
    ##
    rhos_phys_hj = rhosun * (Os/PP.Os)**(0.6)
    Ts_hj = Tsun * (Os/PP.Os)**(0.1)
    cs_vesc_hj = (gamma*PP.kb*Ts_hj/(mu*PP.mh*(vesc**2)))**(0.5)
    rhos_hj = (rhos_phys_hj * mu * PP.mh)/rho_ref
    ## 
    if (Fx != 0):
        #rhos_Fx = ((mdot_sun)*(Fx/Fx_sun)**1.34/(0.28*(1.+(fs/0.087)**2)**(1./0.19))) * Btypical / (4.*np.pi*vesc**2)
        #rhos_Fx = (A1 * (Btypical/(np.sqrt(4.*np.pi)*vesc))**(-p1) * (1.+(fs/A2)**2)**(p2) / (mdot_sun * (Fx/Fx_sun)**1.34 ))**(-2./p1)
        rhos_Fx = (A1 * (Btypical/(np.sqrt(4.*np.pi)*vesc))**(-p1) * (1.+(fs/A2)**2)**(p2) / (mdot_sun_vr * (Fx/Fx_sun)**1.34 ))**(-2./p1)
        #print((Btypical/(np.sqrt(4.*np.pi)*vesc))**(-p1),(1.+(fs/A2)**2)**(p2),mdot_sun * (Fx/Fx_sun)**1.34)
        rhos_phys_Fx = rhos_Fx*rho_ref/(mu*PP.mh)
    #print (1.e9*mu*PP.mh/rho_ref)
    ##
    print ('>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print ('> Values for PLUTO setup <')
    print ('>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print ('> Star <')
    print (' gamma     = %.2f' % (gamma))
    print (' cs/vesc   = %.2e' % (cs_vesc))
    print (' vrot/vesc = %.2e' % (vrot_vesc))
    #print (' Rho star  = %.2e' % (rho_star))
    #print ('rho star = ',rhos)
    if (Ts != 0):
        print ('.. Chosen Temperature ..')
        print (' Ts        = %.2e [K]' % (Ts))
        print (' cs/vesc   = %.2e' % (cs_vesc))
    print ('.. HJ Parametrization ..')
    print (' Ts        = %.2e [K]' % (Ts_hj))
    print (' cs/vesc   = %.2e' % (cs_vesc_hj))
    print (' rhos      = %.2e [%.2e g/cm^3, %.2e cm^{-3}]' % (rhos_hj,rhos_hj*rho_ref,rhos_phys_hj))
    #print (' B norm    = %.2e [G]' % (unit_B))
    if (Fx != 0):
        print ('.. Fx Parametrization ..')
        print (' Ts        = %.2e [K]' % (Ts_hj))
        print (' cs/vesc   = %.2e' % (cs_vesc_hj))
        print (' rhos      = %.2e [%.2e g/cm^3, %.2e cm^{-3}]' % (rhos_Fx,rhos_Fx*rho_ref,rhos_phys_Fx))
    print ('> Planet <')
    print (' Rp/Rs     = ',Rp/Rs)
    print (' Mp/Ms     = ',Mp/Ms)
    print (' Rorb/Rs   = ',Rorb/Rs)
    print ('>>>>>>>>>>>>>>>>>>>>>')

def plot_p_balance(D):
    """ Pressure contributions to detail magnetopshere """
    ### Plot it on the ecliptic only for the time being

    iz = np.abs(D.x3).argmin()
    iy = np.abs(D.x2).argmin()
    ix0 = np.abs(D.x1-(D.R_ORBIT_PLANET-10.*D.R_PLANET)).argmin()
    ix1 = np.abs(D.x1-(D.R_ORBIT_PLANET-D.R_PLANET)).argmin()

    xx = D.x1[ix0:ix1]
    vv = np.sqrt( D.vx1[ix0:ix1,iy,iz]**2 + D.vx2[ix0:ix1,iy,iz]**2 + D.vx3[ix0:ix1,iy,iz]**2)
    bb = np.sqrt( D.bx1[ix0:ix1,iy,iz]**2 + D.bx2[ix0:ix1,iy,iz]**2 + D.bx3[ix0:ix1,iy,iz]**2)

    pram = 0.5*D.rho[ix0:ix1,iy,iz] * vv**2
    pres = D.prs[ix0:ix1,iy,iz]
    pmag = 0.5*bb**2
 
    plot(xx,pres,color='g',label=r'$P$')
    plot(xx,pram,color='b',label=r'$P_{\rm ram}$')
    plot(xx,pmag,color='r',label=r'$P_B$')

    plot(xx,pres+pram+pmag,color='k',label='Total')
    yscale('log')

    legend(loc='upper left').draw_frame(0)

    plot(xx,0.5*D.R_BPEQBS**2/(xx-D.R_ORBIT_PLANET)**6,color='r',ls='--')

def plot_ecliptic_ref(D,mycmap='OrRd',toplot=None,lab='',cclim=[0.,1.],density_stream=3.,lw=0,col=(0,0,0,0.7)):
    iz=np.argmin(np.abs(D.x3)) ; col_annot = 'm'
    # background color
    if toplot != None:
        im=nonuniform_imshow(D.x1,D.x2,toplot[:,:,iz].T,cmap=mycmap,interpolation='bilinear')
        im.set_clim(cclim)
    # streamplot
    y_interp,x_interp = np.mgrid[D.x2[0]:D.x2[-1]:64j,D.x1[0]:D.x1[-1]:64j]
    n_v1 = RectBivariateSpline(D.x1,D.x2,D.vx1[:,:,iz].squeeze())
    n_v2 = RectBivariateSpline(D.x1,D.x2,D.vx2[:,:,iz].squeeze())
    v1_interp = n_v1(x_interp,y_interp,grid=False)
    v2_interp = n_v2(x_interp,y_interp,grid=False)
    if lw != 0:
        speed = np.sqrt(v1_interp**2+v2_interp**2) ; lw = 2.*speed/speed.max()
    else:
        lw = 1
    cc=streamplot(x_interp,y_interp,v1_interp,v2_interp,density=density_stream,color=col,linewidth=lw)
    
def plot_ecliptic(D,mycmap='OrRd',cclim=[0.,1.],save_f='test.png'):
    iz=np.argmin(np.abs(D.x3)) ; col_annot = 'm'
    fig=figure(figsize=(10,5))
    subplot(121)
    #toplot=np.sqrt(D.bx1**2+D.bx2**2+D.bx3**2)/np.sqrt(D.J1**2+D.J2**2+D.J3**2) ; lab=r'$L_J/R_\star$'
    dxx=np.minimum(np.minimum(np.tile( np.tile(D.dx1,(D.n2,1))  , (D.n3,1,1)).T,np.tile( np.tile(D.dx2,(D.n1,1)).T, (D.n3,1,1)).T),np.tile( np.tile(D.dx3,(D.n2,1))  , (D.n1,1,1)))
    toplot=np.sqrt(D.J1**2+D.J2**2+D.J3**2)*dxx/np.sqrt(D.bx1**2+D.bx2**2+D.bx3**2) ; lab=r'$J\, {\rm d}x/R_\star$'
    # Colormap
    im=nonuniform_imshow(D.x1,D.x2,toplot[:,:,iz].T,cmap=mycmap,interpolation='bilinear')
    im.set_clim(cclim)
    # Streamplot
    ## Interpolations
    y_interp,x_interp = np.mgrid[D.x2[0]:D.x2[-1]:64j,D.x1[0]:D.x1[-1]:64j]
    n_v1 = RectBivariateSpline(D.x1,D.x2,D.vx1[:,:,iz].squeeze())
    n_v2 = RectBivariateSpline(D.x1,D.x2,D.vx2[:,:,iz].squeeze())
    v1_interp = n_v1(x_interp,y_interp,grid=False)
    v2_interp = n_v2(x_interp,y_interp,grid=False)
    speed = np.sqrt(v1_interp**2+v2_interp**2) ; lw = 2.*speed/speed.max()
    cc=streamplot(x_interp,y_interp,v1_interp,v2_interp,density=3.,color=(0,0,0,0.3),linewidth=speed)
    # Addictional Contour 
    cs=contour(D.x1,D.x2,toplot[:,:,iz].T,[0.1],colors=['w'],linewidths=2)
    verts=cs.collections[0].get_paths()[0]
    vs=verts.vertices
    x_cs = np.zeros(len(vs)) ; y_cs = np.zeros(len(vs))
    for ii,vvs in enumerate(vs):
        x_cs[ii] = vvs[0] ; y_cs[ii] = vvs[1]
    #i1 = np.argmin(y_cs)
    x_tmp = np.zeros(y_cs.shape)
    x_tmp[np.where(y_cs > 0.)] = 1.e10
    x_tmp[np.where(y_cs < 0.)] = x_cs[np.where(y_cs < 0.)]
    i1 = np.argmin(np.abs(np.sqrt(y_cs**2+x_tmp**2)-D.R_ORBIT_PLANET))
    plot([x_cs[i1]],[y_cs[i1]],marker='d',color='m')
    i2 = np.argmax(y_cs)
    plot([x_cs[i2]],[y_cs[i2]],marker='d',color='m')
    nfx=np.zeros(len(x_cs)-np.abs(i2-i1)+1) ; nfy = np.zeros(len(x_cs)-np.abs(i2-i1)+1)
    for ii in range(len(x_cs)-np.abs(i2-i1)+1):
        nfx[ii] = x_cs[(min(i2,i1)+ii) % len(x_cs)]
        nfy[ii] = y_cs[(min(i2,i1)+ii) % len(x_cs)]
    nfx2=np.zeros(np.abs(i2-i1)) ; nfy2 = np.zeros(np.abs(i2-i1))
    for ii in range(np.abs(i2-i1)):
        nfx2[ii] = x_cs[(max(i1,i2)+ii) % len(x_cs)]
        nfy2[ii] = y_cs[(max(i1,i2)+ii) % len(x_cs)]
    plot(nfx,nfy,color='m',ls='--',lw=2)
    plot(nfx2,nfy2,color='r',ls='--',lw=2)
    # Test streamline of the flow...
    I=pp.Image()
    y0=D.magn_fl['cfl_ymin2']
    x0=np.sqrt(D.R_ORBIT_PLANET**2-y0**2)
    fxv=[x0] ; fyv=[y0] ; dy=D.dx2[np.argmin(np.abs(D.x2))]
    while (max(fyv) < D.magn_fl['cfl_ymax2']) or (fyv[-1] != min(fyv)):
        fls=I.ASfield_line(D.vx1[:,:,iz],D.vx2[:,:,iz],D.x1,D.x2,D.dx1,D.dx2,x0,y0,only_bck=True,rorb=D.R_ORBIT_PLANET,rp=D.R_PLANET,step_max=0.005)
        fxv=fls['qx'];fyv=fls['qy']
        y0=y0-dy
        x0=np.sqrt(D.R_ORBIT_PLANET**2-y0**2)
        if len(fyv) < 1:
            fyv=[y0]
    plot(fxv,fyv,color='r',lw=2)
    iymax_fy = np.argmin(np.abs(np.array(fyv)-D.magn_fl['cfl_ymax2']))
    dpar_new = 0. ; t_advec = 0. 
    vo_loc = RegularGridInterpolator((D.x1,D.x2),np.sqrt(D.vx1[:,:,iz]**2+D.vx2[:,:,iz]**2).squeeze())
    for ix,xx in enumerate(fxv[iymax_fy+1:]):
        dl = np.sqrt( (fxv[iymax_fy+ix]-xx)**2 + (fyv[iymax_fy+ix]-fyv[iymax_fy+ix+1])**2)
        t_advec += dl/vo_loc((xx,fyv[iymax_fy+ix+1]))
        dpar_new += dl
    t_advec2 = 0.
    for ix,xx in enumerate(nfx[:-1]):
        dl = np.sqrt( (nfx[ix+1]-xx)**2 + (nfy[ix+1]-nfy[ix])**2)
        t_advec2 += dl/vo_loc((xx,nfy[ix]))
    t_advec3 = 0.
    for ix,xx in enumerate(nfx2[:-1]):
        dl = np.sqrt( (nfx2[ix+1]-xx)**2 + (nfy2[ix+1]-nfy2[ix])**2)
        t_advec3 += dl/vo_loc((xx,nfy2[ix]))
    # Markers
    ymin = D.magn_fl['cfl_ymin'] ; ymax=D.magn_fl['cfl_ymax'] ; xmin = D.magn_fl['cfl_xmin'] ; xmax = D.magn_fl['cfl_xmax']
    print("Adv times : %.2f,%.2f,%.2f,%.2f,%.2f,%.2f" % (t_advec,t_advec2,t_advec3,D.dpar_th_1*(D.R_ORBIT_PLANET**0.5),D.dpar_th_2*(D.R_ORBIT_PLANET**0.5),(ymax-ymin)*(D.R_ORBIT_PLANET**0.5)))
    #plot([D.R_ORBIT_PLANET],[ymin],marker='x',color=col_annot,mew=2)
    #plot([D.R_ORBIT_PLANET],[ymax],marker='x',color=col_annot,mew=2)
    for  ifx,fx in enumerate(D.magn_fl['flxms']):
        plot([fx[0]],[D.magn_fl['flyms'][ifx][0]],color='b',marker='o')
    for  ifx,fx in enumerate(D.magn_fl['flxms2']):
        plot([fx[0]],[D.magn_fl['flyms2'][ifx][0]],color='g',marker='x')
    # Annotation
    ymin = D.magn_fl['cfl_ymin2'] ; ymax=D.magn_fl['cfl_ymax2']
    plot([D.R_ORBIT_PLANET],[ymin],marker='o',color='c',mew=2)
    plot([D.R_ORBIT_PLANET],[ymax],marker='o',color='c',mew=2)
    annotate(r'',xycoords='data',xy=(xmin-0.1,ymin),xytext=(xmin-0.1,ymax),color=col_annot,arrowprops=dict(arrowstyle='<->',color=col_annot,shrinkA=0.0,shrinkB=0.0),zorder=3)
    annotate(r'$d_{||}$',xycoords='data',xy=(xmin-0.2,(ymin+ymax)*0.5),color=col_annot,ha='right',va='center',zorder=3)
    annotate(r'',xycoords='data',xy=(xmax+0.1,ymax-D.dpar_th_1),xytext=(xmax+0.1,ymax),color='b',arrowprops=dict(arrowstyle='<->',color='b',shrinkA=0.0,shrinkB=0.0),zorder=3)
    annotate(r'',xycoords='data',xy=(xmax+0.15,ymax-D.dpar_th_2),xytext=(xmax+0.15,ymax),color='g',arrowprops=dict(arrowstyle='<->',color='g',shrinkA=0.0,shrinkB=0.0),zorder=3)
    annotate(r'$d_{||}$',xycoords='data',xy=(xmax+0.2,(ymin+ymax)*0.5),color='b',ha='left',va='center',zorder=3)
    annotate(r'',xycoords='data',xy=(xmax+0.3,ymax-dpar_new),xytext=(xmax+0.3,ymax),color='g',arrowprops=dict(arrowstyle='<->',color='r',shrinkA=0.0,shrinkB=0.0),zorder=3)
    plot([fxv[iymax_fy]],[fyv[iymax_fy]],color='r',markersize=10,marker='d')
    rad_c1=D.dpar_th_1/(np.pi) 
    rad_c2=D.dpar_th_2/(np.pi) 
    addcircle(center=(D.R_ORBIT_PLANET,ymax-rad_c1),rp=rad_c1,color='b',ls='dashed')
    addcircle(center=(D.R_ORBIT_PLANET,ymax-rad_c2),rp=rad_c2,color='g',ls='dashed')
    #plot([D.R_ORBIT_PLANET],[D.magn_fl['cfl_ymax']*1.15],marker='x',color='m')
    #iy = np.argmin(np.abs(D.x2)) ; ix = np.argmin(np.abs(D.x1-D.R_ORBIT_PLANET))
    #ymax = D.x2[iy+np.argmax(toplot[ix,iy:,iz])]
    #ymin = D.x2[np.argmax(toplot[ix,:iy,iz])]
    #plot([D.R_ORBIT_PLANET],[ymin],marker='o',color='c')
    #plot([D.R_ORBIT_PLANET],[ymax],marker='o',color='c')    
    # Misc
    addplanet(rorb=D.R_ORBIT_PLANET,fc='w')
    addcircle(center=(0.,0.),rp=D.R_ORBIT_PLANET,color='k',ls='dashed')
    # Labels
    xlim([D.x1[0],D.x1[-1]])
    ylim([-1.,D.x2[-1]])
    xlabel(r'$x/R_\star$')
    ylabel(r'$y/R_\star$')
    fig.gca().set_aspect('equal',adjustable='box')
    # Colorbar
    cax=fig.add_axes([0.5,0.125,0.01,0.775])
    norm=mpl.colors.Normalize(vmin=cclim[0],vmax=cclim[-1])
    cb1=mpl.colorbar.ColorbarBase(cax,cmap=mycmap,norm=norm,orientation='vertical',ticks=[cclim[0],cclim[1]])
    cb1.set_label(lab)

    subplot(122)
    cc=streamplot(x_interp,y_interp,v1_interp,v2_interp,density=3.,color=(0,0,0,0.3),linewidth=speed)
    i1 = np.argmin(np.abs(np.sqrt(y_cs**2+x_cs**2)-D.R_ORBIT_PLANET))
    #i1 = np.argmin(y_cs)
    plot([x_cs[i1]],[y_cs[i1]],marker='d',color='m')
    i2 = np.argmax(y_cs)
    plot([x_cs[i2]],[y_cs[i2]],marker='d',color='m')
    plot(nfx,nfy,color='m',ls='--',lw=2)
    plot(nfx2,nfy2,color='r',ls='--',lw=2)
    # Colormap
    cclim2=[0.,1.]
    toplot=np.sqrt(D.vx1**2+D.vx2**2)
    im=nonuniform_imshow(D.x1,D.x2,toplot[:,:,iz].T,cmap=cmaps.viridis,interpolation='bilinear')
    im.set_clim(cclim2)
    # Misc
    addplanet(rorb=D.R_ORBIT_PLANET,fc='w')
    addcircle(center=(0.,0.),rp=D.R_ORBIT_PLANET,color='k',ls='dashed')
    # Labels
    xlim([D.x1[0],D.x1[-1]])
    ylim([-1.,D.x2[-1]])
    xlabel(r'$x/R_\star$')
    fig.gca().set_aspect('equal',adjustable='box')
    # Colorbar
    cax=fig.add_axes([0.9,0.125,0.01,0.775])
    norm=mpl.colors.Normalize(vmin=cclim[0],vmax=cclim[-1])
    cb1=mpl.colorbar.ColorbarBase(cax,cmap=cmaps.viridis,norm=norm,orientation='vertical',ticks=[cclim2])
    cb1.set_label('|V|')


    print(save_f)
    fig.savefig(save_f)#,bbox_inches='tight')
