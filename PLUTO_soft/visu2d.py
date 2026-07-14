# Barbara C code
#Modified by S.Prenti Nov 19

import pyPLUTO as pp
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
import time as time
import pdb

# ASfield_line gets only cartesian coords, so some trasformations are needed.
def plot_field_lines(x1, x2, bx1, bx2, r_seed, th_seed, ax, visu_type, dx, dy):
  # Seeds
  x0 = r_seed*np.sin(th_seed)
  y0 = r_seed*np.cos(th_seed)
  #if (visu_type == 'eq_slice'):
  #  x0 = np.concatenate((x0, -r_seed*np.sin(th_seed)))
  #  y0 = np.concatenate((y0, r_seed*np.cos(th_seed)))
  plt.scatter(x0,y0)
  # Interpolation
  fbr = interp.RectBivariateSpline(x1,x2,bx1)
  fbth = interp.RectBivariateSpline(x1,x2,bx2)
  print('Before interpolation')
  if (visu_type == 'mer_slice'):
    x = np.arange(0.0,x1[-1],dx)
    y = np.arange(-x1[-1],x1[-1],dy)
    ddx = dx*np.ones(len(x))
    ddy = dy*np.ones(len(y))
    rcart = np.zeros((len(x),len(y)))
    thcart = np.zeros((len(x),len(y)))
    brcart = np.zeros((len(x),len(y)))
    bthcart = np.zeros((len(x),len(y)))
    for i in range(0,len(x)):
      for j in range(0,len(y)):
        rcart[i,j] = np.sqrt(x[i]**2 + y[j]**2)
        thcart[i,j] = np.arccos(y[j]/rcart[i,j])
        brcart[i,j] = fbr(rcart[i,j],thcart[i,j])
        bthcart[i,j] = fbth(rcart[i,j],thcart[i,j])
    bxcart = np.sin(thcart)*brcart + np.cos(thcart)*bthcart
    bycart = np.cos(thcart)*brcart - np.sin(thcart)*bthcart
  elif (visu_type == 'eq_slice'):
    x = np.arange(-x1[-1],x1[-1],dx)
    y = np.arange(-x1[-1],x1[-1],dy)
    ddx = dx*np.ones(len(x))
    ddy = dy*np.ones(len(y))
    rcart = np.zeros((len(x),len(y)))
    phicart = np.zeros((len(x),len(y)))
    brcart = np.zeros((len(x),len(y)))
    bphicart = np.zeros((len(x),len(y)))
    for i in range(0,len(x)):
      for j in range(0,len(y)):
        rcart[i,j] = np.sqrt(x[i]**2 + y[j]**2)
        phicart[i,j] = np.arctan2(y[j],x[i])
        brcart[i,j] = fbr(rcart[i,j],phicart[i,j])
        bphicart[i,j] = fbth(rcart[i,j],phicart[i,j])
    bxcart = np.cos(phicart)*brcart - np.sin(phicart)*bphicart
    bycart = np.sin(phicart)*brcart + np.cos(phicart)*bphicart
  print('After interpolation')
  #print("Stop \n")
  #pdb.set_trace()
  # Get field lines
  I = pp.Image()
  qxlist = []
  qylist = []
  for i in range(len(x0)):
    print(i)
    tmp = I.ASfield_line(bxcart,bycart,x,y,ddx,ddy,x0[i],y0[i],maxsteps=5.0e5)
    qx = tmp.get('qx')
    qy = tmp.get('qy')
    qxlist.append(qx)
    qylist.append(qy)
    ax.plot(qx,qy,color='w',linewidth=1.0)

# To plot the field lines. This is more rapid (but not used by now)
def streamfunction(nr, nth, r, theta, bx1, bx2, ax):

  sintheta = np.sin(theta)
  dpsi_dr = np.zeros((nr, nth))
  dpsi_dtheta = np.zeros((nr, nth))
  dtheta = np.zeros(nth)
  dr = np.zeros(nr)
  psi = np.zeros((nr, nth))       #potential field vector
  psi2 = np.zeros((nr, nth))
  
  # Derivatives
  for j in range(0,nth):
    dpsi_dr[:,j] = -r*sintheta[j]*bx2[:,j]
    dpsi_dtheta[:,j] = r*r*sintheta[j]*bx1[:,j]

  # First computation
  # From left to right
  # From top to bottom
  dtheta[1:nth-1] = theta[1:nth-1] - theta[0:nth-2]
  dr[1:nr-1] = r[1:nr-1] - r[0:nr-2]
  dtheta[0] = 0.
  dr[0] = 0.

  for i in range(1,nr):
    psi[i,1:nth-1] = psi[i-1,1:nth-1] + dpsi_dr[i,1:nth-1]*dr[i]
  for j in range(1, nth):
    psi[1:nr-1,j] = psi[1:nr-1,j-1] + dpsi_dtheta[1:nr-1,j]*dtheta[j]

  # Second computation
  # From right to left
  # From bottom to top
  dtheta[0:nth-2] = theta[0:nth-2] - theta[1:nth-1]
  dr[0:nr-2] = r[0:nr-2] - r[1:nr-1]
  dtheta[nth-1] = 0.
  dr[nr-1] = 0.

  for i in range(nr-2,0,-1):
    psi2[i,0:nth-2] = psi2[i+1,0:nth-2] + dpsi_dr[i,0:nth-2]*dr[i]
  for j in range(nth-2,0,-1):
    psi2[0:nr-2,j] = psi2[0:nr-2,j+1] + dpsi_dtheta[0:nr-2,j]*dtheta[j]

  # Plot
  psi_end = 0.5*(psi + psi2)
  R = np.tile(r, (nth, 1)).T
  Theta = np.tile(theta, (nr, 1))
  rsin = R*np.sin(Theta)
  rcos = R*np.cos(Theta)
  ax.contour(rsin, rcos, psi_end, 30, colors='w', linewidth=0.1)

####################################################################################

num = 25
#wdir =  '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/BARBARA_EX/'
wdir =  '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/SOUMITRA_EX/'
visu_type = 'eq_slice'

t0 = time.clock()
d = pp.pload(num, w_dir=wdir)
t1 = time.clock()
print('File read in {0} seconds'.format(t1-t0))

vpol = np.sqrt(d.vx1**2 + d.vx2**2)
bpol = np.sqrt(d.bx1**2 + d.bx2**2)
psi = np.arctan(d.bx3/d.bx1)
#R, Th, Phi = np.meshgrid()
qty = vpol #d.vx2 #np.log(np.abs(d.bx1))
r_seed = 10.0
if visu_type == 'mer_slice':
  th_seed = d.x2[::5] #d.x2[::8]
elif visu_type == 'eq_slice':
  th_seed = d.x3[::10]

# Meridional slice at Phi = 0
# 
if (visu_type == 'mer_slice'):
  meri = 0  
  Rr = np.tile(d.x1r, (d.n2+1,1)).T
  R = np.tile(d.x1, (d.n2,1)).T
  Thetar = np.tile(d.x2r, (d.n1+1,1))
  Theta = np.tile(d.x2, (d.n1,1))
  rcosr = Rr*np.cos(Thetar)
  rcos = R*np.cos(Theta)
  rsinr = Rr*np.sin(Thetar)
  rsin = R*np.sin(Theta)
  fig = plt.figure()
  plot = plt.pcolormesh(rsinr, rcosr, qty[:,:,meri], cmap='plasma')
  plt.axis('equal')
  plt.colorbar(plot)
  MA = d.rho[:,:,meri]*(d.vx1[:,:,meri]**2+d.vx2[:,:,meri]**2+d.vx3[:,:,meri]**2)/(d.bx1[:,:,meri]**2+d.bx2[:,:,meri]**2+d.bx3[:,:,meri]**2)
  plt.contour(rsin, rcos, MA, levels=[1.0], colors='k', linewidths=2.0)
  # Adding the slow and fast magneto-sonic surfaces
  cs2 = 1.05*d.prs[:,:,meri]/d.rho[:,:,meri]
  Apol2 = (d.bx1[:,:,meri]**2+d.bx2[:,:,meri]**2)/d.rho[:,:,meri]
  Aphi2 = d.bx3[:,:,meri]**2/d.rho[:,:,meri]
  MAs = 2.0*(d.vx1[:,:,meri]**2+d.vx2[:,:,meri]**2)/(cs2+Apol2+Aphi2-np.sqrt((cs2+Apol2+Aphi2)**2-4.0*cs2*Apol2))
  MAf = 2.0*(d.vx1[:,:,meri]**2+d.vx2[:,:,meri]**2)/(cs2+Apol2+Aphi2+np.sqrt((cs2+Apol2+Aphi2)**2-4.0*cs2*Apol2))
  plt.contour(rsin, rcos, MAs, levels=[1.0], colors='k', linewidths=2.0, linestyles='--')
  plt.contour(rsin, rcos, MAf, levels=[1.0], colors='k', linewidths=2.0, linestyles=':')
  ax = plt.gca()
  #streamfunction(d.n1, d.n2, d.x1, d.x2, d.bx1[:,:,0], d.bx2[:,:,0], ax)
  plot_field_lines(d.x1, d.x2, d.bx1[:,:,0], d.bx2[:,:,0], r_seed, th_seed, ax, visu_type, 1.0, 1.0)
  fig.show()

# Equatorial slice at theta = pi/2
# dn2 fix theta and has to be set at the beginning of the run
  
if (visu_type == 'eq_slice'):
  dn2=int(d.n2/2)  
  R = np.tile(d.x1, (d.n3,1)).T
  Rr = np.tile(d.x1r, (d.n3+1,1)).T
  Phi = np.tile(d.x3, (d.n1,1))
  Phir = np.tile(d.x3r, (d.n1+1,1))
  rcosr = Rr*np.cos(Phir)
  rsinr = Rr*np.sin(Phir)
  rcos = R*np.cos(Phi)
  rsin = R*np.sin(Phi)
  fig = plt.figure()
  plot = plt.pcolormesh(rsinr, rcosr, qty[:,int(d.n2/2),:], cmap='plasma')
  #plot = plt.pcolormesh(rsinr, rcosr, qty[:,d.n2/2,:], cmap='CMRmap', vmin = 0.0, vmax = 1.0)
  plt.axis('equal')
  plt.colorbar(plot)
  MA = d.rho[:,dn2,:]*(d.vx1[:,dn2,:]**2+d.vx2[:,dn2,:]**2+d.vx3[:,dn2,:]**2)/(d.bx1[:,dn2,:]**2+d.bx2[:,dn2,:]**2+d.bx3[:,dn2,:]**2)
  plt.contour(rsin, rcos, MA, levels=[1.0], colors='k', linewidths=2.0)
  ax = plt.gca()
  print('Before field lines')
  plot_field_lines(d.x1, d.x3, d.bx1[:,dn2,:], d.bx3[:,dn2,:], r_seed, th_seed, ax, visu_type, 1.0, 1.0)
  print('After field lines')
  #streamfunction(d.n1, d.n3, d.x1, d.x3, d.bx1[:,d.n2/2,:], d.bx3[:,d.n2/2,:], ax)
  fig.show()

# Spherical slice at R = 1
if (visu_type == 'sph_slice'):
  spher=0  
  Theta, Phi = np.meshgrid(d.x2, d.x3)
  fig = plt.figure()
  plot = plt.pcolormesh(Phi, Theta, qty[spher,:,:].T, cmap='seismic')
  plt.colorbar(plot)
  fig.show()
