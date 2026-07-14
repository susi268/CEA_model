import pyPLUTO as pp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

#dir = '/drf/projets/stars2/bperri/PLUTO/coupling/coupling_opt/test_occ4/'
#dir = '/dsm/stars2/b/bperri/coupling/coupling_test/test_new/'
#dir = '/drf/projets/stars2/bperri/PLUTO/coupling/coupling_opt/wind/wind_small_dens_bphi/'
#dir = '/drf/projets/stars2/bperri/PLUTO/coupling/coupling_fast/dyn_1month/'
#dir = '/drf/projets/stars2/bperri/PLUTO/coupling/coupling_b/dyn_fast_va_weak/'
#dir = '/drf/projets/stars2/bperri/PLUTO/dynamo/dynamo_CLpot_pluto/dyn_pluto_unit/'
#dir = '/drf/projets/stars2/bperri/PLUTO/dynamo/dynamo_CLpotb/dyn14_pluto_unit/'
#dir = '/drf/projets/alfven-data/bperri/PLUTO/coupling_b3/'
#dir = '/drf/projets/alfven-data/bperri/PLUTO/coupling_opt_new2/'
#dir = '/drf/projets/alfven-data/bperri/PLUTO/dynamo_solar3/'
#dir = '/drf/projets/stars2/bperri/PLUTO/coupling/coupling_opt/wind/wind_small_dens/'
#dir = '/drf/projets/stars2/bperri/PLUTO/coupling/coupling_b_par/dyn12/coupling_sol/'
#dir = '/drf/projets/stars2/bperri/PLUTO/3D_test/AMR/2d/'
#dir = '/drf/projets/stars2/bperri/PLUTO/wind/SphWind2D_ZF/'
#dir = '/drf/projets/alfven-data/bperri/PLUTO/dynamo_CLpotb_pluto/'
#dir = '/drf/projets/alfven-data/bperri/PLUTO/dynamo_CLpotb2/'
#dir = '/drf/projets/stars2/bperri/PLUTO/dynamo/dynamo_sbp/'
#dir = '/drf/projets/alfven-data/bperri/PLUTO/coupling_b_par2/'
dir = '/drf/projets/stars2/bperri/PLUTO/coupling/coupling_b_par/dyn14/dyn14_sym/'
#dir = '/drf/projets/stars2/bperri/PLUTO/coupling/coupling_b_par/wind/wind_14_denser/'
#dir = '/drf/projets/alfven-data/bperri/PLUTO/dynamo_bushby/'
#dir = '/drf/projets/alfven-data/bperri/PLUTO/coupling_b_par_bushby/'
#dir = '/drf/projets/alfven-data/bperri/PLUTO/coupling_dyn_inter/'
#dir = '/drf/projets/alfven-data/bperri/PLUTO/coupling_rot2/'
#dir = '/dsm/stars2/b/bperri/coupling_sol/'

num = 710
qty_name = 'bpol'
zoom = 'no'
Rmax = 1.2
AMR = 'no'
level = 4
if (AMR == 'yes'):
  d = pp.pload(num, w_dir=dir, level=level)
else:
  d = pp.pload(num, w_dir=dir)

if (zoom == 'yes'):
  idx = 0
  while ((d.x1[idx] < Rmax) & (idx < d.n1-1)):
    idx = idx+1
else:
  idx = d.n1-1

Rr = np.tile(d.x1r[0:idx],(d.n2+1,1)).T
Thetar = np.tile(d.x2r,(idx,1))
R = np.tile(d.x1[0:idx],(d.n2,1)).T
Theta = np.tile(d.x2,(idx,1))
rcosr = Rr*np.cos(Thetar)
rsinr = Rr*np.sin(Thetar)
rcos = R*np.cos(Theta)
rsin = R*np.sin(Theta)

#Omega = d.vx3/rsin
vpol = np.sqrt(d.vx1**2+d.vx2**2)
bpol = np.sqrt(d.bx1**2+d.bx2**2)

#VAVESC = np.sqrt(d.bx1**2+d.bx2**2+d.bx3**2)/np.sqrt(2.0*d.rho)
VAVESC = np.sqrt(d.bx1**2+d.bx2**2)/np.sqrt(2.0*d.rho)
#CSVESC = np.sqrt(d.prs/d.rho)/np.sqrt(2.0)
#ava = d.alpha/np.sqrt(d.bx1**2+d.bx2**2+d.bx3**2)*np.sqrt(d.rho)
#avphi = d.alpha/d.vx3
#aeta = d.alpha/d.eta
#br_bis = d.bx1*(0.7-R)**2/R**2

if (qty_name == 'none'):
  qty = VAVESC[0:idx,:] #d.Ax3 #CSVESC #np.abs(d.bx2/d.bx1)
if (qty_name == 'vpol'):
  qty = vpol[0:idx,:] #d.Ax3 #CSVESC #np.abs(d.bx2/d.bx1)
if (qty_name == 'bpol'):
  qty = bpol[0:idx,:] #d.Ax3 #CSVESC #np.abs(d.bx2/d.bx1)
else:
  qty = getattr(d, qty_name) #d.Ax3 #CSVESC #np.abs(d.bx2/d.bx1)
  qty = qty[0:idx,:]

dev = np.std(qty)

fig = plt.figure()
ax = fig.add_subplot(111)
plot = ax.pcolormesh(rsinr,rcosr,qty,cmap='plasma')
#plot = ax.pcolormesh(rsin,rcos,qty,vmin=-0.001,vmax=0.001,cmap='seismic')
#plt.title('Alpha')
plt.axis('equal')
plt.colorbar(plot)
#plot.set_clim([np.mean(qty)-2.0*dev,np.mean(qty)+2.0*dev])
#plt.axis('off')
plt.plot(0.7*np.sin(Theta)[0,:],0.7*np.cos(Theta)[0,:],ls="--",color="b")
plt.plot(1.0*np.sin(Theta)[0,:],1.0*np.cos(Theta)[0,:],ls="--",color="b")
if (np.max(qty) > 0.0):
  plt.contour(rsin,rcos,qty,20,colors='w')

fig.show()
