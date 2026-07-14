import pyPLUTO as pp
import numpy as np
import scipy.integrate as scint
import matplotlib.pyplot as plt
from AStools import get_caseparams

# User editable
num_beg = 0
num_end = 100
step = 10
#wdir = '/drf/projets/stars2/bperri/PLUTO/quasistat/newBC/newBL_CT_outflow_complete/178/'
#wdir = '/drf/projets/stars2/bperri/PLUTO/quasistat/newBC/test_divc_clean_min/'
#wdir = '/drf/projets/stars2/bperri/PLUTO/coupling/coupling_opt/opt_occ_extra_b2/'
#wdir = '/dsm/stars2/f/bperri/PLUTO/3d/test1au/'
#wdir = '/drf/projets/stars2/bperri/PLUTO/3D_test/Sph3D_map/test2/'
#wdir = '/drf/projets/stars2/bperri/PLUTO/3D_test/1AU/sph2/'
#wdir = '/drf/projets/stars2/bperri/PLUTO/3D_test/22r/min_nopoles_vvlr/'
#wdir = "C:\\Users\\Susanna Parenti\\Documents\\LAVORO\\PROJECTS\\CEA\\VSWMC\\VSWMC_PYTHON\\"
wdir = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/'

# Declarations
d = pp.pload(num_beg, w_dir=wdir)
get_caseparams(d)
nb_time = int((num_end - num_beg)/step)
ra_emp = np.zeros((d.n2, d.n3))
ra_emp_time = np.zeros(nb_time)
ra_eff_time = np.zeros(nb_time)
rad = np.linspace(2.0,19.0,num = 50)
nb_rad = len(rad)
idx_rad = np.zeros(len(rad),dtype=int)
ML = np.zeros(nb_rad)
ML_time = np.zeros((nb_time, nb_rad))
JL = np.zeros(nb_rad)
JL_time = np.zeros((nb_time, nb_rad))
time_vec = np.zeros(nb_time)

# Physical units
Rs = 6.96e10
Ms = 1.99e33
GG = 6.67e-8
unit_length = Rs
unit_density = d.UNIT_DENSITY #1.67e-16 # careful, hardcoded !!!
unit_velocity = np.sqrt(GG*Ms/Rs)
unit_mass = unit_density*unit_length**3
unit_time = unit_length/unit_velocity
mdot_norm = unit_mass/unit_time
jdot_norm = unit_mass*unit_length**2/unit_time**2
VROT_VESC = d.VROT_VESC #2.93e-3 # careful, hard-coded
omega_cgs = VROT_VESC*unit_velocity/Rs

# Loop on time
time = 0

for num in range(num_beg, num_end, step):

  #print('{}'.format(num))
  d = pp.pload(num, w_dir=wdir)

  #############################################################
  # Empirical Alfven Radius
  #############################################################

  MA = (d.vx1**2+d.vx2**2)/(d.bx1**2+d.bx2**2)*d.rho
  for k in range(0, d.n3):
    for j in range(0, d.n2):
      i = 0
      while((MA[i,j,k] < 1.0) & (i < d.n1-2)):
        i = i+1
      ra_emp[j,k] = d.x1[i+1]
  ra_emp_time[time] = np.mean(ra_emp)

  #############################################################
  # Theoritical Alfven Radius
  #############################################################

  for r in range(0,nb_rad):
    step = 0
    while(d.x1[step] < rad[r]):
      step = step + 1
    idx_rad[r] = step
  vpol = np.sqrt(d.vx1**2 + d.vx2**2)
  bpol = np.sqrt(d.bx1**2 + d.bx2**2)

  # Mass Loss
  for r in range(0,nb_rad):
    rho = d.rho[idx_rad[r],:,:]
    v_r = d.vx1[idx_rad[r],:,:]
    ML[r] = rad[r]**2*scint.simps(scint.simps(rho*v_r, x = d.x3)*np.sin(d.x2), x = d.x2)
  mdot_cgs = np.mean(ML[9:])*mdot_norm
  ML_time[time,:] = ML*mdot_norm*(3600.*24.*365./Ms)

  # Angular Momentum Loss
  for r in range(0,nb_rad):
    rho = d.rho[idx_rad[r],:,:]
    v_r = d.vx1[idx_rad[r],:,:]
    sinth = np.tile(d.x2, (d.n3,1)).T
    Lambda = rad[r]*sinth*(d.vx3[idx_rad[r],:,:] - d.bx3[idx_rad[r],:,:]*(d.vx1[idx_rad[r],:,:]*d.bx1[idx_rad[r],:,:] + d.vx2[idx_rad[r],:,:]*d.bx2[idx_rad[r],:,:])/(d.rho[idx_rad[r],:,:]*vpol[idx_rad[r],:,:]**2))
    JL[r] = rad[r]**2*scint.simps(scint.simps(rho*Lambda*v_r, x = d.x3)*np.sin(d.x2), x = d.x2)
  jdot_cgs = np.mean(JL)*jdot_norm
  JL_time[time,:] = JL*jdot_norm

  # Effective Alfven radius
  Ra_cgs = np.sqrt(jdot_cgs/(mdot_cgs*omega_cgs))
  ra_eff_time[time] = Ra_cgs/Rs

  # Time update
  time_vec[time] = num
  time = time+1

# Plot time dependency
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12,8))
ax1.plot(time_vec, ra_emp_time)
ax1.plot(time_vec, ra_eff_time)
ax1.set_ylabel(r'Alfven radii [$R_\odot$]', fontsize=10)
ax2.plot(time_vec, np.mean(ML_time, axis=1))
ax2.set_ylabel(r'Mass loss [$M_\odot$/yr]', fontsize=10)
ax3.plot(time_vec, np.mean(JL_time, axis=1))
ax3.set_ylabel(r'Ang. mom. loss [cgs]', fontsize=10)
ax3.set_xlabel(r'Time [PLUTO unit]')
fig.show()
