import sys,os
import SynthAIAUtils_202004 as aia
import matplotlib.pyplot as plt
import numpy as np
import time

#filters=['94', '131', '171', '193', '211', '304', '335']
#vmin=[0,0,-7,0,0,0,0]
#vmax=[369,241,5735,4861,2701,613,718]
filters=['193']
vmin=[0]
#vmax=[7000] # 171
vmax=[2701]

tt = time.time()

#os.getcwd()
#os.chdir('/mnt/c/Users/Susanna Parenti/Documents/SOFTS/PYTHON_soft/PLUTO_soft/')
#dir=os.getenv("DATA_PATH")+"WaveHeat3D/WaHe3D_TwoTurbMap_NoRot/"
dir='/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/VICTOR_EX/WET_3D_HLL_PSP_E1/'
#sun=aia.create_synth_AIA(dir,filter="193",filename="data.0025.dbl")
sun=aia.synth_AIA(dir,filter="193",filename="data.0156.vtk")
#sun.load_pluto_data(GEOM="Cartesian")
sun.load_pluto_data(GEOM="Spherical")
print('Start interpolate AIA response', time.time() - tt)
sun.interp_AIA_response()
print('Start integrate LOS', time.time() - tt)
sun.integrate_los()
print('End integrate LOS', time.time() - tt)
sun.plot_AIA_synthetic_image(save=True,vmin=vmin[0],vmax=vmax[0],im_dir="/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/VICTOR_EX/")

for ii,ff in enumerate(filters[1:]):
    #sun.view_phi=pp
    sun.filter=ff 
    sun.interp_AIA_response()
    sun.integrate_los()
 
    sun.plot_AIA_synthetic_image(save=True,vmin=vmin[ii+1],vmax=vmax[ii+1],im_dir="/mnt/c/Users/Susanna Parenti/Documents/LAVORO/PROJECTS/CEA/VSWMC/VSWMC_DATA/VICTOR_EX/")

print('Total time', time.time() - tt)

 
"""
for ii,pp in enumerate(rotatingPhi):
    sun96.view_phi=pp
    sun96.integrate_los()
    sun96.plotAIASyntheticImage(save=True,im_dir="/Users/vreville/Movie96/",cbar=True)
    print("Image {0} written out of {1}".format(ii,len(rotatingPhi)))
    plt.close()
"""
