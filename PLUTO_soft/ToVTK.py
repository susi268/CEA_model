#!/usr/bin/env python
import pyPLUTO as pp
import sys
try:
    from evtk.hl import gridToVTK # This can be downloaded there: https://bitbucket.org/pauloh/pyevtk
except ImportError:
    print("You should install VTK python module, see https://bitbucket.org/pauloh/pyevtk")
    exit()

narg = len(sys.argv)
if (narg == 1):
    it0 = 0 ; it1= 0
elif (narg == 2 ):
    it0 = int(sys.argv[1]) ; it1=it0
elif (narg >= 3):
    it0 = int(sys.argv[1]) 
    it1 = int(sys.argv[2]) 

nit=it1+1-it0

bdir='./'
for it in range(nit):
    D = pp.pload(it0+it,w_dir=bdir)
    D.ToVTK()
