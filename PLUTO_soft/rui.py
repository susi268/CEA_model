#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:23:07 2020

@author: sparenti
"""


plfile = "data.0156.vtk"
d = pp.pload(num, w_dir=wdir, datatype='vtk')

d.logN, d.logT (3D matrix in spherical coords)

d.x1, d.x2, d.x3 : 1d matrix for each dimention (r, theta, phi)
d.n1_tot : tot element in dimention 1 (r)
