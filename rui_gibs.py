#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:36:14 2020

Produces a spherical symmetric model of density from Gibson et al. 1999 

@author: sparenti
"""
import numpy   as np

rmin=1.1
rmax=4 
rdim = 256

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

theta = np.linspace(0, np.pi, nang)
phi = np.linspace(0, 2*np.pi, 2 * nang)
