#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:56:47 2020

@author: sparenti
"""

import SPUtils_gen as spugen

file ='23731980pB.fts'
dirf1 = '/mnt/c/Users/Susanna Parenti/Documents/LAVORO/FITS/LASCO/'

image, h = spugen.open1img(dirf1, file)
img_lag = spugen.get_img_Lapgrad(image, sv=True, out='LASCO')
img_wvl = spugen.get_img_sharp(image, weight=[1,1,0], sv=True, out='LASCO')
