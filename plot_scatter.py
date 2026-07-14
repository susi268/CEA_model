#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:29:23 2021

@author: sparenti
"""

import numpy as np
import matplotlib.pyplot as plt
import SPUtils as spu
import SPUtils_AIA as spuaia
import SPUtils_gen as spugen
import scipy.interpolate as interp
import matplotlib.cm as mplcm
import SPUtils_WL as spuwl
import SPUtils_mod as spumod
import os 



def plot_scatter(file1, file2):
    
    dataf1, h1, dataf2_bin, h2 = spugen.open2imgs(dirf1, file1, dirf2, file2):
        
    return