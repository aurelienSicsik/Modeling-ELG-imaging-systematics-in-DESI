#!/usr/bin/env python

import healpy as hp
import numpy as np
import fitsio
import glob
import statistics as st
import os

from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib

from functions import  match_hpx_index


""" This script generates the truth sample from the HSC COSMOS Deep/UltraDeep catalog. The truth sample is then to be used as an input in the Monte-Carlo simulation.
"""


# -------------------------------- LOAD THE CATALOG --------------------------------------------

print("reading LS catalog...")
filename="maps-data/pixweigths-dark-dr9.csv"
t_ls=Table.read(filename)
print("read",filename)
print(t_ls.dtype.names)
print(len(t_ls))

nside = 256
nested= True
pix_area = hp.pixelfunc.nside2pixarea(nside, degrees=True)

t_ls["RA"] = np.zeros(len(t_ls))
t_ls["DEC"] = np.zeros(len(t_ls))
for i in range(len(t_ls)):
    theta, phi = hp.pix2ang(nside, t_ls["HPXPIXEL"][i], nested)
    t_ls["RA"][i], t_ls["DEC"][i] = np.degrees(phi), 90 - np.degrees(theta)


# -------------------------------- CLEAN THE CATALOG --------------------------------------------

def clean_catalog(t):
    ii = np.where((t["DEC"]>-30) & (t['FRACAREA']>0.90))[0]
    return t[ii]

# -------------------------------- MAIN --------------------------------------------

t_ls = clean_catalog(t_ls)
t_ls.write("maps-data/LS_healpix_map.csv", overwrite=True)
print(str(len(t_ls)) + " pixels in the footprint")