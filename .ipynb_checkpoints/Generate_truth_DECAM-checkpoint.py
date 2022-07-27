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


""" This script generates the truth sample from the COSMOS DECAM deep catalog. The truth sample is then to be used as an input in the Monte-Carlo simulation.
"""

data_path = "/global/cfs/cdirs/desi/users/asicsik/"
output_path = "/global/cfs/cdirs/desi/users/asicsik/maps-data/"

# -------------------------------- LOAD THE CATALOG --------------------------------------------

print("reading ...")
filename = "/global/cfs/cdirs/desi/users/rongpu/data/decam_deep_fields/cosmos_incomplete.fits"
t=Table(fitsio.read(filename))
print("read",filename)
print(t.dtype.names)
print(len(t))

nside = 256
nested= True
pix_area = hp.pixelfunc.nside2pixarea(nside, degrees=True)
t["hpxpixel"] = hp.ang2pix(nside, np.radians(90 - t["dec"]), np.radians(t["ra"]), nested)

t["g"] = 22.5 - 2.5*np.log10(t["flux_g"]/t["mw_transmission_g"])
t["r"] = 22.5 - 2.5*np.log10(t["flux_r"]/t["mw_transmission_r"])
t["z"] = 22.5 - 2.5*np.log10(t["flux_z"]/t["mw_transmission_z"])
t["gfib"] = 22.5-2.5*np.log10(t["fiberflux_g"]/t["mw_transmission_g"])

# -------------------------------- CLEAN THE CATALOG AND EXTRACT THE TRUTH SAMPLE --------------------------------------------

def clean_sample(t):
    """ Cleans the initial catalog, according to the criteria described in "The DESI Emission Line Galaxy sample: target selection and validation", A. Raichoor & al., 2022: at least one observation in each band, veto masks. A second selection is done to keep only the objects with magnitudes between 10 and 30, to avoid np.nan or incoherent values.
    """
    
    ii = np.where((t["brick_primary"]==True) & (t["nobs_g"]>0) & (t["nobs_r"]>0) & (t["nobs_z"]>0) & 
                  (t["flux_g"]*np.sqrt(t["flux_ivar_g"])>0) & (t["flux_r"]*np.sqrt(t["flux_ivar_r"])>0) & (t["flux_z"]*np.sqrt(t["flux_ivar_z"])>0) & 
                  ((t["maskbits"]%16384)//8192==0) & ((t["maskbits"]%8192)//4096==0) & ((t["maskbits"]%4)//2==0) & (t["brick_primary"]==True))[0]
    t = t[ii]
    
    ii = np.where(((t["g"]>10) == True) & ((t["g"]<30) == True) & ((t["r"]>10) == True) & ((t["r"]<30) == True) & ((t["z"]>10) == True) & ((t["z"]<30) == True) & ((t["gfib"]>10) == True) & ((t["gfib"]<30) == True))
    
    return t[ii]


def clean_mindepth(t, min_depthg, min_depthr, min_depthz):
    """ Extracts a sample from the truth catalog with minimum depth (to ensure the truth sample is deep enough).
    """
    g_depth = 22.5-2.5*np.log10(5/np.sqrt(t["galdepth_g"]))
    r_depth = 22.5-2.5*np.log10(5/np.sqrt(t["galdepth_r"]))
    z_depth = 22.5-2.5*np.log10(5/np.sqrt(t["galdepth_z"]))
    
    return t[np.where((g_depth>min_depthg) & (g_depth<30) & (r_depth>min_depthr) & (r_depth<30) & (z_depth>min_depthz) & (z_depth<30))]


def extract_truth_sample(t, ratio_density):
    """ Extracts a random truth sample in the truth catalog. The number of objects in the sample is the number of object per healpixel in the truth catalog (Nobj), computed with the assumption that the footprint of DECAM COSMOS is elliptical, * a ratio used to increase the statistics in the simulation (the density is divided by ratio_density at the end of the simulation).
    """
    Nobj = int(len(t)/(np.pi*(np.max(t["ra"])-np.min(t["ra"]))*(np.max(t["dec"])-np.min(t["dec"]))/4)*pix_area)
    print("Average number of objects per healpix: " + str(Nobj))
    N = Nobj*ratio_density
    print("Ratio between sample density and real density: " + str(ratio_density))
    index = np.random.choice(len(t), N, replace=False)
    t = t[index]
    return t

# -------------------------------- MAIN --------------------------------------------

ratio_density = 4
t["ratio_density"] = ratio_density * np.ones(len(t))

min_depthg, min_depthr, min_depthz = 24.2, 23.8, 23.2
t = clean_sample(t)
t = clean_mindepth(t, min_depthg, min_depthr, min_depthz)
t.keep_columns(['release', 'brickname', 'objid', 'brick_primary', 'maskbits', 'type', 'ra', 'dec', 'ebv', 'flux_g', 'flux_r', 'flux_z', 'flux_w1', 'flux_w2', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z', 'flux_ivar_w1', 'flux_ivar_w2', 'fiberflux_g', 'fiberflux_r', 'fiberflux_z', 'mw_transmission_g', 'mw_transmission_r', 'mw_transmission_z', 'mw_transmission_w1', 'mw_transmission_w2', 'nobs_g', 'nobs_r', 'nobs_z', 'nobs_w1', 'nobs_w2', 'galdepth_g', 'galdepth_r', 'galdepth_z', 'psfdepth_g', 'psfdepth_r', 'psfdepth_z', 'sersic', 'shape_r', 'shape_e1', 'shape_e2', 'psfsize_g', 'psfsize_r', 'psfsize_z', 'g', 'r', 'z', 'gfib', 'nea_g', 'nea_r', 'nea_z', 'blob_nea_g', 'blob_nea_r', 'blob_nea_z', 'hpxpixel'])
print(str(len(t)) + " objects in the cleaned truth catalog")

t.write(output_path+"DECAM_cleaned_truth_catalog.csv", overwrite=True)

t = extract_truth_sample(t, ratio_density)
print(str(len(t)) + " objects in the truth sample")

t.write(output_path+"DECAM_truth_sample.csv", overwrite=True)
