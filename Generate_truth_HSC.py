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

print("reading HSC cosmos deep data...")
filename="maps-data/229328.csv"
t=Table.read(filename)
print("read",filename)
print(t.dtype.names)
print(len(t))

nside = 256
nested= True
pix_area = hp.pixelfunc.nside2pixarea(nside, degrees=True)
t["hpxpixel"] = hp.ang2pix(nside, np.radians(90 - t["dec"]), np.radians(t["ra"]), nested)

# -------------------------------- CONVERT MAGNITUDES WITH PHOTOMETRIC TRANSFORMS --------------------------------------------

def convert_mags_hsc2ls(t, region="N"):
    g = t["g"]
    r = t["r"]
    z = t["z"]
    
    if region == "N":
        newg = g - 0.003 + 0.029*(g-r)
        newr = r + 0.003 - 0.130*(r-z) + 0.053*(r-z)**2 - 0.013*(r-z)**3
        newz = z - 0.011 - 0.076*(r-z) + 0.003*(r-z)**2
    else:
        newg = g + 0.003 - 0.014*(g-r)
        newr = r - 0.011 - 0.154*(r-z) + 0.055*(r-z)**2 - 0.013*(r-z)**3
        newz = z - 0.024 - 0.098*(r-z) + 0.004*(r-z)**2
    
    return newg, newr, newz

def convert_fluxes_hsc2ls(t, region="N"):
    g, r, z = convert_mags_hsc2ls(t, region)
    return flux_from_mag(g), flux_from_mag(r), flux_from_mag(z)

def flux_from_mag(mag):
    return 3630.78*10**(0.4*(22.5-mag))

def mag_from_flux(flux):
    return 22.5 - 2.5*np.log10(flux)

# -------------------------------- CONVERT G-BAND MAGNITUDE INTO G-BAND FIBER MAGNITUDE -----------------------------------------

#The transformation has been fitted with a polynome, see . Warning ! This transformation is not precise and introduces a huge error.
def convert_gmag2fibmag(g):
    polyg_fit = [-4.65011229e-05,  5.36886061e-03, -2.45838755e-01,  5.59914789e+00,
 -6.38127754e+01,  2.94538817e+02]
    return np.array(np.polyval(polyg_fit, t["g"]) + t["g"])

# -------------------------------- CLEAN THE CATALOG AND EXTRACT THE TRUTH SAMPLE --------------------------------------------

def clean_sample(t):
    """ Cleans the initial catalog, according to the criteria described in "The DESI Emission Line Galaxy sample: target selection and validation", A. Raichoor & al., 2022: at least one observation in each band, veto masks. A second selection is done to keep only the objects with magnitudes between 10 and 30, to avoid np.nan or incoherent values.
    """
    
    ii = np.where(((t["gN"]>10) == True) & ((t["gN"]<30) == True) & ((t["rN"]>10) == True) & ((t["rN"]<30) == True) & ((t["zN"]>10) == True) & ((t["zN"]<30) == True) & ((t["gS"]>10) == True) & ((t["gS"]<30) == True) & ((t["rS"]>10) == True) & ((t["rS"]<30) == True) & ((t["zS"]>10) == True) & ((t["zS"]<30) == True) & (t["g_mask_pdr2_bright_objectcenter"]=='False') & (t["r_mask_pdr2_bright_objectcenter"]=='False') & (t["z_mask_pdr2_bright_objectcenter"]=='False'))
    return t[ii]


def extract_truth_sample(t, ratio_density):
    """ Extracts a random truth sample in the truth catalog. The number of objects in the sample is the number of object per healpixel in the truth catalog (Nobj) * a ratio used to increase the statistics in the simulation (the density is divided by ratio_density at the end of the simulation).
    """
    hsc_hpx = np.unique(t["hpxpixel"])
    tab_nb = []
    for i in range(len(hsc_hpx)):
        tab_nb.append(len(t["hpxpixel"][np.where(t["hpxpixel"]==hsc_hpx[i])]))
    tab_nb = np.array(tab_nb)
    Nobj = int(np.mean(tab_nb))
    print("Average number of objects per healpix: " + str(Nobj))
    N = Nobj*ratio_density
    print("Ratio between sample density and real density: " + str(ratio_density))
    index = np.random.choice(len(t), N, replace=False)
    tpix = t[index]
    return tpix

# -------------------------------- MAIN --------------------------------------------

ratio_density = 4
t["ratio_density"] = ratio_density * np.ones(len(t))



t["g"] = mag_from_flux(t["g_cmodel_flux"]/3630.78)
t["r"] = mag_from_flux(t["r_cmodel_flux"]/3630.78)
t["z"] = mag_from_flux(t["z_cmodel_flux"]/3630.78)
t["gfib"] = convert_gmag2fibmag(t["g"])

t["g_cmodel_flux_N"], t["r_cmodel_flux_N"], t["z_cmodel_flux_N"] = convert_fluxes_hsc2ls(t, "N")
t["g_cmodel_flux_S"], t["r_cmodel_flux_S"], t["z_cmodel_flux_S"] = convert_fluxes_hsc2ls(t, "S")
t["gN"] = mag_from_flux(t["g_cmodel_flux_N"]/3630.78)
t["rN"] = mag_from_flux(t["r_cmodel_flux_N"]/3630.78)
t["zN"] = mag_from_flux(t["z_cmodel_flux_N"]/3630.78)
t["gS"] = mag_from_flux(t["g_cmodel_flux_S"]/3630.78)
t["rS"] = mag_from_flux(t["r_cmodel_flux_S"]/3630.78)
t["zS"] = mag_from_flux(t["z_cmodel_flux_S"]/3630.78)
t["gfibN"] = convert_gmag2fibmag(t["gN"])
t["gfibS"] = convert_gmag2fibmag(t["gS"])

t = clean_sample(t)
t.write("maps-data/HSC_cleaned_truth_catalog.csv", overwrite=True)

t = extract_truth_sample(t, ratio_density)
print(str(len(t)) + " objects in the truth sample")

t.write("maps-data/HSC_truth_sample.csv", overwrite=True)
