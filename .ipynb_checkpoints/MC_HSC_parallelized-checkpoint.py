#!/usr/bin/env python

import healpy as hp
import numpy as np
import fitsio
import glob
import os

from astropy.table import Table, hstack, vstack
from functions import match_hpx_index
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

n_processes = cpu_count()  # number of cores on Perlmutter
data_path = "/global/cfs/cdirs/desi/users/asicsik/"
output_path = "/global/cfs/cdirs/desi/users/asicsik/MC_results/"

"""
This script performs the Monte-Carlo simulation. It uses two catalogs, the truth sample t and the Healpix map of the Legacy Survey DR9 t_ls. All steps of the simulation can be found below: random depth/seeing-based noise, detection according to the Tractor pipeline, selection of ELGs according to the LS criteria. This version also compute the distribution of n(z) of the selected ELGs.
The code is parallelized to maximise the efficiency. With Cori or Perlmutter (NERSC), the computation time should be of a few minutes for the whole DR9 footprint and about 50k objects in the truth sample.
This version of the code uses the COSMOS DECAM deep catalog as the truth catalog. Anther version of this script using HSC as the truth catalog is in "MC_parallelized.py" 
"""

# -------------------------------- LOAD THE CATALOGS --------------------------------------------

#The catalogs must have been cleaned already
print("reading truth sample...")
filename=data_path + "maps-data/HSC_truth_sample.csv"
t=Table.read(filename)
print("read",filename)
print(t.dtype.names)
print(str(len(t)) + " objects in the HSC truth sample")

print("\nreading LS catalog...")
filename=data_path+"maps-data/LS_healpix_map.csv"
t_ls=Table.read(filename)
print("read",filename)
print(t_ls.dtype.names)
print(str(len(t_ls)) + " pixels in DR9")

ratio_density = t["ratio_density"][0]

def mag_from_flux(flux):
    return 22.5 - 2.5*np.log10(flux)

# -------------------------------- LOAD THE CATALOGS --------------------------------------------

#Compute the true gal_depth
epsilon = 0.7
t_ls["TRUE_DEPTH_G"] = t_ls["PSFDEPTH_G"] / (1+1.30*epsilon/t_ls["PSFSIZE_G"]**2)
t_ls["TRUE_DEPTH_R"] = t_ls["PSFDEPTH_R"] / (1+1.30*epsilon/t_ls["PSFSIZE_R"]**2)
t_ls["TRUE_DEPTH_Z"] = t_ls["PSFDEPTH_Z"] / (1+1.30*epsilon/t_ls["PSFSIZE_Z"]**2)

#Fiber magnitude, transformation from magnitude obtained with the polynomial fit, see Describe_truth_HSC.ipynb
polyg_fit = [-4.65011229e-05,  5.36886061e-03, -2.45838755e-01,  5.59914789e+00,
 -6.38127754e+01,  2.94538817e+02]

# -------------------------------- TRACTOR DETECTION --------------------------------------------

def sed_filters():
    """SED filters used in the Tractor detection: g-band, r-band, z-band, flat and red"""
    SEDs   = []
    bands  = ['g','r','z']
    for i,band in enumerate(bands):
        sed    = np.zeros(len(bands))
        sed[i] = 1.
        SEDs.append((band, sed))
    # Reverse the order -- run z-band detection filter *first*.
    SEDs = list(reversed(SEDs))
    # flat , red sed
    if len(bands) > 1:
        flat = dict(g=1., r=1., i=1., z=1.)
        SEDs.append(('Flat', [flat[b] for b in bands]))
        red = dict(g=2.5, r=1., i=0.4, z=0.4)
        SEDs.append(('Red', [red[b] for b in bands]))
    return SEDs

def isdetect(SEDs, flux_g,flux_r,flux_z,flux_ivar_g,flux_ivar_r,flux_ivar_z):
    """
    Runs a single SED-matched detection filter and assess whether the object are detected or not.
    The Tractor detection threshold is a SNR criterion (SNR>6) described in the paper "Principled point-source detection in collections of astronomical images", D. Lang & D.W. Hogg, 2020
    This code has been adapted from:
    https://github.com/legacysurvey/legacypipe/blob/fe4fd85c665d808097d8f231269783841ea633bc/py/legacypipe/detection.py#L234-L333
    
    Parameters
    ----------
    SEDs : list of tuple
        The SED -- a list of tuple composed of the name of the SED and its values for bands g,r,z
    flux_{g,r,z} : array
        Flux values of each objects in the band {g,r,z}
    flux_ivar_{g,r,z} : array
        Inverse variance of the flux of each objects in the band {g,r,z}
        
    Returns
    -------
    peaks : array
        Boolean array that state whether the object is detected for at least one SED filter
    """
    
    nobj                 = len(flux_g)
    # putting in a dictionary
    fdict                = {}
    fdict['flux_g']      = flux_g
    fdict['flux_r']      = flux_r
    fdict['flux_z']      = flux_z
    fdict['flux_ivar_g'] = flux_ivar_g
    fdict['flux_ivar_r'] = flux_ivar_r
    fdict['flux_ivar_z'] = flux_ivar_z
    #
    bands  = ['g','r','z']
    nsigma = 6
    
    peaks = np.zeros(nobj,dtype=bool)
    for sedname,sed in SEDs:
        sedmap = np.zeros(nobj)
        sediv  = np.zeros(nobj)
        for iband,band in enumerate(bands):
            if (sed[iband]!=0):
                # We convert the detmap to canonical band via
                #   detmap * w
                # And the corresponding change to sig1 is
                #   sig1 * w
                # So the invvar-weighted sum is
                #    (detmap * w) / (sig1**2 * w**2)
                #  = detmap / (sig1**2 * w)
                sedmap += fdict['flux_'+band] * fdict['flux_ivar_'+band] / sed[iband]
                sediv  += fdict['flux_ivar_'+band] / sed[iband]**2
        sedmap /= np.maximum(1e-16, sediv)
        sedsn   = sedmap * np.sqrt(sediv)
        del sedmap
        peaks[sedsn>nsigma] = True
    return peaks


# -------------------------------- ELG SELECTION --------------------------------------------

""" ELG selection cuts for ELGs, ELGs LOP and ELGs VLO described in "The DESI Emission Line Galaxy sample: target selection and validation", A. Raichoor & al., 2022"""

def is_ELG(g, r, z):
    x = r-z
    y = g-r
    gfib = np.polyval(polyg_fit, g) + g
    return (g>20) & (gfib<24.1) & (x>0.15) & (y < 0.5*x+0.1) & (y < -1.2*x+1.6)

def is_ELG_lop(g, r, z):
    x = r-z
    y = g-r
    gfib = np.polyval(polyg_fit, g) + g
    return (g>20) & (gfib<24.1) & (x>0.15) & (y < 0.5*x+0.1) & (y < -1.2*x+1.3)

def is_ELG_vlo(g, r, z):
    x = r-z
    y = g-r
    gfib = np.polyval(polyg_fit, g) + g
    return (g>20) & (gfib<24.1) & (x>0.15) & (y < 0.5*x+0.1) & (y > -1.2*x+1.3) & (y < -1.2*x+1.6)


# -------------------------------- MONTE-CARLO NOISE --------------------------------------------

def random_MC_noise(t_truth, seds, ihpx):
    
    """ Computes the random noise based on the local depth and seeing of the iphx healpixel in the t_ls map, adds it to the truth flux of the t_truth sample and runs the detection process. The 
    extinction is accounted for along the algorithm. The photometric transforms from HSC to DR9 differ in the North and South region, so the initial truth flux used depends on the region of the healpixel ihpx.
    
    Parameters
    ----------
    t_truth : Table
        Truth catalog
    seds : list of tuples
        The SED -- a list of tuple composed of the name of the SED and its values for bands g,r,z
    ihpx : int
        Number of the local current healpixel in the t_ls map (nest=True)
        
    Returns
    -------
    g, r, z, gfib : array of floats
        New simulated magnitudes in g, r, z bands and g-band fiber magnitudes
    detect : array of booleans
        States whether the object is detected or not
    """
    
    region = t_ls["REGION"][ihpx]
    
    #True flux, corrected from the truth catalog extinction
    flux_g0 = t_truth['g_cmodel_flux_' + region]/3630.78 * (10**(0.4*t_truth['a_g']))
    flux_r0 = t_truth['r_cmodel_flux_' + region]/3630.78 * (10**(0.4*t_truth['a_r']))
    flux_z0 = t_truth['z_cmodel_flux_' + region]/3630.78 * (10**(0.4*t_truth['a_z']))
    
    #Measurement equivalent of the true flux (inverse correction with the LS extinction): the flux is now the one that would have been obtained in LS DR9.
    flux_g0 = flux_g0 * (10**(-0.4*3.214*t_ls["EBV"][ihpx]))
    flux_r0 = flux_r0 * (10**(-0.4*2.165*t_ls["EBV"][ihpx]))
    flux_z0 = flux_z0 * (10**(-0.4*1.211*t_ls["EBV"][ihpx]))
    
    #Add the random noise to the truth flux
    flux_g = flux_g0 + (1/np.sqrt(t_ls["TRUE_DEPTH_G"][ihpx]))*np.random.normal(0, 1, len(t_truth))
    flux_r = flux_r0 + (1/np.sqrt(t_ls["TRUE_DEPTH_R"][ihpx]))*np.random.normal(0, 1, len(t_truth))
    flux_z = flux_z0 + (1/np.sqrt(t_ls["TRUE_DEPTH_Z"][ihpx]))*np.random.normal(0, 1, len(t_truth))
    
    #Control positive flux: if negative, very small SNR and the object will be discarded
    flux_g[flux_g<=0] = 1e-12
    flux_r[flux_r<=0] = 1e-12
    flux_z[flux_z<=0] = 1e-12
    
    #Detection according to the Tractor pipeline, performed on non-corrected fluxes
    detect = isdetect(seds, flux_g, flux_r, flux_z, t_ls["TRUE_DEPTH_G"][ihpx], t_ls["TRUE_DEPTH_R"][ihpx], t_ls["TRUE_DEPTH_Z"][ihpx])
    
    #Convert into magnitudes and correct for extinction for the selection
    g = mag_from_flux(flux_g) - 3.214*t_ls["EBV"][ihpx]
    r = mag_from_flux(flux_r) - 2.165*t_ls["EBV"][ihpx]
    z = mag_from_flux(flux_z) - 1.211*t_ls["EBV"][ihpx]

    return g, r, z, detect


def random_MC_noise_inverted(t_truth, seds, ihpx):
    
    """ Same function as above, but two steps are inverted: the operation of inverse-correction of the extinction (*10**(-0.4*R*EBV)) is performed after adding the noise, whereas it is performed
    before in the previous function.
    """
    
    region = t_ls["REGION"][ihpx]
    
    #True flux, corrected from the truth catalog extinction
    flux_g0 = t_truth['g_cmodel_flux_' + region]/3630.78 * (10**(0.4*t_truth['a_g']))
    flux_r0 = t_truth['r_cmodel_flux_' + region]/3630.78 * (10**(0.4*t_truth['a_r']))
    flux_z0 = t_truth['z_cmodel_flux_' + region]/3630.78 * (10**(0.4*t_truth['a_z']))
    
    #Add the random noise to the truth flux
    flux_g = flux_g0 + (1/np.sqrt(t_ls["TRUE_DEPTH_G"][ihpx]))*np.random.normal(0, 1, len(t_truth))
    flux_r = flux_r0 + (1/np.sqrt(t_ls["TRUE_DEPTH_R"][ihpx]))*np.random.normal(0, 1, len(t_truth))
    flux_z = flux_z0 + (1/np.sqrt(t_ls["TRUE_DEPTH_Z"][ihpx]))*np.random.normal(0, 1, len(t_truth))
    
    #Control positive flux: if negative, very small SNR and the object will be discarded
    flux_g[flux_g<=0] = 1e-12
    flux_r[flux_r<=0] = 1e-12
    flux_z[flux_z<=0] = 1e-12
    
    #Remove the extinction correction to yield an output of "measured" fluxes, same as input. THIS OPERATION WAS BEFORE THE ADDITIVE NOISE IN THE FUNcTION random_MC_noise
    flux_g = flux_g * (10**(-0.4*3.214*t_ls["EBV"][ihpx]))
    flux_r = flux_r * (10**(-0.4*2.165*t_ls["EBV"][ihpx]))
    flux_z = flux_z * (10**(-0.4*1.211*t_ls["EBV"][ihpx]))
    
    #Detection according to the Tractor pipeline, performed on non-corrected fluxes
    detect = isdetect(seds, flux_g, flux_r, flux_z, t_ls["TRUE_DEPTH_G"][ihpx], t_ls["TRUE_DEPTH_R"][ihpx], t_ls["TRUE_DEPTH_Z"][ihpx])
    
    #Convert into magnitudes and correct for extinction for the selection
    g = mag_from_flux(flux_g) - 3.214*t_ls["EBV"][ihpx]
    r = mag_from_flux(flux_r) - 2.165*t_ls["EBV"][ihpx]
    z = mag_from_flux(flux_z) - 1.211*t_ls["EBV"][ihpx]

    return g, r, z, detect



# -------------------------------- MONTE-CARLO ALGORITHM --------------------------------------------

def monte_carlo(t, seds, ihpx):
    
    """ Run the Monte-Carlo simulation for the healpix ihpx of the t_ls map
    Returns
    -------
    n_elg0 : array
        Initial number of ELGs LOP in the truth sample
    n_detect : array
        Number of detected objects
    n_elg : array
        Final number of simulated ELGs LOP (detected and selected)
    nz : array
        histogram of the number of ELGs in bins of redshifts (defined by zbins)
    """
    n_elg0 = np.count_nonzero(is_ELG_lop(t["g"], t["r"], t["z"]))
    new_g, new_r, new_z, detect = random_MC_noise(t, seds, ihpx)
    n_detect = np.count_nonzero(detect)
    elgs = np.logical_and(is_ELG_lop(new_g, new_r, new_z), detect)
    nz = np.histogram(t["demp_photoz_best"][elgs], zbins)[0]
    n_elg = np.count_nonzero(elgs)
    
    return n_elg0, n_detect, n_elg, nz


def main(j):
    return monte_carlo(t, SEDs, j)

    
print("\nSimulation in progress...")
SEDs = sed_filters()
zbins = np.arange(0,2, 0.03)
if __name__ == '__main__':
    with Pool(processes=n_processes) as pool:
        res_mc = pool.map(main, range(len(t_ls)))
print("Simulation done")

t_ls["NELG0"] = np.array([res_mc[i][0] for i in range(len(t_ls))]) / ratio_density
t_ls["DETECTED"] = np.array([res_mc[i][1] for i in range(len(t_ls))]) / ratio_density
t_ls["NELG"] = np.array([res_mc[i][2] for i in range(len(t_ls))]) / ratio_density
tab_nz = np.array([res_mc[i][3] for i in range(len(t_ls))]) / ratio_density
print("Table has been updated")

filename = output_path+"MC_elg_lop_eps=0.7_1_withnz"
t_ls.write(filename + ".csv", overwrite=True)
np.save(filename + "_nz.npy", tab_nz)
np.save(filename + "_zbins.npy", zbins)
print(filename)