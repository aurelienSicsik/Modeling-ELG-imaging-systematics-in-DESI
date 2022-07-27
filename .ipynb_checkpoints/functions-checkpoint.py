#!/usr/bin/env python

import numpy as np
import healpy as hp

from astropy.table import Table
import matplotlib.pyplot as plt
from __future__ import division, print_function
import subprocess
import sys
from desiutil.plots import prepare_data, init_sky, plot_grid_map, plot_healpix_map, plot_sky_circles, plot_sky_binned


"""This script contains useful functions to manipulate catalogs: to match them and to visualize the data. """


# --------- MATCH CATALOGS ------------------------------------

def match_hpx_index(t1, t2, quantities2match, name1):
    """Match t1 and t2 healpixels
    quantities2match is a list of strings with the names of the quantities that have to be matched over the 2 tables
    The final matched catalog is t2"""
    
    healpix_to_index_in_1 = { h:i for i,h in enumerate(t1["HPXPIXEL"])}
    index_of_2_in_1 = [ healpix_to_index_in_1[h] for h in t2["HPXPIXEL"]]
    for quant in quantities2match:
        t2[quant + "_in_" + name1] = t1[quant][index_of_2_in_1]
    return

def match_targetID_index(t1, t2, quantities2match, name1):
    """Match t1 and t2 targets
    quantities2match is a list of strings with the names of the quantities that have to be matched over the 2 tables
    The final matched catalog is t2"""
    
    tid_to_index_in_1 = { h:i for i,h in enumerate(t1["TARGETID"])}
    index_of_2_in_1 = [ tid_to_index_in_1[h] for h in t2["TARGETID"]]
    for quant in quantities2match:
        t2[quant + "_in_" + name1] = t1[quant][index_of_2_in_1]
    return


# --------- PLOT FUNCTIONS ------------------------------------


def compute_density_bin(density,  x_quantity, min_x, max_x, binsize, mean_density, minbin):
    """Density bins of a Healpix catalogs, based on the x_quantity value. Used to obtain the density fluctuations vs. depth plots
    Parameters
    ----------
    density : :class: array
        array of density values
    x_quantity : :class: array
        array of the quantity used to bin the density (same order as density)
    min_x, max_x : :class: float
        minimum and maximum values used to bin
    binsize : :class: float
        size of the bins
    mean_density: :class: float
        average density that is used to normalize the density fluctuations
    minbin: :class: int
        minimum number of pixels in a bin
        
    Returns
    -------
    x_axis : :class:`array`
        x_axis of bins with >minbin healpixels
    mean_value : :class:`array`
        mean value of the density over the bin
    std_value : :class:`array`
        standard deviation of the density over the bin
    nb_pix : :class:array
        nb of healpixels in the bin
    """
    
    x_axis = []
    mean_value = []
    std_value = []
    nb_pix = []
    current_x = min_x
    while True:
        ii = np.where((x_quantity >= current_x) & (x_quantity < current_x+binsize))
        densities = density[ii]
        if densities.size>minbin:
            x_axis.append(current_x+binsize/2)
            mean_value.append(np.mean(densities)/mean_density - 1)
            std_value.append((np.std(densities)/mean_density)/np.sqrt(len(densities)))
            nb_pix.append(densities.size)
        current_x += binsize
        if current_x > max_x:
            break
    return x_axis, mean_value, std_value, nb_pix

def compute_density_bin_with_hpxlist(table,  x_quantity, min_x, max_x, binsize, mean_density):
    """Same functions as above, but returns 
    Parameters
    ----------
    same as compute_density_bin
        
    Returns
    -------
    same as compute_density_bin
    + hpx_list : class: array
        return the list of corresponding healpixels in each bin
    """
    
    x_axis = []
    mean_value = []
    std_value = []
    nb_pix = []
    current_x = min_x
    hpx_list = []
    while True:
        ii = np.where((x_quantity >= current_x) & (x_quantity < current_x+binsize))
        densities = table["N_ELG_LOP"][ii]/pix_area
        if densities.size>50:
            x_axis.append(current_x+binsize/2)
            hpx_list.append(table["HPXPIXEL"][ii])
            mean_value.append(np.mean(densities)/mean_density - 1)
            std_value.append((np.std(densities)/mean_density)/np.sqrt(len(densities)))
            nb_pix.append(densities.size)
        current_x += binsize
        if current_x > max_x:
            break
    return x_axis, mean_value, std_value, nb_pix, hpx_list


def plot_curve_density(x_axis, mean_value, std_value, x_min, x_max, y_min, y_max, color, label, linestyle='solid'):
    """ Plot the binned density fluctuations, the parameters are the output of the compute_density_bin function + plotting parameters (x_min,x_max,...)"""
    plt.hlines(0., x_min, x_max, linestyles='dashed')
    top_line = [mean_value[i] + std_value[i] for i in range(len(mean_value))]
    bottom_line = [mean_value[i] - std_value[i] for i in range(len(mean_value))]
    #plt.plot(x_axis, top_line, color=color)
    plt.plot(x_axis, mean_value, color=color, label=label, linestyle=linestyle)
    #plt.plot(x_axis, bottom_line, color=color)
    plt.fill_between(x_axis, bottom_line, top_line, color=color, alpha=0.2)
    plt.ylim(y_min,y_max)
    

def plot_histogram_bin(x_axis, nb_pix, binsize, color):
    """ Plot the histogram with the number of healpixels in each bin, the parameters are the output of the compute_density_bin function"""
    plt.bar(x_axis, nb_pix, width=binsize, align='center', color=color, alpha=0.3)
    
    
def plot_map(hpx, quantity,  nside, vmin, vmax):
    """ Plot a healpix map with nside parameter in a beautiful way. But very long to compute with nside>256 !
    Parameters
    ----------
    hpx : :class: array
        healpix catalog
    quantity : :class: array
        the quantity to plot for each healpixel (same order as hpx)
    nside : :class: int
        nside value of the Healpix map
    vmin, vmax : :class: float
        minimum and maximum values of the colorbar for quantity"""
    
    h2i={h:i for i,h in enumerate(hpx)}
    all_healpixels=np.arange(hp.nside2npix(nside))
    ii=np.array([h2i[h] if h in h2i else -1 for h in all_healpixels])
    values = np.zeros(all_healpixels.size)
    values[ii>=0] = quantity[ii[ii>=0]]
    values[ii<0] = np.nan
    values[values<vmin]=vmin
    values[values>vmax]=vmax
    ax = plot_healpix_map(values, True, galactic_plane_color=None, ecliptic_plane_color=None,cmap='jet')
    

    
# --------- OTHER FUNCTIONS ------------------------------------

# Anand Raichoor's code for checking if ra, dec are in the DES footprint
def get_isdes(ra, dec, nside):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymangle"])
    import pymangle
    import healpy as hp
    
    npix = hp.nside2npix(nside)
    # checking hp pixels
    mng = pymangle.Mangle('/global/cfs/cdirs/desi/users/asicsik/first-day/maps-data/des.ply')
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)
    hpra, hpdec= 180./np.pi*phi, 90.-180./np.pi*theta
    hpindes = (mng.polyid(hpra, hpdec)!=-1).astype(int)
    # pixels with all neighbours in des
    hpindes_secure = np.array([i for i in range(npix) if hpindes[i]+hpindes[hp.get_all_neighbours(nside, i)].sum()==9])
    # pixels with all neighbours outside des
    hpoutdes_secure = np.array([i for i in range(npix) if hpindes[i]+hpindes[hp.get_all_neighbours(nside, i)].sum()==0])
    # hpind to be checked
    tmp = np.ones(npix, dtype=bool)
    tmp[hpindes_secure] = False
    tmp[hpoutdes_secure]= False
    hp_tbc = np.arange(npix)[tmp]

    # now checking indiv. obj. in the tbc pixels
    hppix = hp.ang2pix(nside, (90.-dec)*np.pi/180., ra*np.pi/180., nest=False)
    hpind = np.unique(hppix)

    isdes = np.zeros(len(ra), dtype=bool)
    isdes[np.in1d(hppix, hpindes_secure)] = True
    tbc = np.where(np.in1d(hppix, hp_tbc))[0]
    tbcisdes = (mng.polyid(ra[tbc], dec[tbc])!=-1)
    isdes[tbc][tbcisdes] = True

    return isdes