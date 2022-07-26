# Modeling-ELG-imaging-systematics-in-DESI

This github repository contains the python scripts and Jupyter notebooks that compose my work during my internship at the LBNL. It consisted in creating a simple forward-modeling Monte-Carlo simulation to model the effect of imaging systematics on the Emission Line Galaxy (ELG) target selection of DESI. The code can be run on NERSC with the DESI environment.

# The simulation

The idea of the simulation is to add random depth and seeing-related noise to the flux of objects coming from a truth sample. The detection and selection processes are then applied to obtain a target sample of ELGs, which can be compared to the real ELG selection of the Legacy Survey (see "The DESI Emission Line Galaxy sample: target selection and validation", A. Raichoor et al., in prep.). 

# Description


 * The files `Generate_` are used to generate either the Healpix map with observing parameters of the Legacy Survey (`Generate_LS_hpx_map.py`) or the truth samples used as an input in the Monte-Carlo simulation, for both truth catalogs DECAM and HSC in the COSMOS region (`Generate_truth_DECAM.py`, `Generate_truth_HSC.py`). These truth catalogs are described respectively in the notebooks `Describe_truth_DECAM.ipynb` and `Describe_truth_HSC.ipynb`.

 * The Monte-Carlo simulation is the main program of this repository. It can be run with the script `MC_DECAM_parallelized.py` or `MC_HSC_parallelized.py` depending on the truth sample used. The algorithm is parallelized and can be run on Cori or Perlmutter (NERSC). The computation time is of several minutes for the whole footprint of the Legacy Survey and about 50k objects in the truth sample. In the case of HSC, the final n(z) distribution is also included as an output of the simulation.
The results of the simulation can be visualized and compared to the real DR9 density maps in the notebook `visualize_ELG.ipynb`. One can also look at the new distribution of objects when the Monte-Carlo is run on ony one healpixel in the notebooks `MC_1hpx_DECAM.ipynb` and `MC_1hpx_HSC.ipynb`.

 * Finally, the output density map can be used to compute weights for the correction of imaging systematics in the correlation function. The files `compute_weight_map.py` and `assign_weights.py` create the Healpix weights and associate each object in DESI Guadalupe data with its weight. The data files of Guadalupe (LSS catalog, DA02) are not yet public but can be found in the DESI database. The weights are computed according to: https://desi.lbl.gov/trac/wiki/ClusteringWG/LSScat/DA02main/current_version#Weights.
The correlation function has been computed using the code of https://github.com/desihub/LSS/blob/master/scripts/xirunpc.py, which has been adapted.


# Acknowledgements
This research used resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility operated under Contract No. DE-AC02-05CH11231.
