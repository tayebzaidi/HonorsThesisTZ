import sys
import os
import json
import numpy as np
from ANTARES_object import TouchstoneObject
import scipy.interpolate as scinterp
from mpi4py import MPI
import pickle
import pandas as pd

def periodicProcessing():
    """
    This method does the equivalent that parse_sne_... does but for the periodic
    lightcurves.  Currently, it only takes the subsample of good lightcurves selected
    by Monika Soraisam for the ANTARES demo
    Each lightcurve is smoothed with a gaussian process

    Each band is treated separately and can fail separately
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procs_num = comm.Get_size()
    print(procs_num, rank)

    #File selection (for now just the subset)
    raw_lightcurves_path = '/mnt/data/antares/aux/OGLE/parsed/'
    good_periodic_list = '/home/antares/nfs_share/tzaidi/HonorsThesisTZ/ThesisCode/gen_lightcurves/goodPeriodic/'
    good_per_lightcurve_types = os.listdir(good_periodic_list)
    print("Good types: ", good_per_lightcurve_types)
    dat_files = []
    #Types where the name in the 'good_*.json' file differs from the actual file
    check_types = ['dsct', 'acep', 'dpv']
    for periodic_type_file in good_per_lightcurve_types:
        #Eliminates the 'good_' and the '.json' to leave only the type
        periodic_type = periodic_type_file[5:][:-5]
        type_directory = os.path.join(raw_lightcurves_path, periodic_type)

        good_lightcurves_path = good_periodic_list + periodic_type_file
        print(good_lightcurves_path)
        with open(good_lightcurves_path) as f:
            good_filenames = json.load(f)
        for lightcurve_name in good_filenames:
            # bands = list(good_filenames[key].keys())
            if periodic_type in check_types:
                #Fix the lightcurve_name so that it matches the actual file name
                lightcurve_name = lightcurve_name.lower().replace('-','_').replace('ogle', 'ogle3')
            #print(lightcurve_name)
            dat_files.append(os.path.join(type_directory, lightcurve_name + '.dat'))

    #print(dat_files)
    nfiles = len(dat_files)
    quotient = nfiles/procs_num+1
    P = rank*quotient
    Q = (rank+1)*quotient
    if P > nfiles:                                                                                              
        P = nfiles                                                                                                
    if Q > nfiles:                                                                                              
        Q = nfiles
    print(procs_num, rank, nfiles, quotient, P, Q)

    destination = "/home/antares/nfs_share/tzaidi/HonorsThesisTZ/ThesisCode/gen_lightcurves/gp_smoothed/" 

    kernelpars = []
    P = int(P)
    Q = int(Q)
    for f in dat_files[P:Q]:
    #for f in dat_files:
        #Load in the lightcurve using np.genfromtxt (there are faster options if necessary later)
        #The names of the columns are HJD, mag, dmag, pb
        #lightcurve = np.genfromtxt(f,dtype=['d','d', 'd', 'U1'], names=True)
        lightcurve = pd.read_csv(f, delim_whitespace=True)
        #Only take the 'I' band (That was all that was highlighted in the 'good_*.json files)
        band_mask = (lightcurve['pb'] == 'I')
        #lightcurve[band_mask]['HJD']

if __name__=="__main__":
    sys.exit(periodicProcessing())