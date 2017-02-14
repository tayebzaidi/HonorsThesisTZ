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
    #period_database = '/mnt/data/antares/aux/OGLE/'
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

    destination = "/home/antares/nfs_share/tzaidi/HonorsThesisTZ/ThesisCode/gen_lightcurves/gp_smoothed_periodic/" 

    kernelpars = []
    P = int(P)
    Q = int(Q)

    print(dat_files)

    for f in dat_files[P:Q]:
    #for f in dat_files:

        #Print object name
        print(f)

        #Load in the lightcurve using np.genfromtxt (there are faster options if necessary later)
        #The names of the columns are "HJD, mag, dmag, pb"

        #The pandas.read_csv is by far faster than the numpy implementation
        #lightcurve = np.genfromtxt(f,dtype=['d','d', 'd', 'U1'], names=True)
        lightcurve = pd.read_csv(f, delim_whitespace=True)
        #Only take the 'I' band (That was all that was highlighted in the 'good_*.json files)
        band_mask = (lightcurve['pb'] == 'I')
        passband = lightcurve[band_mask]['pb']
        hjd = lightcurve[band_mask]['HJD']
        mag = lightcurve[band_mask]['mag']
        mag_err = lightcurve[band_mask]['dmag']
        #The [:-4] eliminates the '.dat' from the end of the path
        objname = os.path.basename(f)[:-4]
        print("Object name:", objname)

        # The periodic type is the directory just prior to the file (hence the [-2])
        f = os.path.normpath(f)
        periodic_type = f.split(os.sep)[-2]

        #For now these objects are not being input as periodic objects
        # because they are already parsed
        tobj = TouchstoneObject(objname, hjd, mag, mag_err, passband, per=False)
        # I may need to mess around more with the parameters to get smooth curves
        outgp = tobj.gaussian_process_alt_smooth(per=True, scalemin=np.log(25.),
                                                 scalemax=np.log(5000.), minobs=10)
        outjson = {}

        for filt in outgp:

            #Generate resampled values from the Gaussian Process regression
            thisgp, thisjd, thismag, thisdmag = outgp[filt]
            print("Thisjd:", thisjd)
            print("Length of Thisjd: ", len(thisjd))
            print("Min: ", thisjd.min())
            print("Max: ", thisjd.max())

            print("Thismag: ", thismag)
            print("Length of thismag: ", len(thismag))
            mod_dates = np.arange(0, 1, 0.01)
            thismod, modcovar = thisgp.predict(thismag, mod_dates)
            thismody, modcovary = thisgp.predict(thismag, thisjd)
            thiserr = np.sqrt(np.diag(modcovar))

            #Rescale the resampled dates values to a 0-1 phase
            print(mod_dates)
            goodstatus = True

            mad_test = np.median(np.abs(thismody - np.median(thismody)))
            mad_mod  = np.median(np.abs(thismod  - np.median(thismod )))
            mad_data = np.median(np.abs(thismag  - np.median(thismag )))
            
            if (mad_test - mad_data) > 0.5 or np.abs(mad_mod - mad_data) > 0.5:
                goodstatus=False
                message = 'Outlier rejection failed (data: %.3f  model: %.3f  interp: %.3f)'%(mad_data, mad_test, mad_mod)
                print(message)

            outjson[filt] = {'kernel':list(thisgp.kernel.pars),\
                                'mjd':thisjd.tolist(),\
                                'mag':thismag.tolist(),\
                                'dmag':thisdmag.tolist(),\
                                'modeldate':mod_dates.tolist(),\
                                'modelmag':thismod.tolist(),\
                                'modelerr':thiserr.tolist(),\
                                'goodstatus':goodstatus,\
                                'type': periodic_type}
            kernelpars.append(thisgp.kernel.pars[0])

        
        if len(outjson.keys()) > 0:
            with open(destination + objname+'_gpsmoothed.json', mode='w') as f:
                json.dump(outjson, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    sys.exit(periodicProcessing())
