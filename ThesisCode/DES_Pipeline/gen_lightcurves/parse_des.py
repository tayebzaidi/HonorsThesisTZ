#!/usr/bin/env python2
import sys
#sys.path.append('/data/antares/aux')
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import rec2csv, rec2txt
from astropy.visualization import hist
from collections import Counter, OrderedDict
from ANTARES_object import TouchstoneObject
import scipy.interpolate as scinterp



# since the claimedtype in the sne.space data is ordered by time (newest claimedtype first)
# it makes sense to store this, and keep a count of how many studies agree with that type
# this effectively decides what the final classification should be
# since, of course, people don't actually agree on type, despite the spectra
class OrderedCounter(Counter, OrderedDict):
    """
    trivial implementation of an ordered counter
    """
    pass


def check_bad_types(ntype):
    if ntype == 'Candidate' or\
        ntype.endswith('?') or\
        ntype =='I' or\
        ntype.startswith('Star') or\
        ntype.startswith('CV') or\
        ntype.startswith('AGN') or\
        ntype.startswith('LBV') or\
        ntype == 'Radio':
        return True
    else:
        return False

def GProcessing():
    """
    This method does the heavy lifting of actually processing all the sne.space lightcurves
    Each lightcurve is read in parallel with MPI, and has to pass various cuts
    A dictionary of all the objects is built up, containing auxillary information on the object
    as well as the status of processing and the output of the processing 

    If it fails the cuts, the object is not used, and simply marked as failed
    
    If it passes the cuts, a gaussian process is used to attempt to smooth the light curve in each band
    Individual bands are treated separately, and allowed to fail independent of other bands
    
    If all the bands fail, the object is marked as having failed, even if it did pass the cuts 
    (as no useful data was extracted) 

    We attempt to align the lightcurves in an absolute sense (i.e. max to fixed phase)
    rather than relative to each other (as this processing is done in parallel, and we don't have that info)

    A single json file is written out with the gaussian process smoothed data
    """
    

    # setup the MPI process, and divide up the files for processing
    # this division is just by number of files, not relative amount of data in each file
    
    #Set up final json file
    des_sn = {}
    outfile = 'des_sn.json'


    #Generate dictionary of all SN types from key file
    base_path = '../gen_lightcurves/DES_lcurves/DES_BLIND+HOSTZ/'
    key_file = '../gen_lightcurves/DES_lcurves/TEST+HOST.KEY'
    with open(key_file, 'r') as f:
        data = f.readlines()
    SN_key = {}
    for line in data:
        if line.startswith('NVAR') or line.startswith('VARNAMES'):
            continue
        #Only need 2nd 3rd and 4th element
        _, sn_id, sntype, confirm_type, genz, hostz, hostzerr = line.split()
        SN_key[int(sn_id)] = {'sntype': int(sntype), 'confirm_type': int(confirm_type),\
                        'genz': float(genz), 'hostz': float(hostz), 'hostzerr': float(hostzerr)}


    lightcurves = os.listdir(base_path)

    for i,lightcurve in enumerate(lightcurves):
        #Eliminate the three header files
        base_header = 'DES_BLIND+HOSTZ'
        if lightcurve in [base_header+'.README', base_header+'.IGNORE',base_header+'.LIST']:
            continue
        
        tobj = TouchstoneObject.fromfile(base_path + lightcurve)
        
        mwebv = tobj.header['mwebv']
        
        #Look up the types for future analysis
        object_key = SN_key[int(tobj.objectname)]
        
        sntype = object_key['sntype']
        confirm_type = object_key['confirm_type']
        hostz = object_key['hostz']
        hostzerr = object_key['hostzerr']
        genz = object_key['genz']
        
        
        outbspline = tobj.spline_smooth(per = False, minobs = 6)
        outgp = tobj.gaussian_process_alt_smooth(per = False, scalemin=np.log(10**-4), scalemax=np.log(10**5), minobs=6)
        outjson = {}

        #Only loop over filters that both outgp and outbspline share
        #print("OutGP: ", list(outgp.keys()))
        #print("OutBspline: ", list(outbspline.keys()))
        #print(outgp.keys(),outbspline.keys())
        outfilters = list(set(outgp.keys()) & set(outbspline.keys()))
        if set(outgp.keys()) != set(outbspline.keys()):
            print("Filter difference between bspline and GP")

        for filt in outfilters:

            # Generate resampled values from the Gaussian Process regression
            thisgp, thisjd, thismag, thisdmag = outgp[filt]
            mod_dates = np.arange(thisjd.min(), thisjd.max(), 1.)
            thismod, modcovar = thisgp.predict(thismag, mod_dates)
            thismody, modcovary = thisgp.predict(thismag, thisjd)
            thiserr = np.sqrt(np.diag(modcovar))

            # Generate resampled values from the spline model
            thisbspline = outbspline[filt]
            thismod_bspline = scinterp.splev(mod_dates, thisbspline)

            goodstatus = True

            mad_test = np.median(np.abs(thismody - np.median(thismody)))
            mad_mod  = np.median(np.abs(thismod  - np.median(thismod )))
            mad_data = np.median(np.abs(thismag  - np.median(thismag )))

            if (mad_test - mad_data) > 0.5 or np.abs(mad_mod - mad_data) > 0.5:
                goodstatus=False
                message = 'Outlier rejection failed (data: %.3f  model: %.3f  interp: %.3f)'%(mad_data, mad_test, mad_mod)
                #print(message)

            outjson[filt] = {'kernel':list(thisgp.kernel.pars),\
                            'mjd':thisjd.tolist(),\
                            'mag':thismag.tolist(),\
                            'dmag':thisdmag.tolist(),\
                            'modeldate':mod_dates.tolist(),\
                            'modelmag':thismod.tolist(),\
                            'modelerr':thiserr.tolist(),\
                            'bsplinemag':thismod_bspline.tolist(),\
                            'goodstatus':goodstatus,\
                            'hostz': hostz,\
                            'hostzerr': hostzerr,\
                            'confirm_type': confirm_type,\
                            'type': sntype}
        if len(outjson.keys()) == 0:
            continue
        des_sn[sn_id] = outjson

    with open(outfile, mode='w') as f:
        json.dump(des_sn, f, indent=2, sort_keys=True)


        #close JSON
    #endfor over files




def main():
    GProcessing()









if __name__=='__main__':
    sys.exit(main())
