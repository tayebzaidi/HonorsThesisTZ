#!/usr/bin/env python2
import sys
#sys.path.append('/data/antares/aux')
sys.path.append('/mnt/data/antares/aux/sne.space/')
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
from mpi4py import MPI
import pickle


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

    A file is written out per object with the gaussian process smoothed data 
    """
    

    # setup the MPI process, and divide up the files for processing
    # this division is just by number of files, not relative amount of data in each file
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procs_num = comm.Get_size()
    print(procs_num, rank)
    json_files = glob.glob('sne-*/*json')
    nfiles = len(json_files)
    quotient = nfiles/procs_num+1
    P = rank*quotient
    Q = (rank+1)*quotient
    if P > nfiles:                                                                                              
        P = nfiles                                                                                                
    if Q > nfiles:                                                                                              
        Q = nfiles
    print(procs_num, rank, nfiles, quotient, P, Q)

    destination = "/home/antares/nfs_share/tzaidi/HonorsThesisTZ/ThesisCode/gen_lightcurves/gp_smoothed/"
    destination2 = "/mnt/data/antares/aux/sne.space/parsed/"
    dict_list = []

    # setup quantities to test the presence/absence of
    badflags  = ("kcorrected", "scorrected", "mcorrected", "upperlimit", "lowerlimit")
    goodflags = ('time', 'magnitude', 'e_magnitude', 'band')


    # setup the quantities we want to save for each object, as well as default datatypes for each one
    obj_data = OrderedDict()
    obj_data_keys = OrderedDict([('ra',np.nan),('dec',np.nan),('maxdate',np.nan),('redshift', np.nan),('ebv',np.nan),\
            ('host','--'),('hostra',np.nan),('hostdec',np.nan),('hostoffsetang',np.nan),('hostoffsetdist',np.nan),\
            ('maxappmag',np.nan),("maxband",'--')])


    kernelpars = []
    P = int(P)
    Q = int(Q)
    for f in json_files[P:Q]:
    #for f in json_files:
        print(f)
        with open(f, 'r') as j:
            data = json.load(j)
            objname = list(data.keys())[0]
            keys = list(data[objname].keys())
            tempo_dict = {}
            tempo_dict['name'] = objname
            tempo_dict['status'] = 'good'

            # do we have photometry
            if not "photometry" in keys:
                tempo_dict['status'] = 'bad'
                continue

            # this cut is from the histogram the number of observations at the end  - this value isn't known apriori
            # it's just a reasonable value where we aren't discarding too much data 
            # but still making sure the objects we get are well observed
            thisnobs = len(data[objname]['photometry'])
            if thisnobs < 25:
                tempo_dict['status'] = 'bad'
                continue

            # do we have any claimedtype
            if not "claimedtype" in keys:
                tempo_dict['status'] = 'bad'
                continue
        
            # if we have claimed types, check them
            types = OrderedCounter()
            for x in data[objname]['claimedtype']:
                types[x['value']] = len(x['source'].split(','))

            # the claimedtype list is descending order list with more recent claimed type listed first
            # via James Gulliochon (@astrocrash) on twitter 
            ntype = list(types.keys())[0]

            if len(types) == 1:
                # if we have only one claimed type, then as long as it's not bad, we're done
                if check_bad_types(ntype):
                    tempo_dict['status'] = 'cand'
                    continue
            else:
                # if we have multiple claimed types, as long as two or more sources agree
                # on the most recent claimedtype, then accept it
                if types[ntype] >= 2:
                    if check_bad_types(ntype):
                        tempo_dict['status'] = 'cand'
                        continue
                else:
                    # if two sources don't agree on the most recent claimed type
                    # check if three or more sources agree on the most common claimed type
                    most_claims = np.array(types.values()).argmax()
                    nclaims = list(types.values())[most_claims]
                    print("THIS NEEDS MORE ANALYSIS!")
                    print(nclaims)
                    if nclaims >= 3:
                        # we'll accept the most common claimed type as the type then
                        ntype   = list(types)[most_claims]
                        if check_bad_types(ntype):
                            tempo_dict['status'] = 'cand'
                            continue 
                    else:
                        # three sources can't even agree on the most common type, and only one sources claims
                        # the most recent type
                        # we treat that as lack of consensus 
                        #print "MaybeWeird ", objname, ntype, types, types[ntype]
                        tempo_dict['status'] = 'cand'
                        continue

            obj_data[objname] = []
            obj_data[objname].append(objname)
            for key, default_value in obj_data_keys.items():
                if key in data[objname]:
                    value = data[objname][key][0]['value']
                else:
                    value = default_value
                obj_data[objname].append(value)

            thisnobs = len(data[objname]['photometry'])
            time   = []
            band   = []
            mag    = []
            magerr = []
            for obs in data[objname]['photometry']:
                keys = obs.keys()
                if any(key in obs for key in badflags):
                    # photometry is not observer frame or is a lower or upperlimit - skip
                    continue 
                if not all(key in obs for key in goodflags):
                    # photometry is not UV/Optical/NIR - all these keys must be present 
                    continue 
                thisfilt = ''
                if 'telescope' in obs:
                    thisfilt = obs['telescope']
                if 'system' in obs:
                    thisfilt = '_'.join((thisfilt, obs['system']))


                time.append(obs['time'])
                band.append('_'.join((obs['band'].replace('\'','prime').replace(' ','_'),thisfilt)))
                mag.append(obs['magnitude'])
                magerr.append(obs['e_magnitude'])
            if len(time) == 0:
                tempo_dict['status'] = 'nogoodobs'
                continue

            out = np.rec.fromarrays((time, mag, magerr, band), names='time,mag,magerr,pb')
            with open(destination2 +objname+'_lc.dat', 'w') as f:
                f.write(rec2txt(out,precision=8)+'\n')
            tempo_dict['status'] = 'good'
            #print out

            # Do Gaussian Process Fitting right here
            try:
                #Fix the type for each of the arrays sent to the TouchstoneObject
                band = np.array(band)
                tobj = TouchstoneObject(objname, time, mag, magerr, band)
                outbspline = tobj.spline_smooth(per = False, minobs = 10)
                outgp = tobj.gaussian_process_alt_smooth(per = False, scalemin=np.log(25.), scalemax=np.log(5000.), minobs=10)
                outjson = {}
                for filt in outgp:

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
                        print(message)

                    outjson[filt] = {'kernel':list(thisgp.kernel.pars),\
                                        'mjd':thisjd.tolist(),\
                                        'mag':thismag.tolist(),\
                                        'dmag':thisdmag.tolist(),\
                                        'modeldate':mod_dates.tolist(),\
                                        'modelmag':thismod.tolist(),\
                                        'modelerr':thiserr.tolist(),\
                                        'bsplinemag':thismod_bspline.tolist(),\
                                        'goodstatus':goodstatus,\
                                        'type': ntype}
                    kernelpars.append(thisgp.kernel.pars[0])
                if len(outjson.keys()) > 0:    
                    with open(destination + objname+'_gpsmoothed.json', mode='w') as f:
                        json.dump(outjson, f, indent=2, sort_keys=True)
        
            except np.linalg.linalg.LinAlgError as e:
                print(e)
                print("Failed to complete Gaussian Processing")
                continue    

        #close JSON
    #endfor over files




def main():
    GProcessing()









if __name__=='__main__':
    sys.exit(main())
