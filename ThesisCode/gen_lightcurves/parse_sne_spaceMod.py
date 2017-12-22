#!/usr/bin/env python2
import sys
#sys.path.append('/data/antares/aux')
#sys.path.append('/mnt/data/antares/aux/sne.space/')
#sys.path.append('/home/antares/nfs_share/tzaidi/HonorsThesisTZ/ThesisCode/classification/')
sys.path.append('../classification')
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import rec2csv, rec2txt
from astropy.visualization import hist
from collections import Counter, OrderedDict
from ANTARES_object import LAobject
import scipy.interpolate as scinterp
from mpi4py import MPI
import pickle
import bandMap


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
    ^^^ Not the case the lightcurves are not aligned relative to one another 
    This is deemed unnecessary because the wavelet transforms are approximately translation invariant


    A file is written out per object with the gaussian process smoothed data 
    """
    source = "../data/OSC/raw/"
    destination = "../data/OSC/parsed/"    

    # setup the MPI process, and divide up the files for processing
    # this division is just by number of files, not relative amount of data in each file
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procs_num = comm.Get_size()
    print(procs_num, rank)
    sne_files = glob.glob(source + 'sne-*/*json')
    sne_files = np.random.permutation(sne_files)
    nfiles = len(sne_files)
    quotient = nfiles/procs_num+1
    P = rank*quotient
    Q = (rank+1)*quotient
    if P > nfiles:                                                                                              
        P = nfiles                                                                                                
    if Q > nfiles:                                                                                              
        Q = nfiles
    print(procs_num, rank, nfiles, quotient, P, Q)

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
    list_outfiles = []
    P = int(P)
    Q = int(Q)

    lcurve_losses = Counter()

    print("There are a total of {} SNe lightcurves".format(len(sne_files)))
    for num, lcurve in enumerate(sne_files[P:Q]):

        try:

            print("Starting file {:<7}\r".format(num),end="")
            with open(lcurve, 'r') as f:
                data = json.load(f)
            objname = list(data.keys())[0]
            #print(objname)
            keys = list(data[objname].keys())
            tempo_dict = {}
            tempo_dict['name'] = objname
            tempo_dict['status'] = 'good'

            # do we have photometry
            if not "photometry" in keys:
                tempo_dict['status'] = 'bad'
                lcurve_losses.update(['Photometry not in keys'])
                continue

            # this cut is from the histogram the number of observations at the end  - this value isn't known apriori
            # it's just a reasonable value where we aren't discarding too much data 
            # but still making sure the objects we get are well observed
            thisnobs = len(data[objname]['photometry'])
            if thisnobs < 25:
                tempo_dict['status'] = 'bad'
                lcurve_losses.update(['Not enough data in SNe stage (all filters)'])
                continue

            # do we have any claimedtype
            if not "claimedtype" in keys:
                tempo_dict['status'] = 'bad'
                lcurve_losses.update(['No claimed type in SNe stage'])
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
                    lcurve_losses.update(['Bad type with only one type available SNe'])
                    continue
            else:
                # if we have multiple claimed types, as long as two or more sources agree
                # on the most recent claimedtype, then accept it
                if types[ntype] >= 2:
                    if check_bad_types(ntype):
                        tempo_dict['status'] = 'cand'
                        lcurve_losses.update(['Bad types with only two types available SNe'])
                        continue
                else:
                    # if two sources don't agree on the most recent claimed type
                    # check if three or more sources agree on the most common claimed type
                    most_claims = np.array(types.values()).argmax()
                    nclaims = list(types.values())[most_claims]
                    #print("THIS NEEDS MORE ANALYSIS!")
                    #print(nclaims)
                    if nclaims >= 3:
                        # we'll accept the most common claimed type as the type then
                        ntype   = list(types)[most_claims]
                        if check_bad_types(ntype):
                            tempo_dict['status'] = 'cand'
                            lcurve_losses.update(['Bad types with 3 or more types available SNe'])
                            continue 
                    else:
                        # three sources can't even agree on the most common type, and only one sources claims
                        # the most recent type
                        # we treat that as lack of consensus 
                        #print "MaybeWeird ", objname, ntype, types, types[ntype]
                        tempo_dict['status'] = 'cand'
                        lcurve_losses.update(["Three source can't agree on type, lack of consensus"])
                        continue

            thisnobs = len(data[objname]['photometry'])
            time   = []
            band   = []
            mag    = []
            magerr = []
            for obs in data[objname]['photometry']:
                keys = obs.keys()
                if any(key in obs for key in badflags):
                    # photometry is not observer frame or is a lower or upperlimit - skip
                    #lcurve_losses.update(['Bad flags found SNe'])
                    continue 
                if not all(key in obs for key in goodflags):
                    # photometry is not UV/Optical/NIR - all these keys must be present 
                    #lcurve_losses.update(['Not all necessary info found SNe'])
                    continue 
                thisfilt = ''
                if 'telescope' in obs:
                    thisfilt = obs['telescope']
                if 'system' in obs:
                    thisfilt = '_'.join((thisfilt, obs['system']))
                
                time.append(float(obs['time']))
                band.append('_'.join((obs['band'].replace('\'','prime').replace(' ','_'),thisfilt)))
                mag.append(float(obs['magnitude']))
                magerr.append(float(obs['e_magnitude']))
    
            if len(time) == 0:
                tempo_dict['status'] = 'nogoodobs'
                continue

            out = np.rec.fromarrays((time, mag, magerr, band), names='time,mag,magerr,pb')
            #with open(destination2 +objname+'_lc.dat', 'w') as f:
            #    f.write(rec2txt(out,precision=8)+'\n')
            tempo_dict['status'] = 'good'
            #print out

            # Do Gaussian Process Fitting right here
            try:
                #Fix the type for each of the arrays sent to the TouchstoneObject
                band = np.array(band)
                mag = np.array(mag)
                #Fake antares locus ID
                locusID = objname
                #Fake obsids
                obsids = np.array(['a']*len(time))
                
                zp=np.array([27.5]*len(time))
                Z = 27.5 # Set the zeropoint for mag-flux conversion
                flux = 10**(-0.4 * (mag - Z))
                ## Alternate form in base e --> 10^10 * exp(-0.921034 * mag)
                ## Propagation of error formula --> abs(flux * -0.921034 * magerr)
                fluxerr = np.abs(flux * -0.921034 * magerr)


                tobj = LAobject(locusID, objname, time, flux, fluxerr, obsids, band, zp)
                #outbspline = tobj.spline_smooth(per = False, minobs = 6)
                outgp = tobj.gaussian_process_smooth(per = False, minobs=15)
                outjson = {}

                #Only loop over filters that both outgp and outbspline share
                #print("OutGP: ", list(outgp.keys()))
                #print("OutBspline: ", list(outbspline.keys()))
                outfilters = list(set(outgp.keys()))
                #if set(outgp.keys()) != set(outbspline.keys()):
                #    print("Different sets of filters!!!")

                #Should I do bandMapping right here and head off all future difficulties?

                for filt in outfilters:

                    # Generate resampled values from the Gaussian Process regression
                    thisgp, thisjd, thismag, thisdmag = outgp[filt]

                    #I need to choose whether to sample at a frequency or
                    # a fixed number of points

                    ## FOR NOW, I'M CHOOSING A FIXED NUMBER OF POINTS
                    #mod_dates = np.arange(thisjd.min(), thisjd.max(), 1.)
                    ### Using 128 points to allow for multi-level wavelet analysis
                    mod_dates = np.linspace(thisjd.min(), thisjd.max(), 128)

                    thismod, modcovar = thisgp.predict(thismag, mod_dates)
                    thismody, modcovary = thisgp.predict(thismag, thisjd)
                    thiserr = np.sqrt(np.diag(modcovar))

                    # Generate resampled values from the spline model
                    #thisbspline = outbspline[filt]
                    #thismod_bspline = scinterp.splev(mod_dates, thisbspline)
                    #print(thismod_bspline)
                    #orig_val_bspline = scinterp.splev(thisjd, thisbspline)

                    #This is inefficient, but will allow me to subtract the bspline
                    # before re-running the gaussian process regression
                    #temp_passband = np.array([filt] * len(thisjd))
                    #mag_subtracted = thismag - orig_val_bspline
                    #print("Orig_val_bspline: ", orig_val_bspline)
                    #print("Mag sub: ", mag_subtracted)
                    #tobj_subtracted = TouchstoneObject(objname, thisjd, mag_subtracted, thisdmag, temp_passband)
                    #outgp_subtracted = tobj_subtracted.gaussian_process_alt_smooth(per = False, scalemin=np.log(10**-4), scalemax=np.log(10**5), minobs=10)
                    #Since I only gave it values for a single filter, the output will only have one filter in the dictionary
                    #thisgp_subtracted, _, thismag_subtracted, _ = outgp_subtracted[filt]
                    #thismod_subtracted, modcovar_subtracted = thisgp_subtracted.predict(thismag_subtracted, mod_dates)
                    #thiserr_subtracted = np.sqrt(np.diag(modcovar_subtracted))

                    #Re-add back in the b-spline values for the magnitude
                    #thismod_subtracted = thismod_subtracted + thismod_bspline

                    goodstatus = True

                    mad_test = np.median(np.abs(thismody - np.median(thismody)))
                    mad_mod  = np.median(np.abs(thismod  - np.median(thismod )))
                    mad_data = np.median(np.abs(thismag  - np.median(thismag )))
                    
                    if (mad_test - mad_data) > 0.5 or np.abs(mad_mod - mad_data) > 0.5:
                        goodstatus=False
                        message = 'Outlier rejection failed (data: %.3f  model: %.3f  interp: %.3f)'%(mad_data, mad_test, mad_mod)
                        #print(message)
                        #lcurve_losses.update(['Outlier rejection failes'])
                        #continue

                    #Get rid of the straight line approximations
                    if np.ptp(thismod) < 2:
                        continue

                    #print(thisgp.get_parameter_vector())
                    outjson[filt] = {'kernel':list(thisgp.get_parameter_vector()),\
                                        'mjd':thisjd.tolist(),\
                                        'mag':thismag.tolist(),\
                                        'dmag':thisdmag.tolist(),\
                                        'modeldate':mod_dates.tolist(),\
                                        'modelmag':thismod.tolist(),\
                                        'modelerr':thiserr.tolist(),\
                                        #'modelmag_sub':thismod_subtracted.tolist(),\
                                        #'bsplinemag':thismod_bspline.tolist(),\
                                        'goodstatus':goodstatus,\
                                        'type': ntype}
                    kernelpars.append(thisgp.get_parameter_vector()[0])
                outjson_mapped = bandMap.remapBands(outjson)
                #print(outjson_mapped.keys())
                #print(outjson.keys())
                if len(outjson_mapped.keys()) > 0:
                    list_outfiles.append(objname + '_gpsmoothed.json')
                    with open(destination + objname+'_gpsmoothed.json', mode='w') as f:
                        json.dump(outjson_mapped, f, indent=2, sort_keys=True)
                else:
                    lcurve_losses.update(['No keys in output json'])
        
            except np.linalg.linalg.LinAlgError as e:
                print(e)
                print("Failed to complete Gaussian Processing")
                lcurve_losses.update(['Linear algebra error in GP processing'])
                continue
        except:
            continue
        #close JSON
    #endfor over files
    with open(destination + 'LCURVES.LIST', mode='w') as outfile:
        outfile.write("\n".join(map(str, list_outfiles)))
    
    loss_file = 'sne_losses.json'
    with open(loss_file, mode='w') as lfile:
        json.dump(lcurve_losses, lfile)




def main():
    GProcessing()









if __name__=='__main__':
    sys.exit(main())
