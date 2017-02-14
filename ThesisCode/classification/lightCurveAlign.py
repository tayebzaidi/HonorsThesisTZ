#!/usr/bin/env python
import sys
import os
import json
import numpy as np
from scipy.signal import find_peaks_cwt
import bisect
import peakutils

def main():
    """
    Parse the lightcurves already in gp_smoothed and gp_smoothed_periodic
    and align them
    """
    #First align the supernovae
    num_failed_shifts, failed_per, failed_per_norm = supernovae_align()
    print("Failed_shifts: ", num_failed_shifts)
    print("Failed per: ", failed_per)
    print("Failed per noralized: ", failed_per_norm)

    #Then the periodics
    #periodic_align()

def supernovae_align():
    """
    Align the supernovae using the first peak in flux.  The 'B' band maximum or equivalent
    is used as the basis for all shifts of the same object. (NOTE: This is not implemented yet)
    The first minimum in magnitude is taken to be zero in phase, and data will be extrapolated
    only from -10 to +50 in phase (or another variant, tbd)

    This will return:
    num_failed_shifts                   -- The number of total failed shifts
    failed_shifts_per_object            -- The average number of total failed shifts per object
    failed_shifts_per_object_normalized -- The average number of total failed shifts 
                                            normalized by the amount of bands in a given object
    As well as output individual files for each lightcurve into gp_smoothed_aligned
    """
    source_folder = '../gen_lightcurves/gp_smoothed/'
    destination = '../gen_lightcurves/gp_smoothed_aligned/'

    lightcurve_filenames = os.listdir(source_folder)

    ## Choose global parameters for the phase restrictions on the aligned lightcurves
    phase_min = -5
    phase_max = 50

    total_failed_shifts = 0
    failed_shifts_per = 0
    failed_shifts_per_norm = 0
    num_objects_processed = 0

    for lightcurve_file in lightcurve_filenames:
        lightcurve_path = source_folder + lightcurve_file
        with open(lightcurve_path, 'r') as f:
            file_data = json.load(f)

        #Prepare the renamed file
        aligned_lightcurve = {}

        #This hack removes the '_gpsmoothed.json' from the string to return the objname
        objname = lightcurve_file[:-16]
        print("Object Name: ", objname)

        #Return the list of keys from the file_data
        data = list(file_data)

        num_shifts = 0
        num_failed_shifts = 0

        for i in range(len(data)):
            filt = data[i]
            mjd = file_data[filt]['mjd']
            mag = file_data[filt]['mag']
            mag_err = file_data[filt]['dmag']
            model_phase = np.array(file_data[filt]['modeldate'])
            model_mag = np.array(file_data[filt]['modelmag'])
            model_err = file_data[filt]['modelerr']
            bspline_mag = file_data[filt]['bsplinemag']
            goodstatus = file_data[filt]['goodstatus']
            kernel_params = file_data[filt]['kernel']
            stype = file_data[filt]['type']

            # WHAT TO DO HERE!
            threshold = 0.5
            minimum_distance = 20
            indexes = peakutils.indexes((1-model_mag), thresh=threshold, min_dist=minimum_distance)

            #Only take the first index if there is one, and align to that
            shift_index = indexes[0]
            #First though, some sanity checking on the first index to make sure
            #If a peak comes more than 50 days after the first recorded observation, throw out
            if model_phase[shift_index] > model_phase[0] + 30:
                num_failed_shifts += 1
                continue

            #Only take the lightcurve if it covers the appropriate range
            if model_phase[shift_index] - model_phase[0] < phase_min:
                num_failed_shifts += 1
                continue
            if model_phase[-1] - model_phase[shift_index] < phase_max:
                num_failed_shifts += 1
                continue
            
            phase_aligned = model_phase - model_phase[shift_index]
            #phase_aligned_original = mjd - model_phase[shift_index]

            # Delete data points outside of the selected range to allow for periodic comparison
            #Because I want to start right at 5 (Hence the -1)
            start_index = bisect.bisect(phase_aligned, phase_min) - 1 
            end_index = bisect.bisect(phase_aligned, phase_max)

            phase_aligned = phase_aligned[start_index:end_index]
            model_mag = model_mag[start_index:end_index]

            #Successful shift
            num_shifts += 1


            aligned_lightcurve[filt] = {'kernel':kernel_params,\
                                        'mjd':mjd,\
                                        'mag':mag,\
                                        'dmag':mag_err,\
                                        'modeldate':phase_aligned.tolist(),\
                                        'modelmag':model_mag.tolist(),\
                                        'modelerr':model_err,\
                                        'bsplinemag':bspline_mag,\
                                        'goodstatus':goodstatus,\
                                        'type': stype}

        with open(destination + objname + '_aligned_lc.json', mode='w') as f:
                json.dump(aligned_lightcurve, f, indent=2, sort_keys=True)

        num_objects_processed += 1
        total_failed_shifts += num_failed_shifts
        #Compute cumulative average
        failed_shifts_per = (num_failed_shifts/num_shifts + \
                                    (num_objects_processed-1)*failed_shifts_per)\
                                    /(num_objects_processed)

    return total_failed_shifts, failed_shifts_per, failed_shifts_per_norm

def periodic_align():
    """
    Align only the periodic lightcurves
    """

if __name__=="__main__":
    sys.exit(main())
