"""A script to run wavelet decomposition on all SNe and store them"""
#!/usr/bin/env python
import os
import sys
import json
import featureExtraction
import numpy as np
import bandMap

def generate_decomposition(hyperparams, output_file='wavelet_coeffs.json'):
    """
    Generates a file with the coefficients stored in the json format
    Arguments:
        output_file -- Name of output file (str)
        hyperparams -- Dictionary containing:
            wavelet_type  -- Type of wavelet decomposition (bagidis, a_trous)
            wavelet_level -- Levels of wavelet decomposition (only relevant for a_trous)
            num_band_coeffs    -- Number of coefficients from the wavelet decomposition (per band)
    Returns:
        Output file stored on disk (we'll have to see about size optimizations)
    """

    #Get hyperparameter values
    num_band_coeffs = hyperparams['num_band_coeffs']
    wavelet_type = hyperparams['wavelet_type']
    wavelet_level = hyperparams['wavelet_level']


    num_bands = 4
    num_coeffs = num_band_coeffs * num_bands
    num_classes = 3

    lightcurve_directory = '../gen_lightcurves/'

    #Define metrics for seeing lightcurve loss
    deleted_lightcurves = 0
    analyzed_lightcurves = 0

    #filename = '../gen_lightcurves/CfA_CSP_lightcurves'
    #with open(filename, 'r') as f:
    #    #FIX THIS HACK LATER
    #    SNe_lightcurves = [line[:-17] + '_gpsmoothed.json' for line in f]
    with open('des_sn.json', 'r') as f:
        SNe_lightcurves = json.load(f)

    wavelet_coeffs = {}
    object_types = []

    for lightcurve in SNe_lightcurves:
        
        lightcurve_mapped = SNe_lightcurves[lightcurve]

        #This hack removes the '_gpsmoothed.json' from the string to return the objname
        objname = lightcurve
        #print(objname)

        #print(list(file_data.keys()))

        deleted_filters = 0
        ## For now, take only filter 'g'

        req_filters = set(['g','r','i','z'])

        if req_filters.issubset(set(lightcurve_mapped.keys())):
            for filt in list(lightcurve_mapped.keys()):
                if filt not in ['g', 'r', 'i','z']:
                    deleted_filters += 1
                    #print("Deleted {}".format(filt))
                    del lightcurve_mapped[filt]
            analyzed_lightcurves += 1
        else:
            #print("Does not contain all bands")
            deleted_lightcurves += 1
            continue
        #print("{} filters deleted".format(deleted_filters))

        if len(lightcurve_mapped) == 0:
            print("No values in the file")
            continue

        wavelet_coeffs[objname] = {}
        all_coeffs = np.zeros((num_coeffs,))

        for i, filt in enumerate(lightcurve_mapped):
            #mjd = lightcurve_mapped[filt]['mjd']
            #mag = lightcurve_mapped[filt]['mag']
            #mag_err = lightcurve_mapped[filt]['dmag']
            model_phase = lightcurve_mapped[filt]['modeldate']
            model_mag = lightcurve_mapped[filt]['modelmag']
            #bspline_mag = file_data[filt]['bsplinemag']
            #goodstatus = lightcurve_mapped[filt]['goodstatus']
            object_type = lightcurve_mapped[filt]['type']

            raw_coeffs = featureExtraction.general_wavelet_coeffs(wavelet_type, model_phase,\
                                                                model_mag, num_coeffs=num_band_coeffs)
            #Unravel the different filters by appending the information
            #print("Left: ", i*num_band_coeffs)
            #print("Right: ", (i+1)*num_band_coeffs)
            #print(raw_coeffs.reshape(num_band_coeffs))
            all_coeffs[i*num_band_coeffs:(i+1)*num_band_coeffs] = raw_coeffs.reshape(num_band_coeffs)

        #print(all_coeffs)
        wavelet_coeffs[objname]['coeffs'] = all_coeffs.tolist()
        #print(raw_coeffs)
        wavelet_coeffs[objname]['type'] = object_type
        #print(object_type)
        object_types.append(object_type)
        #print(i)
        #if i > 8:
        #    break
    #Write all lightcurve parameters to a file (json format)
    with open(output_file, 'w') as output:
        json.dump(wavelet_coeffs, output, sort_keys=True, indent=2)

if __name__ == "__main__":
    hp = {'wavelet_type': 'bagidis', 'wavelet_level': 1, 'num_band_coeffs': 10}
    sys.exit(generate_decomposition(hp))