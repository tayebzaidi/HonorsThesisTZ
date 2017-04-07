"""A script to run wavelet decomposition on all SNe and store them"""
#!/usr/bin/env python
import os
import sys
import json
import featureExtraction
import numpy as np
import bandMap
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

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
    wavelet_type = hyperparams['wavelet_type']
    num_band_coeffs = hyperparams['num_band_coeffs']
    wavelet_level = hyperparams['wavelet_level']


    num_bands = 3
    #num_coeffs = num_band_coeffs * num_bands
    #num_classes = 3

    lightcurve_directory = '../gen_lightcurves/gp_smoothed/'

    #Define metrics for seeing lightcurve loss
    deleted_lightcurves = 0
    analyzed_lightcurves = 0

    #filename = '../gen_lightcurves/CfA_CSP_lightcurves'
    #with open(filename, 'r') as f:
    #    #FIX THIS HACK LATER
    #    SNe_lightcurves = [line[:-17] + '_gpsmoothed.json' for line in f]
    SNe_lightcurves = os.listdir(lightcurve_directory)

    wavelet_coeffs = {}
    object_types = []


    print("Iterating over", len(SNe_lightcurves), "lightcurves for feature storing into ", output_file)
    for lightcurve in SNe_lightcurves:
        lightcurve_path = lightcurve_directory + lightcurve

        if not os.path.isfile(lightcurve_path):
            print("Cant find {}".format(lightcurve_path))
            continue

        with open(lightcurve_path, 'r') as f:
            #print(lightcurve)
            file_data = json.load(f)

        #This hack removes the '_gpsmoothed.json' from the string to return the objname
        objname = lightcurve[:-16]
        #print(objname)

        #print(list(file_data.keys()))

        deleted_filters = 0
        ## For now, take only filter 'g'
        lightcurve_mapped = file_data
        #print(list(lightcurve_mapped.keys()))

        req_filters = set(['g','r','i'])

        if req_filters.issubset(set(lightcurve_mapped.keys())):
            for filt in list(lightcurve_mapped.keys()):
                if filt not in ['g', 'r', 'i']:
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
        
        all_coeffs = featureExtraction.general_wavelet_coeffs(lightcurve_mapped, wavelet_type, num_levels=wavelet_level, num_band_coeffs=num_band_coeffs)

                #print(all_coeffs)
        wavelet_coeffs[objname]['coeffs'] = all_coeffs.tolist()
        #print(raw_coeffs)
        wavelet_coeffs[objname]['type'] = lightcurve_mapped['g']['type']
        #print(object_type)
        #object_types.append(object_type)
        #print(i)
        #if i > 8:
        #    break

    ### Run PCA during classification!!!
    #If the SWT is chosen, use PCA to reduce dimensionality
    #if wavelet_type != 'bagidis':
    #    print("Running PCA on data")
    #    wavelet_coeffs = return_pca(wavelet_coeffs)
    
    print("Deleted lightcurves: ", deleted_lightcurves)

    #Write all lightcurve parameters to a file (json format)
    with open(output_file, 'w') as output:
        json.dump(wavelet_coeffs, output, sort_keys=True, indent=2)


def return_pca(wavelet_coeffs):
    """
    Take the entire wavelet_coeffs structure with all objects and reduce the dimensionality
    using PCA
    """
    num_objects = len(wavelet_coeffs)
    num_object_coeffs = len(wavelet_coeffs[list(wavelet_coeffs)[0]]['coeffs'])
    print("Number of Object coeffs: ", num_object_coeffs)
    all_coeffs = np.zeros((num_objects, num_object_coeffs))
    print(all_coeffs.shape)
    all_types = []

    for i, obj in enumerate(wavelet_coeffs):
        all_coeffs[i,:] = wavelet_coeffs[obj]['coeffs']
        all_types.append(wavelet_coeffs[obj]['type'])
    
    X = all_coeffs
    #Center
    X = scale(X)
    #print(X.mean(axis=0))
    #print(X.std(axis=0))

    pca = PCA()
    pca.fit(X)
    X_pca = pca.transform(X)

    #Following Lochner et al., take up to tol=0.98 (98% of the variation)
    tol = 0.98
    var_ratios = pca.explained_variance_ratio_

    total_variance = 0
    for i, variance in enumerate(var_ratios):
        total_variance += variance
        if total_variance >= tol:
            coeff_idx = i
            break
    
    X_pca = X_pca[:,0:coeff_idx+1]
    print(X_pca.shape)

    #print(X_pca.shape)

    #Wrap back up into the wavelet_coeffs format
    wav_coeffs = {}
    for i, obj in enumerate(wavelet_coeffs):
        wav_coeffs[obj] = {}
        wav_coeffs[obj]['coeffs'] = X_pca[i,:].tolist()
        wav_coeffs[obj]['type'] = all_types[i]

    return wav_coeffs

if __name__ == "__main__":
    hp = {'wavelet_type': 'sym2', 'wavelet_level': 1, 'num_band_coeffs': 10}
    sys.exit(generate_decomposition(hp))
