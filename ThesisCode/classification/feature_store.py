"""A script to run wavelet decomposition on all SNe and store them"""
#!/usr/bin/env python
import os
import sys
import json
import glob
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
    num_band_coeffs = hyperparams['num_band_coeffs']
    wavelet_type = hyperparams['wavelet_type']
    wavelet_level = hyperparams['wavelet_level']

    num_bands = 3
    num_coeffs = num_band_coeffs * num_bands
    num_classes = 2

    SNe_lightcurve_directory = '../data/OSC/parsed/'
    #periodic_lightcurve_directory = '../data/OGLE/parsed/'
    all_lightcurves = glob.glob(SNe_lightcurve_directory + '*.json')
    print(all_lightcurves)

    #Define metrics for seeing lightcurve loss
    deleted_lightcurves = 0
    analyzed_lightcurves = 0

    wavelet_coeffs = {}
    #object_types = []

    #Master data structure for later PCA reduction if wavelet_type not bagidis
    for i,lightcurve_path in enumerate(all_lightcurves):

        with open(lightcurve_path, 'r') as f:
            lightcurve_mapped = json.load(f)

        #HACK to get lightcurve name (remove '_gpsmoothed.json')
        objname = os.path.basename(lightcurve_path)[:-16]
        print('Working on objnum {}\r'.format(i),end="")

        #print(list(file_data.keys()))

        deleted_filters = 0
        ## For now, take only filter 'g' and 'i'

        req_filters = set(['g','r','i'])

        if req_filters.issubset(set(lightcurve_mapped.keys())):
            for filt in list(lightcurve_mapped.keys()):
                if filt not in ['g','r','i']:
                    deleted_filters += 1
                    print("Deleted {}".format(filt))
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


    print("Deleted lightcurves: ", deleted_lightcurves)
    print("Analyzed Lightcurves: ", len(wavelet_coeffs))

    #Write all lightcurve parameters to a file (json format)
    with open(output_file, 'w') as output:
        json.dump(wavelet_coeffs, output, sort_keys=True, indent=2)
    
    #if training_set == "rep":
    #    return n_coeffs, objs
    #return n_coeffs

def return_pca(wavelet_coeffs, num_components=False):
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

    if num_components:
        pca = PCA(n_components=num_components)
        pca.fit(X)
        X_pca = pca.transform(X)
    else:
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

    return wav_coeffs, X_pca.shape[1]

def extract_rand_select(wavelet_coeffs, num_lightcurves=10000):
    num_objects = len(wavelet_coeffs)
    num_object_coeffs = len(wavelet_coeffs[list(wavelet_coeffs)[0]]['coeffs'])
    print("Number of Object coeffs: ", num_object_coeffs)
    print("Number of used lcurves: ", num_lightcurves)
    # Plus 2 for the type and the object number
    all_coeffs = np.zeros((num_objects, num_object_coeffs+2))
    print(all_coeffs.shape)
    all_types = []

    for i, obj in enumerate(wavelet_coeffs):
        all_coeffs[i,0:num_object_coeffs] = wavelet_coeffs[obj]['coeffs']
        all_types.append(wavelet_coeffs[obj]['type'])
        all_coeffs[i,num_object_coeffs] = wavelet_coeffs[obj]['type']
        all_coeffs[i,num_object_coeffs+1] = obj
    
    np.random.permutation(all_coeffs)
    all_coeffs = all_coeffs[0:num_lightcurves,:]

    wav_coeffs = {}
    objs = np.zeros((num_lightcurves))
    print(all_coeffs.shape)
    for i in range(all_coeffs.shape[0]):
        obj = all_coeffs[i,num_object_coeffs+1]
        #print(objs[i])
        objs[i] = obj
        wav_coeffs[obj] = {}
        wav_coeffs[obj]['coeffs'] = all_coeffs[i,0:num_object_coeffs]
        wav_coeffs[obj]['type'] = all_coeffs[i, num_object_coeffs]
        print(all_coeffs[i,num_object_coeffs])
    return wav_coeffs, objs

def extract_select(wavelet_coeffs, ignore_objects):
    num_objects = len(wavelet_coeffs)
    num_object_coeffs = len(wavelet_coeffs[list(wavelet_coeffs)[0]]['coeffs'])
    print("Number of Object coeffs: ", num_object_coeffs)
    print("Number of used objects", len(wavelet_coeffs)-len(ignore_objects))
    # Plus 2 for the type and the object number

    wav_coeffs_out = {}
    for i, obj in enumerate(wavelet_coeffs):
        #Ignore previously generated lightcurves
        if obj in ignore_objects:
            continue
        wav_coeffs_out[obj] = {}
        wav_coeffs_out[obj]['coeffs'] = wavelet_coeffs[obj]['coeffs']
        wav_coeffs_out[obj]['type'] = wavelet_coeffs[obj]['type']
    
    return wav_coeffs_out

def label_class(wavelet_coeffs, num_coeffs):
    feature_class_array = np.zeros((len(list(wavelet_coeffs)), num_coeffs+1))
    type_array = []

    for i, obj in enumerate(wavelet_coeffs):
        #print(len(wavelet_coeffs[obj]['coeffs']))
        feature_class_array[i,0:num_coeffs] = wavelet_coeffs[obj]['coeffs']

        type_1a = ['Ia', 1]
        type_2 = ['II', 2, 21, 22]
        type_1bc = ['Ib', 'Ib/c', 'Ic', 3, 32, 33]

        type_array.append(wavelet_coeffs[obj]['type'])

        if wavelet_coeffs[obj]['type'] in type_1a:
            feature_class_array[i, num_coeffs] = 1
        elif wavelet_coeffs[obj]['type'] in (type_2 or type_1bc):
            feature_class_array[i, num_coeffs] = 0
        else:
            feature_class_array[i, num_coeffs] = 0

    #print(Counter(type_array))
    #print(feature_class_array)
    return feature_class_array




if __name__ == "__main__":
    hp = {'wavelet_type': 'sym2', 'wavelet_level': 1, 'num_band_coeffs': 10,'non_representative': True}
    sys.exit(generate_decomposition(hp))
