"""Manages and automates the classification for nested classification or plain classification"""
import json
import os, sys
import feature_store
import classify_SNe
import classify_nested
import matplotlib.pyplot as plt
import pickle

def run_pipeline(wavelet_type, wavelet_level, coeffs_file, outfile, num_band_coeffs, nested_cross_val=False):

    hyperparams = {'wavelet_type': wavelet_type,
                    'num_band_coeffs': num_band_coeffs,
                    'wavelet_level': wavelet_level}
    

    if not os.path.isfile(coeffs_file):
        num_components = feature_store.generate_decomposition(hyperparams, output_file=coeffs_file)

    if nested_cross_val: 
        scores, estimators = classify_nested.classify_supernovae(hyperparams, input_file=coeffs_file)
        for estimator in estimators:
            print(estimator.best_params_)
        estimator_file = wavelet_type+'_'+wavelet_level+'_'+'estimators.p'
        with open(estimator_file, 'wb') as f:
            pickle.dump(estimators, f)
            #print(estimator.clf)
    else:
        results = classify_SNe.classify_supernovae(hyperparams, input_file=coeffs_file)
        print(results[0:3])
        with open(outfile, 'w') as f:
            json.dump(results, f)


if __name__=="__main__":
    wavelet_types = ['bagidis', 'sym2', 'db2']

    for wavelet_type in wavelet_types:
        print("RUNNING FOR ", wavelet_type)
        print("")
        wavelet_level = 1
        coeffs_file = wavelet_type + "_" + str(wavelet_level) + '_coeffs.json'
        outfile = wavelet_type+"_"+str(wavelet_level)+"_results.json"
        num_band_coeffs = 10
        nested_cross_val = False
        run_pipeline(wavelet_type, wavelet_level, coeffs_file, outfile, num_band_coeffs)
