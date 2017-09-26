"""Manages and automates the classification for nested classification or plain classification"""
import json
import os, sys
import feature_store
import classify_SNe
import classify_nested
import matplotlib.pyplot as plt
import pickle
import pywt

def run_pipeline(wavelet_type, wavelet_level, coeffs_file, outfile, num_band_coeffs, nested_cross_val=False):

    hyperparams = {'wavelet_type': wavelet_type,
                    'num_band_coeffs': num_band_coeffs,
                    'wavelet_level': wavelet_level}
    

    if not os.path.isfile(coeffs_file):
        num_components = feature_store.generate_decomposition(hyperparams, output_file=coeffs_file)

    if nested_cross_val:
        if wavelet_type == 'bagidis':
            estimator_file = wavelet_type+'_1_estimators.p'
        else:
            estimator_file = wavelet_type+'_'+str(wavelet_level)+'_estimators.p'
        
        if not os.path.isfile(estimator_file):
            scores, estimators, original_data = classify_nested.classify_supernovae(hyperparams, input_file=coeffs_file)
            for estimator in estimators:
                print(estimator.best_params_)
                print(estimator.error_score)
                print(estimator.best_score_)

            print("Storing estimators")
            dump_file = [estimators, original_data, scores]
            with open(estimator_file, 'wb') as f:
                pickle.dump(dump_file, f)
                #print(estimator.clf)
    else:
        results = classify_SNe.classify_supernovae(hyperparams, input_file=coeffs_file)
        print(results[0:3])
        with open(outfile, 'w') as f:
            json.dump(results, f)


if __name__=="__main__":
    #wavelet_types = ['sym2', 'db2', 'bagidis']
    num_points = 128
    wavelet_types = ['bagidis', 'db2', 'sym2']
    max_level = pywt.swt_max_level(num_points)
    wavelet_levels = [max_level, 1]

    for wavelet_type in wavelet_types:
        for wavelet_level in wavelet_levels:
            print("RUNNING FOR ", wavelet_type, wavelet_level)
            print("")

            if wavelet_type == 'bagidis':
                coeffs_file = wavelet_type + '_1_coeffs.json'
                outfile = wavelet_type+'_1_results.json'
            else:
                coeffs_file = wavelet_type + '_' + str(wavelet_level) +'_coeffs.json'
                outfile = wavelet_type+"_"+str(wavelet_level)+"_results.json"

            num_band_coeffs = 50
            nested_cross_val = True
            run_pipeline(wavelet_type, wavelet_level, coeffs_file, outfile, num_band_coeffs, nested_cross_val=nested_cross_val)
