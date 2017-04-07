"""Manages and automates the classification for nested classification or plain classification"""
import json
import os, sys
import feature_store
import classify_SNe
import classify_nested
import matplotlib.pyplot as plt
import pickle

def main():
    wavelet_type = 'sym2'
    wavelet_level = 1
    training_set = "total"
    coeffs_file = training_set + "_" + wavelet_type + "_" + str(wavelet_level) + '_coeffs.json'
    
    num_band_coeffs = 10
    nested_cross_val = True
    #Choosing training set from "rep", "non_rep", "total"


    hyperparams = {'wavelet_type': wavelet_type,
                    'num_band_coeffs': num_band_coeffs,
                    'wavelet_level': wavelet_level,
                    'training_set': training_set}
    

    if not os.path.isfile(coeffs_file):
        if training_set != "rep":
            num_components = feature_store.generate_decomposition(hyperparams, output_file=coeffs_file)
        else:
            num_components, ignore_objs = feature_store.generate_decomposition(hyperparams, output_file=coeffs_file)

    if training_set == "non_rep":
        coeffs_file_test = "test_" + coeffs_file
        if not os.path.isfile(coeffs_file_test):
            hyperparams_nonrep = dict(hyperparams)
            hyperparams_nonrep['training_set'] = "test_non_rep"
            hyperparams_nonrep['num_components'] = num_components
            feature_store.generate_decomposition(hyperparams_nonrep, output_file=coeffs_file_test)
    elif training_set == "rep":
       coeffs_file_test = "test_" + coeffs_file
       if not os.path.isfile(coeffs_file_test):
            hyperparams_nonrep = dict(hyperparams)
            hyperparams_nonrep['training_set'] = "test_rep"
            hyperparams_nonrep['num_components'] = num_components
            hyperparams_nonrep['already_stored'] = ignore_objs
            feature_store.generate_decomposition(hyperparams_nonrep, coeffs_file_test)


    if nested_cross_val: 
        scores, estimators = classify_nested.classify_supernovae(hyperparams, input_file=coeffs_file)
        for estimator in estimators:
            print(estimator.best_params_)
        estimator_file = wavelet_type+'_'+wavelet_level+'_'+'estimators.p'
        with open(estimator_file, 'wb') as f:
            pickle.dump(estimators, f)
            #print(estimator.clf)
    else:
        if training_set != "total":
            fpr, tpr, auc_val, ranking = classify_SNe.classify_supernovae(hyperparams, input_file=coeffs_file)
            print(ranking)
            print(auc_val)
            plt.plot(fpr, tpr)
            plt.show()
            plt.hist(ranking, len(ranking), normed=1)
            plt.show()
        else:
            fpr, tpr, auc_val, ranking = classify_SNe.classify_supernovae(hyperparams, input_file=coeffs_file)
            print(ranking)
            print(auc_val)
            plt.plot(fpr, tpr)
            plt.show()


if __name__=="__main__":
    sys.exit(main())
