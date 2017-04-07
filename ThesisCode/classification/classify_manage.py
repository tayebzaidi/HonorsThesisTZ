import feature_store
import classify_nested
import classify_SNe
import os, json, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

def main():
    wavelet_type = 'db2'
    wavelet_level = 1
    coeffs_file = wavelet_type + str(wavelet_level) + '_coeffs.json'
    
    num_band_coeffs = 10
    nested_cross_val = True

    hyperparams = {'wavelet_type': wavelet_type,
                    'num_band_coeffs': num_band_coeffs,
                    'wavelet_level': wavelet_level}
    

    if not os.path.isfile(coeffs_file):
        feature_store.generate_decomposition(hyperparams, output_file=coeffs_file)
    
    if nested_cross_val: 
        estimators = classify_nested.classify_supernovae(hyperparams, input_file=coeffs_file)

        for estimator in estimators:
            print(estimator.best_params_)
            #print(estimator.feature_importances_)
        with open('estimators.p', 'wb') as f:
            pickle.dump(estimators, f)
    else:
        scores = classify_SNe.classify_supernovae(hyperparams, input_file=coeffs_file)
        #plt.plot(fpr, tpr)
        #plt.show()
        #print(fpr, tpr, auc)
        print(scores)
        print(scores.mean())


if __name__=="__main__":
    sys.exit(main())
    
