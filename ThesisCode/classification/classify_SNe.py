"""A script to classify supernovae from wavelet decomposition"""
#!/usr/bin/env python
import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

def classify_supernovae(hyperparams, input_file='wavelet_coeffs.json'):
    """
    Takes a file with the coefficients stored in the json format
    Arguments:
        input_file -- Name of output file (str)
        hyperparams -- Dictionary containing:
            num_band_coeffs -- Number of coefficients from the wavelet decomposition (per band)
    Returns:
        tuple of (accuracy, avg_precision, avg_recall, avg_fscore)
    """

    #Get hyperparameter values
    num_band_coeffs = hyperparams['num_band_coeffs']
    wavelet_type = hyperparams['wavelet_type']


    #Load the wavelet coefficients from the specified file
    with open(input_file, 'r') as f:
        wavelet_coeffs = json.load(f)
        

    num_coeffs = len(wavelet_coeffs[list(wavelet_coeffs)[0]]['coeffs'])
    num_classes = 2


    test_train_data, objnames = label_class(wavelet_coeffs, num_coeffs)

    #print(test_train_data.shape)
    arr = np.arange(test_train_data.shape[0])
    #print(arr)
    permutation_idx = np.random.permutation(arr)
    #print(permutation_idx)

    test_train_data = test_train_data[permutation_idx]
    #print(test_train_data.shape)
    objnames = objnames[permutation_idx]

    X = test_train_data[:,0:num_coeffs]
    y = np.ravel(test_train_data[:,num_coeffs])
    print("Num Type Ia", sum(y==1))
    print("Num Non-Ia", sum(y==0))


    if wavelet_type == 'bagidis':
        pipeline = Pipeline([
                        ('reduce_dim', SelectKBest(k=24)),
                        ('classify', RandomForestClassifier(n_estimators=500, oob_score=True))
        ])
    else:
        #Do PCA for non-bagidis wavelets
        pipeline = Pipeline([
                        ('reduce_dim', PCA(n_components=24)),
                        ('classify', RandomForestClassifier(n_estimators=500, oob_score=True))
        ])
    
    #Set proportions to be used for test/train/validation
    #test_prop = 0.2
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)


    print("Training Random forest")
    print("Num Objects: ", X.shape[0])
    print("Num Features: ", X.shape[1])

    #print(dir(pipeline))
    pipeline.fit(X, y)
    output = pipeline._final_estimator.oob_decision_function_
    y_pred = np.around(output[:,1])
    fpr, tpr, thresholds = roc_curve(y, output[:,1])
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_val = roc_auc_score(y, output[:,1])

    misclassified_idxs = np.where(y != y_pred)
    misclassified = objnames[misclassified_idxs]
    print(misclassified)

    results = [accuracy, f1, auc_val, tpr.tolist(), fpr.tolist(), thresholds.tolist(), misclassified.tolist()]
    return results


def label_class(wavelet_coeffs, num_coeffs):
    feature_class_array = np.zeros((len(list(wavelet_coeffs)), num_coeffs+1))
    type_array = []
    objnames = []

    for i, obj in enumerate(wavelet_coeffs):
        #print(len(wavelet_coeffs[obj]['coeffs']))
        feature_class_array[i,0:num_coeffs] = wavelet_coeffs[obj]['coeffs']

        type_1a = ['Ia', 1]
        type_2 = ['II', 2, 21, 22]
        type_1bc = ['Ib', 'Ib/c', 'Ic', 3, 32, 33]

        if wavelet_coeffs[obj]['type'] in type_1a:
            feature_class_array[i, num_coeffs] = 1
        elif wavelet_coeffs[obj]['type'] in (type_2 or type_1bc):
            feature_class_array[i, num_coeffs] = 0
        else:
            feature_class_array[i, num_coeffs] = 0

        type_array.append(wavelet_coeffs[obj]['type'])
        objnames.append(obj)

    print(Counter(type_array))
    objnames = np.array(objnames)
    return feature_class_array, objnames

if __name__ == "__main__":
    hp = {'num_band_coeffs': 10, 'wavelet_type': 'sym2'}
    sys.exit(classify_supernovae(hp))
