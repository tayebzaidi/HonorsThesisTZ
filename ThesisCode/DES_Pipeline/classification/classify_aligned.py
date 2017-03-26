"""A script to run wavelet decomposition on arbitrary objects"""
#!/usr/bin/env python
import os
import sys
import json
import featureExtraction
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

def main():

    #Define constants for analysis
    num_coeffs = 10

    filename = '../gen_lightcurves/CfA_CSP_lightcurves'
    with open(filename, 'r') as f:
        #FIX THIS LATER
        
        SNe_lightcurves = [line[:-17] + '_aligned_lc.json' for line in f]
    
    all_aligned = os.listdir('../gen_lightcurves/gp_smoothed_aligned/')
    periodic_lightcurves = [lcurve for lcurve in all_aligned if lcurve.lower().startswith('ogle')]

    all_lightcurves = periodic_lightcurves + SNe_lightcurves
    print(all_lightcurves)

    periodic_types = ['dpv', 'lpv', 'rrlyr', 'ecl', 'dcep', 'acv', 'acep', 'dsct']
    wavelet_coeffs = {}
    object_types = []
    i = 1
    for lightcurve in all_lightcurves:
        i += 1
        lightcurve_path = '../gen_lightcurves/gp_smoothed_aligned/' + lightcurve

        if not os.path.isfile(lightcurve_path):
            print("Cant find {}".format(lightcurve_path))
            continue

        with open(lightcurve_path, 'r') as f:
            file_data = json.load(f)

        #This hack removes the '_gpsmoothed.json' from the string to return the objname
        objname = lightcurve[:-16]
        #print(objname)

        #print(list(file_data.keys()))

        ## For now, take only filters starting with B_ (Assuming only one filter with B_)
        for k in list(file_data.keys()):
            if (not k.startswith('B_')) and (not k == 'I'):
                del file_data[k]
        if len(file_data) == 0:
            print("No values in the file")
            continue
        wavelet_coeffs[objname] = {}

        filt = list(file_data.keys())[0]
        mjd = file_data[filt]['mjd']
        mag = file_data[filt]['mag']
        mag_err = file_data[filt]['dmag']
        model_phase = file_data[filt]['modeldate']
        model_mag = file_data[filt]['modelmag']
        #bspline_mag = file_data[filt]['bsplinemag']
        goodstatus = file_data[filt]['goodstatus']
        object_type = file_data[filt]['type']

        raw_coeffs = featureExtraction.general_wavelet_coeffs('bagidis', model_phase,\
                                                             model_mag, num_coeffs=num_coeffs)
        wavelet_coeffs[objname]['coeffs'] = raw_coeffs
        print(raw_coeffs)
        wavelet_coeffs[objname]['type'] = object_type
        #print(object_type)
        object_types.append(object_type)
        #print(i)
        #if i > 8:
        #    break


    test_train_data = np.zeros((len(list(wavelet_coeffs)), num_coeffs+1))
    print(test_train_data)
    i = 0
    for obj in wavelet_coeffs:
        #print(len(wavelet_coeffs[obj]['coeffs']))
        test_train_data[i,0:num_coeffs] = wavelet_coeffs[obj]['coeffs']

        #Assign types and print out the resulting table
        # For now, type Ia -- 0
        # all else --> 1
        if wavelet_coeffs[obj]['type'] == 'Ia':
            test_train_data[i, num_coeffs] = 1
        elif wavelet_coeffs[obj]['type'] in periodic_types:
            test_train_data[i, num_coeffs] = 0
        else:
            test_train_data[i, num_coeffs] = 1

        i += 1
    print(test_train_data[:, num_coeffs])
    #Split the data into test and train data
    #First randomize the object order to eliminate bias towards object_type
    # and set the seed value to ensure repeatability
    #np.random.seed(40)
    test_train_data = np.random.permutation(test_train_data)

    X = test_train_data[:,0:(num_coeffs-1)]
    y = test_train_data[:,num_coeffs]

    y = label_binarize(y, classes=[0,1])
    
    #Set proportions to be used for test/train/validation
    test_prop = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)
    #test_prop = 1 - train_prop

    #Use the proportions to calculate row indices
    #split_idx = int(test_train_data.shape[0] * train_prop)

    #train = test_train_data[0:split_idx,:]
    #test = test_train_data[split_idx:,:]

    #print(train.shape)
    #print(test.shape)
    #print(train[:,num_coeffs].shape)

    #Setup the Random Forest Classifier
    forest = RandomForestClassifier(n_estimators = 100)

    #scores = cross_val_score(forest, test_train_data[:, 0:num_coeffs], test_train_data[:,num_coeffs], cv=10)
    forest.fit(X_train, y_train)

    output = forest.predict_proba(X_test)
    y_score = forest.predict(X_test)

    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)

    #print("Random Forest Regression: ", scores)
    print(np.sum(y_score == y_test))
    print(output)
    print(y_score)
    print(y_test)

    print(average_precision)

    #with open('waveletcoeffs.json', 'w') as out:
    #    json.dump(wavelet_coeffs, out)


if __name__ == "__main__":
    sys.exit(main())
