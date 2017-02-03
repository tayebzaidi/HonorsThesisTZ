"""A script to run wavelet decomposition on arbitrary objects"""
#!/usr/bin/env python
import sys
import json
import featureExtraction
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def main():

    filename = '../gen_lightcurves/CfA_CSP_lightcurves'
    with open(filename, 'r') as f:
        lightcurves = [line.rstrip('\n') for line in f]

    wavelet_coeffs = {}
    stypes = []
    i = 1
    for lightcurve in lightcurves:
        i += 1
        lightcurve_path = '../gen_lightcurves/gp_smoothed/' + lightcurve
        with open(lightcurve_path, 'r') as f:
            file_data = json.load(f)

        #This hack removes the '_gpsmoothed.json' from the string to return the objname
        objname = lightcurve[:-16]
        print(objname)


        ## For now, take only filters starting with B_ (Assuming only one filter with B_)
        for k in list(file_data.keys()):
            if not k.startswith('B_'):
                del file_data[k]
        if len(file_data) == 0:
            continue
        wavelet_coeffs[objname] = {}

        filt = list(file_data.keys())[0]
        mjd = file_data[filt]['mjd']
        mag = file_data[filt]['mag']
        mag_err = file_data[filt]['dmag']
        model_phase = file_data[filt]['modeldate']
        model_mag = file_data[filt]['modelmag']
        bspline_mag = file_data[filt]['bsplinemag']
        goodstatus = file_data[filt]['goodstatus']
        stype = file_data[filt]['type']

        num_coeffs = 10
        raw_coeffs = featureExtraction.general_wavelet_coeffs('bagidis', model_phase,\
                                                             bspline_mag, num_coeffs=num_coeffs)
        wavelet_coeffs[objname]['coeffs'] = raw_coeffs
        wavelet_coeffs[objname]['type'] = stype
        stypes.append(stype)
        print(i)
        #if i > 8:
        #    break


    test_train_data = np.zeros((len(list(wavelet_coeffs)), num_coeffs+1))
    i = 0
    for obj in wavelet_coeffs:
        print(len(wavelet_coeffs[obj]['coeffs']))
        test_train_data[i,0:num_coeffs] = wavelet_coeffs[obj]['coeffs']

        #Assign types and print out the resulting table
        # For now, type Ia -- 0
        # all else --> 1
        if wavelet_coeffs[obj]['type'] == 'Ia':
            test_train_data[i,num_coeffs] = 0
        else:
            test_train_data[i,num_coeffs] = 1
        i += 1

    #Split the data into test and train data
    #First randomize the object order to eliminate bias towards stype
    # and set the seed value to ensure repeatability
    #np.random.seed(40)
    #test_train_data = np.random.permutation(test_train_data)

    #Set proportions to be used for test/train/validation
    #train_prop = 0.7
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

    scores = cross_val_score(forest, test_train_data[:, 0:num_coeffs], test_train_data[:,num_coeffs], cv=5)
    #forest.fit(train[:,0:num_coeffs], train[:,num_coeffs])

    #output = forest.predict(test[:,0:num_coeffs])

    print("Random Forest Regression: ", scores)
    #print(np.sum(output == test[:,num_coeffs]))



    #with open('waveletcoeffs.json', 'w') as out:
    #    json.dump(wavelet_coeffs, out)


if __name__ == "__main__":
    sys.exit(main())
