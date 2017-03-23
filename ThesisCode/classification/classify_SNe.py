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

    num_bands = 3
    num_coeffs = num_band_coeffs * num_bands
    num_classes = 5

    #Load the wavelet coefficients from the specified file
    with open(input_file, 'r') as f:
        wavelet_coeffs = json.load(f)

    test_train_data = np.zeros((len(list(wavelet_coeffs)), num_coeffs+1))
    #print(test_train_data)

    for i, obj in enumerate(wavelet_coeffs):
        #print(len(wavelet_coeffs[obj]['coeffs']))
        test_train_data[i,0:num_coeffs] = wavelet_coeffs[obj]['coeffs']

        #Assign types and print out the resulting table
        # For now, type Ia -- 0
        # all else --> 1
        #if wavelet_coeffs[obj]['type'] == 'Ia':
        #    test_train_data[i, num_coeffs] = 0
        #elif wavelet_coeffs[obj]['type'] == 'II':
        #    test_train_data[i, num_coeffs] = 1
        #else:
        #    test_train_data[i, num_coeffs] = 2

        if wavelet_coeffs[obj]['type'] == 'Ia':
            test_train_data[i, num_coeffs] = 0
        elif wavelet_coeffs[obj]['type'] == 'II':
            test_train_data[i, num_coeffs] = 1
        elif wavelet_coeffs[obj]['type'] == 'Ib':
            test_train_data[i, num_coeffs] = 2
        elif wavelet_coeffs[obj]['type'] == 'Ic':
            test_train_data[i, num_coeffs] = 3
        else:
            test_train_data[i, num_coeffs] = 4

    #print([wavelet_coeffs[obj]['type'] for obj in wavelet_coeffs])
    #print(test_train_data[:, num_coeffs])

    #Split the data into test and train data
    #First randomize the object order to eliminate bias towards object_type
    # and set the seed value to ensure repeatability
    #np.random.seed(40)
    test_train_data = np.random.permutation(test_train_data)

    X = test_train_data[:,0:num_coeffs]
    y = np.ravel(test_train_data[:,num_coeffs])
    #print(X,y)
    #print(X.shape, y.shape)

    #y = label_binarize(y, classes=[0,1,2])
    
    #Set proportions to be used for test/train/validation
    test_prop = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)
    #test_prop = 1 - train_prop

    #print(X_train.shape, y_train.shape)

    #Use the proportions to calculate row indices
    #split_idx = int(test_train_data.shape[0] * train_prop)

    #train = test_train_data[0:split_idx,:]
    #test = test_train_data[split_idx:,:]

    #print(train.shape)
    #print(test.shape)
    #print(train[:,num_coeffs].shape)

    #Setup the Random Forest Classifier
    forest = RandomForestClassifier(n_estimators=100)

    #scores = cross_val_score(forest, test_train_data[:, 0:num_coeffs], test_train_data[:,num_coeffs], cv=10)
    #print(y_train.shape)
    #print(np.ravel(y_train).shape)
    #y_train = np.ravel(y_train)
    forest.fit(X_train, y_train)

    output = forest.predict_proba(X_test)
    y_score = forest.predict(X_test)

    print(output)

    # Compute Precision-Recall and plot curve
    #precision = dict()
    #recall = dict()
    #average_precision = dict()
    #for i in range(num_classes):
    #    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
    #                                                        y_score[:, i])
    #    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    #precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    #average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

    #print("Random Forest Regression: ", scores)
    #print(np.sum(y_score == y_test))
    #print(output)

    #print(y_score)
    #print(y_test)
    accuracy = np.sum(y_score == y_test)/len(y_test)
    precisions = np.zeros(num_classes)
    recalls = np.zeros(num_classes)
    fscores = np.zeros(num_classes)

    #Calculation adapted from Michelle Lochner's code for classification
    #Using 0.5 as threshold for classification
    for chosen_class in range(num_classes):
        print(chosen_class)
        Y_bool = (y_test == chosen_class)
        preds = (output[:,chosen_class] >= 0.5)
        #print(Y_bool)
        #print(preds)

        TP = (preds & Y_bool).sum()
        FP = (preds & ~Y_bool).sum()
        TN = (preds & ~Y_bool).sum()
        FN = (~preds & Y_bool).sum()

        print("TP: ", TP)
        print("FP: ", FP)
        print("TN: ", TN)

        precision = TP/(TP + FP)
        recall = TP/(TP+FN)
        precisions[chosen_class] = precision
        recalls[chosen_class] = recall
        fscores[chosen_class] = 2 * precision * recall / (precision + recall)

    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_fscore = np.mean(fscores)
    print(precisions, avg_precision)
    print(recalls, avg_recall)
    print(fscores, avg_fscore)
    #print(accuracy)

    #with open('waveletcoeffs.json', 'w') as out:
    #    json.dump(wavelet_coeffs, out)
    return avg_fscore


if __name__ == "__main__":
    hp = {'num_band_coeffs': 10}
    sys.exit(classify_supernovae(hp))
