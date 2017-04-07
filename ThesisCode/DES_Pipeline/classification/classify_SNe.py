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
    training_set = hyperparams['training_set']

    #Load the wavelet coefficients from the specified file
    with open(input_file, 'r') as f:
        wavelet_coeffs = json.load(f)

    if training_set != "total":
        #Hack to get rid of "non_rep_" at start of name
        all_coeffs_file = "test_" + input_file
        with open(all_coeffs_file, 'r') as f:
            wavelet_coeffs_all = json.load(f)
        

    num_coeffs = len(wavelet_coeffs[list(wavelet_coeffs)[0]]['coeffs'])
    num_classes = 2


    test_train_data = label_class(wavelet_coeffs, num_coeffs)

    if training_set != "total":
        test_all = label_class(wavelet_coeffs_all, num_coeffs)
        X_all = test_all[:,0:num_coeffs]
        y_all = np.ravel(test_all[:,num_coeffs])
        print(X_all.shape, y_all.shape)
        print("Num Type Ia", sum(y_all==1))
        print("Num Non-Ia", sum(y_all==0))

    #Split the data into test and train data
    #First randomize the object order to eliminate bias towards object_type
    # and set the seed value to ensure repeatability
    #np.random.seed(40)

    test_train_data = np.random.permutation(test_train_data)

    X = test_train_data[:,0:num_coeffs]
    y = np.ravel(test_train_data[:,num_coeffs])
    print("Num Type Ia", sum(y==1))
    print("Num Non-Ia", sum(y==0))


    if wavelet_type == 'bagidis':
        pipeline = Pipeline([
                        ('reduce_dim', SelectKBest(k=24)),
                        ('classify', RandomForestClassifier())
        ])
    else:
        #Do PCA for non-bagidis wavelets
        pipeline = Pipeline([
                        ('reduce_dim', PCA(n_components=24)),
                        ('classify', RandomForestClassifier())
        ])
    
    #Set proportions to be used for test/train/validation
    test_prop = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop)

    #print(X_train.shape, y_train.shape)

    #Setup the Random Forest Classifier
    if training_set != "total":
        print("Training Random forest")
        pipeline.fit(X_train, y_train)
        output = pipeline.predict_proba(X_test)
        y_pred = pipeline.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, output[:,1])
        acccuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_val = roc_auc_score(y_test, output[:,1])

    else:
        print("Training Random forest for total")
        forest = RandomForestClassifier(n_estimators=500, oob_score=True)
        forest.fit(X_train, y_train)
        output = forest.oob_decision_function_
        ranking = forest.feature_importances_
        fpr, tpr, thresholds = roc_curve(y_test, output[:,1])
        auc_val = roc_auc_score(y_test, output[:,1])
    #print(y_train.shape)
    #print(np.ravel(y_train).shape)
    #y_train = np.ravel(y_train)
    #forest.fit(X_train, y_train)

    #output = forest.predict_proba(X_test)
    #y_score = forest.predict(X_test)

    #print(output)

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
    '''accuracy = np.sum(y_score == y_test)/len(y_test)
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
    #print(accuracy)'''

    #with open('waveletcoeffs.json', 'w') as out:
    #    json.dump(wavelet_coeffs, out)
    return fpr, tpr, auc_val, ranking


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

    print(Counter(type_array))
    return feature_class_array

if __name__ == "__main__":
    hp = {'num_band_coeffs': 10, 'wavelet_type': 'sym2'}
    sys.exit(classify_supernovae(hp))
