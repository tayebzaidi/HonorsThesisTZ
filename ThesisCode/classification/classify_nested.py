"""A script to classify supernovae from wavelet decomposition"""
#!/usr/bin/env python
import os
import sys
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

n_cpu = multiprocessing.cpu_count() + 2


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

    #Load the wavelet coefficients from the specified file
    with open(input_file, 'r') as f:
        wavelet_coeffs = json.load(f)

    wavelet_type = hyperparams['wavelet_type']
    if wavelet_type != 'bagidis':
        example_entry = wavelet_coeffs[list(wavelet_coeffs)[0]]
        num_coeffs = len(example_entry['coeffs'])
    else:
        num_bands = 3
        num_coeffs = num_band_coeffs * num_bands
    num_classes = 2

    test_train_data = np.zeros((len(list(wavelet_coeffs)), num_coeffs+1))
    print(num_coeffs)

    for i, obj in enumerate(wavelet_coeffs):

        test_train_data[i,0:num_coeffs] = wavelet_coeffs[obj]['coeffs']

        type_1a = ['Ia', 1]
        type_2 = ['II', 2, 21, 22]

        if wavelet_coeffs[obj]['type'] in type_1a:
            test_train_data[i, num_coeffs] = 1
        elif wavelet_coeffs[obj]['type'] in type_2:
            test_train_data[i, num_coeffs] = 0
        else:
            test_train_data[i, num_coeffs] = 0

    #print([wavelet_coeffs[obj]['type'] for obj in wavelet_coeffs])
    #print(test_train_data[:, num_coeffs])

    #Split the data into test and train data
    #First randomize the object order to eliminate bias towards object_type
    # and set the seed value to ensure repeatability
    #np.random.seed(40)
    test_train_data = np.random.permutation(test_train_data)

    X = test_train_data[:,0:num_coeffs]
    y = np.ravel(test_train_data[:,num_coeffs])

    #X = X[0:100,:]
    #y = y[0:100]

    #print(X[0:10,:])

    #Rescale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #X = scaler.transform(X)

    #print(X[0:10,:])

    #Implement PCA on both the BAGIDIS coefficients and the SWT coeffs
    #Modeled after nested CV and dim reduction with Pipeline examples for sklearn
    NUM_TRIALS = 5
    verbosity=0

    if wavelet_type == 'bagidis':
        pipeline = Pipeline([
                        ('reduce_dim', SelectKBest()),
                        ('classify', RandomForestClassifier())
        ])
        # Set up possible values of parameters to optimize over
        num_feature_options = range(10, 30, 2)
        p_grid = {
                    'reduce_dim__k': num_feature_options,
                    'classify__n_estimators': range(300, 700, 100)
        }
    else:
        #Do PCA for non-bagidis wavelets
        pipeline = Pipeline([
                        ('reduce_dim', PCA()),
                        ('classify', RandomForestClassifier())
        ])
        # Set up possible values of parameters to optimize over
        num_feature_options = range(10, 30, 2)
        p_grid = {
                    'reduce_dim__n_components': num_feature_options,
                    'classify__n_estimators': range(300, 700, 100)
        }        

    # Arrays to store scores
    non_nested_estimators = []
    non_nested_scores = np.zeros(NUM_TRIALS)
    nested_scores = np.zeros(NUM_TRIALS)

    # Loop for each trial
    for i in range(NUM_TRIALS):
        print(i)
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
        inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        #clf = RandomizedSearchCV(estimator=svr, param_distributions=p_grid, cv=inner_cv, n_jobs=n_cpu, scoring="f1", verbose=5)
        clf = RandomizedSearchCV(estimator=pipeline, param_distributions=p_grid, cv=inner_cv, n_jobs=n_cpu, scoring="f1",verbose=verbosity)
        print("Made it through innercv", i)
        clf.fit(X, y)
        print("Fitted the classifiers")
        non_nested_scores[i] = clf.best_score_
        non_nested_estimators.append(clf)

        print(clf.best_params_)
        print(clf.best_score_)

        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring='f1',verbose=verbosity)
        #nested_score_svm = cross_val_score(clf_svm, X=X, y=y, cv=outer_cv, scoring='roc_auc')
        print("Made it through outercv", i)
        nested_scores[i] = nested_score.mean()
        #nested_scores_svm[i] = nested_score_svm.mean()
    

    print(non_nested_scores)
    print(nested_scores)
    #print("")
    #print(non_nested_scores)
    #print(nested_scores_svm)
    #score_difference = non_nested_scores - nested_scores
    '''test_train_data = np.random.permutation(test_train_data)


    #print(y)
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
    print(fscores, avg_fscore)'''
    #print(accuracy)

    #with open('waveletcoeffs.json', 'w') as out:
    #    json.dump(wavelet_coeffs, out)
    return non_nested_estimators


if __name__ == "__main__":
    hp = {'num_band_coeffs': 10, 'wavelet_type': 'sym2'}
    sys.exit(classify_supernovae(hp))
