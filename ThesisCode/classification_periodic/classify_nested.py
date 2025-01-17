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
from sklearn.metrics import confusion_matrix
from scipy.stats import randint
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from feature_store import label_class
from collections import Counter

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
    print("Beginning optimized classification")
    #Get hyperparameter values
    num_band_coeffs = hyperparams['num_band_coeffs']

    #Load the wavelet coefficients from the specified file
    with open(input_file, 'r') as f:
        wavelet_coeffs = json.load(f)
 
    wavelet_type = hyperparams['wavelet_type']
    example_entry = wavelet_coeffs[list(wavelet_coeffs)[0]]
    num_coeffs = len(example_entry['coeffs'])

    num_classes = 2

    test_train_data = label_class(wavelet_coeffs, num_coeffs)

    #Split the data into test and train data
    #First randomize the object order to eliminate bias towards object_type
    # and set the seed value to ensure repeatability
    #np.random.seed(40)
    test_train_data = np.random.permutation(test_train_data)

    X = test_train_data[:,0:num_coeffs]
    y = np.ravel(test_train_data[:,num_coeffs])
    print(y)

    ## Reduce training set size drastically for initial testing
    #X = X[0:500,:]
    #y = y[0:500]

    #print(X[0:10,:])

    #Rescale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #X = scaler.transform(X)

    #print(X[0:10,:])

    #Model the nested cross validation off of the scikit learn example
    NUM_TRIALS = 5
    verbosity=0

    # Set up possible values of parameters to optimize over
    num_feature_options_n = range(40, 130, 4)
    num_feature_options_k = range(2, 20, 1)
    num_estimators_range = range(200,201,100)

    #TESTING to see if Pipeline functions
    #num_feature_options = range(10,20,2)
    #num_estimators_range = range(300,600,100)

    if wavelet_type == 'bagidis':
        pipeline = Pipeline([
                        ('reduce_dim', SelectKBest(customScore)),
                        ('classify', RandomForestClassifier(oob_score=True))
        ])
        p_grid = {
                    'reduce_dim__k': num_feature_options_k,
                    'classify__n_estimators': num_estimators_range
        }
    else:
        #Do PCA for non-bagidis wavelets
        pipeline = Pipeline([
                        ('reduce_dim', PCA()),
                        ('classify', RandomForestClassifier(oob_score=True))
        ])
        # Set up possible values of parameters to optimize over
       
        p_grid = {
                    'reduce_dim__n_components': num_feature_options_n,
                    'classify__n_estimators': num_estimators_range
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
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)
        #print(inner_cv, outer_cv)
        #sys.exit()

        # Non_nested parameter search and scoring
        #clf = RandomizedSearchCV(estimator=svr, param_distributions=p_grid, cv=inner_cv, n_jobs=n_cpu, scoring="f1", verbose=5)
        clf = RandomizedSearchCV(estimator=pipeline, param_distributions=p_grid, cv=inner_cv, n_jobs=n_cpu, scoring="roc_auc",verbose=verbosity)
        #print("Made it through innercv", i)
        clf.fit(X, y)
        print("Fitted the classifiers")
        non_nested_scores[i] = clf.best_score_
        non_nested_estimators.append(clf)

        print(clf.best_params_)
        print(clf.best_score_)
        #print(dir(clf))
        #print(type(clf.best_estimator_.steps[1][1].oob_decision_function_))
        y_pred_prob = clf.best_estimator_.steps[1][1].oob_decision_function_
        y_pred = [np.argmax(probs) for probs in y_pred_prob]
        #print(y_pred)
        conf_mat = confusion_matrix(y, y_pred)
        print(conf_mat)

        

        # Nested CV with parameter optimization
        #nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, scoring='roc_auc',verbose=verbosity)
        nested_score = np.zeros(len(nested_scores))
        #nested_score_svm = cross_val_score(clf_svm, X=X, y=y, cv=outer_cv, scoring='roc_auc')
        #print("Made it through outercv", i)
        nested_scores[i] = nested_score.mean()
        print("Nested Score: ", nested_score.mean())
        #nested_scores_svm[i] = nested_score_svm.mean()
    

    print(non_nested_scores)
    print(nested_scores)
    score_difference = non_nested_scores - nested_scores

    print("Average difference of {0:6f} with std. dev. of {1:6f}."
        .format(score_difference.mean(), score_difference.std()))
    
    #scores = [non_nested_scores, non_nested_scores.std()]
    scores = [nested_scores, nested_scores.std()]
    original_data = [X, y]

    #with open('waveletcoeffs.json', 'w') as out:
    #    json.dump(wavelet_coeffs, out)
    return scores, non_nested_estimators, original_data

def customScore(X, y):
    """
    A dumb function to score the first vectors in an array the highest (descending order)
    to simulate feature selection taking only the first k components
    Uses the monotonically decreasing negative exponential to ensure consistency
    """
    num_features = X.shape[1]
    features_per_band = 50
    F = [40*np.exp(-i/20) for i in range(num_features)]
    F = np.array(F)
    pval = np.repeat(np.array([0.01]), num_features)
    return (F, pval)


if __name__ == "__main__":
    hp = {'num_band_coeffs': 10, 'wavelet_type': 'sym2'}
    sys.exit(classify_supernovae(hp))
