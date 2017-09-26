import os, sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score
from numpy import mean
import math

def main():
    """
    Iterate over the estimators in the directory and extract the scores
    and plot ROC curves
    """
    all_files = os.listdir()
    estimator_files = [file for file in all_files if file.endswith('estimators.p')]
    estimator_files = sorted(estimator_files)
    base_path = '../classification/'

    fig1 = plt.figure(figsize=(9,6))
    ax1 = fig1.add_subplot(111)

    for dump_file in estimator_files:
        fig2 = plt.figure(figsize=(12,6))
        ax2 = fig2.add_subplot(111)
        with open(base_path+dump_file, 'rb') as f:
            dump_data = pickle.load(f)
        estimators = dump_data[0]
        y = dump_data[1][1]
        cross_val_auc = dump_data[2][0].mean().round(decimals=3)
        cross_val_auc_std = np.array(dump_data[2][1]).round(decimals=3)
        feature_importances = np.zeros((len(estimators), len(estimators[0].best_estimator_.steps[1][1].feature_importances_)))
        
        current_wd = os.getcwd()
        #Assume things about the directory structure
        if current_wd.split(os.sep)[-2] == "DES_Pipeline":
            survey = 'DES'
        else:
            survey = 'OSC'
        wav_type = dump_file.split('_')[0]
        wav_level = dump_file.split('_')[1]
        
        accuracies = []
        f1_scores = []
        auc_vals = []
        max_auc_val = 0
        num_thresholds = len(roc_curve(y, estimators[0].best_estimator_.steps[1][1].oob_decision_function_[:,1])[0])
        num_estimators = len(estimators)
        num_components = []
        
        for i, estimator in enumerate(estimators):
            num_components.append(len(estimator.best_estimator_.steps[1][1].feature_importances_))
            #print("Number of Components: ", len(estimator.best_estimator_.steps[1][1].feature_importances_))
            #feature_importances[i,:] = estimator.best_estimator_.steps[1][1].feature_importances_
            #print(estimator.best_estimator_.steps)
            y_probas = estimator.best_estimator_.steps[1][1].oob_decision_function_
            y_pred = np.around(y_probas[:,1])
            
            accuracies.append(accuracy_score(y, y_pred))
            f1_scores.append(f1_score(y, y_pred))
            auc_vals.append(roc_auc_score(y, y_probas[:,1]))
            
            
            if auc_vals[-1] > max_auc_val:
                num_comp_feature_importances = len(estimator.best_estimator_.steps[1][1].feature_importances_)
                feature_importances = estimator.best_estimator_.steps[1][1].feature_importances_
                max_auc_val = auc_vals[-1]
                fpr, tpr, thresholds = roc_curve(y, y_probas[:,1])
            
        avg_accuracy = np.around(mean(accuracies), decimals=3)
        avg_f1 = np.around(mean(f1_scores), decimals=3)
        avg_auc_val = np.around(mean(auc_vals), decimals=3)

        print("{} --> CrossValAUC: {:.3} {:.3} -- Acc: {} -- F-Score: {} -- AUC: {}".format(dump_file, cross_val_auc, cross_val_auc_std, avg_accuracy, avg_f1, avg_auc_val))
        #print("Num Components", mean(num_components), np.std(num_components), num_components)
        #print("Accuracy: ", avg_accuracy)
        #print("F-Score: ", avg_f1)
        #print("AUC: ", avg_auc_val)
        label = "{} {} -- AUC: {:.3}".format(wav_type.upper(), wav_level, cross_val_auc)
        ax1.plot(fpr, tpr, label=label)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        filt_order = sorted(['g', 'r','i','z'])
        components = []
        for i in range(num_comp_feature_importances):
            filt = filt_order[i%len(filt_order)]
            coeff_idx = str(math.ceil(i/len(filt_order)))
            components.append(filt+'['+coeff_idx+']')
        #components = [filt_order[i%len(filt_order)]+'['+str(i+1)+']' for i in range(num_comp_feature_importances)]
        #print(components)
        ind = range(len(components))
        fig2_label = "Feature Importance for {} at {} Level{} of Analysis".format(wav_type.upper(), wav_level, (lambda x: ("s" if x > 1 else ""))(int(wav_level)))
        ax2.bar(ind, feature_importances, label=fig2_label)
        if dump_file.startswith('bagidis'):
            ax2.set_xticklabels(components)
            ax2.set_xticks(ind)
        ax2.set_xlabel('Feature', {'fontsize': 'larger'})
        ax2.set_ylabel('Relative Importance', {'fontsize': 'larger'})
        ax2.set_title(fig2_label, {'fontsize': 'larger'}, fontweight='bold')
        #ax2.legend()

        fig2_name = "_".join([survey, wav_type, wav_level])+'_feature_importances.png'
        fig2.savefig(fig2_name)
        plt.close(fig2)
    ax1.legend()
    fig1_name = "_".join([survey, 'estimators','ROC.png'])
    fig1.savefig(fig1_name)



        

if __name__=="__main__":
    sys.exit(main())