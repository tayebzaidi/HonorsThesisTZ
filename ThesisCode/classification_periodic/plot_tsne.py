#!/usr/bin/env python
import sys
import os
import json
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
# sys.path.insert(1,'~/Multicore-TSNE')
# from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def main():
    metric = "correlation" #"euclidean", "mahalanobis"
    # the best perplexity seems to be ~300 - it's essentially the number of neighbors you think share a label

    best_learning_rate = 100
    maxiter = 2000

    coeffs = []
    labels = []
    keys   = []

    # supply filename as argument 1
    fn = sys.argv[1]

    # there's just too little data in this dataset to warrant all those groups
    #groups = { 0:(u'II', u'II P', u'IIb', u'IIn'),
    #           1:(u'Ia',),
    #           2:( u'Ia Pec', u'Ia-02cx'),
    #           3:(u'Ib', u'Ib Pec', u'Ib-n/IIb-n', u'Ib/c',\
    #             u'Ibn', u'Ic', u'Ic BL', u'Ic Pec', u'Ic/Ic-bl'),
    #           4:(u'SLSN', u'SLSN-I', u'SLSN-II')}

    groups = { 0:(u'II', u'II P', u'IIb', u'IIn',\
                  u'Ib', u'Ib Pec', u'Ib-n/IIb-n', u'Ib/c',\
                  u'Ibn', u'Ic', u'Ic BL', u'Ic Pec', u'Ic/Ic-bl',\
                  u'SLSN', u'SLSN-I', u'SLSN-II'),
               1:(u'Ia',)}
               #2:( u'Ia Pec', u'Ia-02cx'),
               #3:(u'Ib', u'Ib Pec', u'Ib-n/IIb-n', u'Ib/c',\
               #  u'Ibn', u'Ic', u'Ic BL', u'Ic Pec', u'Ic/Ic-bl'),
               #4:(u'SLSN', u'SLSN-I', u'SLSN-II')}

    # parse the data
    with open(fn, 'r') as f:
        d = json.load(f)
        for key in d.keys():
            keys.append(key)
            coeffs.append(d[key]['coeffs'])
            type  = d[key]['type']
            for group, types in groups.items():
                if type in types:
                    labels.append(group)
                    continue
    coeffs = np.array(coeffs)
    labels = np.array(labels)
    keys   = np.array(keys)
    colors = ['C{:n}'.format(x) for x in labels]
    ucolors = ['C{:n}'.format(x) for x in sorted(np.unique(labels))]
    #stringlabels = ['II', 'Ia', 'Iax', 'Ib/c', 'SLSN']
    stringlabels = ['Ia', 'Non-Ia']

    # make the TSNE
    X_scaled = preprocessing.scale(coeffs)
    model = TSNE(n_components=2, random_state=0, perplexity=float(sys.argv[2]),\
            n_iter=maxiter, verbose=2, learning_rate=100, init="pca", metric=metric)

    # there's an alternate package from github, but doesn't matter for this dataset since it is small
    #model = TSNE(n_jobs=8, n_iter=max_iter, method='exact', perplexity=float(sys.argv[2]))

    # find the transformation to a 2D space
    X_scaled = preprocessing.scale(coeffs)
    out = model.fit_transform(X_scaled)

    ###Print outliers and check manually
    #x first
    print(out.shape)
    bad_idxs_x = mad_based_outlier(out[:,0])
    bad_idxs_y = mad_based_outlier(out[:,1])
    print(keys[bad_idxs_x])
    print(keys[bad_idxs_y])
    bad_idxs = np.bitwise_or(bad_idxs_x, bad_idxs_y)
    out = np.delete(out, np.where(bad_idxs==True), 0)
    labels = elim_idxs(labels, bad_idxs_x, bad_idxs_y)
    keys = elim_idxs(keys, bad_idxs_x, bad_idxs_y)
    colors = elim_idxs(colors, bad_idxs_x, bad_idxs_y)
    ucolors = elim_idxs(ucolors, bad_idxs_x, bad_idxs_y)
    

    # plot the results
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    ax.scatter(out[:,0], out[:,1], c=colors, alpha=0.7)

    # add a legend
    lines = []
    for col, type in zip(ucolors, stringlabels):
        lines.append(mlines.Line2D([], [], color=col, marker='o', ls='None', ms=10, label=type))
    ax.legend(handles=lines, frameon=False, fontsize='large')

    # label axes
    ax.set_xlabel('TSNE u', fontsize='x-large')
    ax.set_ylabel('TSNE v', fontsize='x-large')
    ax.set_title(sys.argv[3], fontsize='xx-large')
    #ax.set_ylim(-5,5)
    #ax.set_xlim(-5,5)
    plt.tight_layout()

    # save plot
    outf = os.path.basename(fn)
    fig.savefig('tsne_{}.pdf'.format(outf.replace('.json','')))

    plt.ion()
    plt.show(fig)
    plt.close(fig)

def elim_idxs(obj, idxs_x, idxs_y):
    return np.delete(np.delete(obj, idxs_x), idxs_y)

###Code copied from stack overflow 22354094 for MAD outlier detection
def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

if __name__=='__main__':
    sys.exit(main())
