#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import json
import os
import sys
import numpy as np
import math

def main():
    destination_file = 'subsampleLightcurves.pdf'

    filename = 'goodLightcurves'
    with open(filename, 'r') as f:
        lightcurves = [line.rstrip('\n') for line in f]
        lightcurves

    with PdfPages(destination_file) as pdf:

        for lightcurve in lightcurves:
            lightcurve_path = '../gen_lightcurves/gp_smoothed/' + lightcurve
            with open(lightcurve_path, 'r') as f:
                file_data = json.load(f)

            #This hack removes the '_gpsmoothed.json' from the string to return the objname
            objname = lightcurve[:-16]

            #Number of filters
            N = len(file_data.keys())
            print(N)
            cols = 3
            if N < 3:
                cols = 1
            rows = int(math.ceil(N / cols))

            #To ensure that plot text fits without overlay
            #Change font size to fit the text, taken from http://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot answer by Pedro M. Duarte
            SIZE = 5
            MEDIUM_SIZE = 8
            BIGGER_SIZE = 10

            plt.rc('font', size=SIZE)                # controls default text sizes
            plt.rc('axes', titlesize=SIZE)           # fontsize of the axes title
            plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SIZE)          # fontsize of the tick labels
            plt.rc('ytick', labelsize=SIZE)          # fontsize of the tick labels
            plt.rc('legend', fontsize=SIZE)          # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

            if rows > 3:
                small_size = 2
                plt.rc('font', size=small_size)                # controls default text sizes
                plt.rc('axes', titlesize=small_size)           # fontsize of the axes title
                plt.rc('axes', labelsize=small_size)    # fontsize of the x and y labels
                plt.rc('xtick', labelsize=small_size)          # fontsize of the tick labels
                plt.rc('ytick', labelsize=small_size)

            gs = gridspec.GridSpec(rows, cols)

            fig = plt.figure(figsize=(6, 6))
            fig.suptitle(objname)

            #Return the list of keys from the file_data
            data = list(file_data)

            for i in range(len(data)):
                filt = data[i]
                mjd = file_data[filt]['mjd']
                mag = file_data[filt]['mag']
                mag_err = file_data[filt]['dmag']
                model_phase = file_data[filt]['modeldate']
                model_mag = file_data[filt]['modelmag']
                bspline_mag = file_data[filt]['bsplinemag']
                goodstatus = file_data[filt]['goodstatus']

                ax = fig.add_subplot(gs[i])
                ax.errorbar(mjd, mag, fmt='r', yerr=mag_err,label='Original')
                ymin, ymax = ax.get_ylim()
                ax.plot(model_phase, model_mag, '-k', label='GP')
                ax.plot(model_phase, bspline_mag, '-g', label='BSpline')
                ax.set_title(filt)
                handles, labels = ax.get_legend_handles_labels()
                if(not goodstatus):
                    ax.set_ylim(ymin, ymax)

            fig.legend(handles, labels)
            pdf.savefig()  # saves the current figure into a pdf page
    

            


if __name__ == "__main__":
    sys.exit(main())



