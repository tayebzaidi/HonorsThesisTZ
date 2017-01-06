#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import os
import sys
import numpy as np

def main():
    destination_file = 'subsampleLightcurves'

    filename = 'selectedLightcurves'
    with open(filename, 'r') as f:
        lightcurves = [line.rstrip('\n') for line in f]

    with PdfPages('multipage_pdf.pdf') as pdf:

        for lightcurve in lightcurves:
            lightcurve_path = '../gp_smoothed/' + lightcurve
            with open(lightcurve_path, 'r') as f:
                file_data = json.load(f)

            #This hack removes the '_gpsmoothed.json' from the string to return the objname
            objname = lightcurve[:-16]

            #Number of filters
            N = len(file_data.keys())
            if N < 3:
                cols = 1
            rows = int(math.ceil(N / cols))

            gs = gridspec.GridSpec(rows, cols)

            plt.figure(figsize=(6, 6))
            plt.title(objname)

            #Return the list of keys from the file_data
            data = list(file_data)

            for i in range(len(data)):
                filt = data[i]
                mjd = file_data[filt][mjd]
                mag = file_data[filt][mag]
                mag_err = file_data[filt][dmag]
                model_phase = file_data[filt][modeldate]
                model_mag = file_data[filt][modelmag]
                bspline_mag = file_data[filt][bsplinemag]

                ax = fig.add_subplot(gs[i])
                ax.errorbar(mjd, mag, fmt='r', yerr=mag_err,label='Original Data')
                ymin, ymax = ax.get_ylim()
                ax.plot(model_phase, model_mag, '-k', label='GP Smoothed Data')
                ax.plot(model_phase, bspline_mag, '-g', label='Spline Smoothed Data')
                ax.ylim(ymin, ymax)


                

            pdf.savefig()  # saves the current figure into a pdf page

            


if __name__ == "__main__":
    sys.exit(main())



