#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
import sys
import numpy as np
import math

def main():
    path = "./gp_smoothed_periodic/"
    output_lightcurves_file = 'selectedLightcurves'
    output_lightcurves = []

    filenames = os.listdir(path)

    #Randomize the file order to allow for fairer selection of the sub-sample
    filenames = np.random.permutation(filenames)

    for filename in filenames:
        objname = filename
        with open(os.path.join(path, filename), mode='r') as f:
            file_data = json.load(f)

        #Ignore all non-CSP or CfA entries
        #for k in list(file_data.keys()):
        #    if not (k.endswith('CSP') or ('CfA' in k)):
        #        del file_data[k]
        if len(file_data) == 0:
            continue

        N = len(file_data)
        if N < 3:
            cols = 1
        else:
            cols = 3
        rows = int(math.ceil(N / cols))

        gs = gridspec.GridSpec(rows, cols)

        fig = plt.figure(figsize=(10, 12))
        fig.suptitle(objname)



        for i, filt in enumerate(file_data.keys()):

            mjd = file_data[filt]['mjd']
            mag = file_data[filt]['mag']
            mag_err = file_data[filt]['dmag']
            model_phase = file_data[filt]['modeldate']
            model_mag = file_data[filt]['modelmag']
            #bspline_mag = file_data[filt]['bsplinemag']
            #modelmag_sub = file_data[filt]['modelmag_sub']
            type = file_data[filt]['type']

            ax = fig.add_subplot(gs[i])
            ax.errorbar(mjd, mag, fmt='r', yerr=mag_err,label='Original Data', alpha=0.7)
            ymin, ymax = ax.get_ylim()
            ax.plot(model_phase, model_mag, '-k', label='GP Smoothed Data')
            #ax.plot(model_phase, bspline_mag, '-g', label='Spline Smoothed Data')
            #ax.plot(model_phase, modelmag_sub, '-k', label='GP/Bspline subtracted', linewidth=1.5)
            ax.set_ylim(ymin, ymax)


            #Print outlier stats
            mag_range = np.ptp(model_mag)
            old_mag_range = np.ptp(mag)
            print(objname, filt)

        plt.draw()
        plt.pause(0.05)
        print("Number of files currently: ", len(output_lightcurves))
        print("Supernova Type: ", type)
        keystroke = input("<Hit Enter To Close>")
        #keystroke = '.'
        if keystroke == '.':
            output_lightcurves.append(objname)
        elif keystroke == 'q':
            with open(output_lightcurves_file, 'w') as out:
                for objname in output_lightcurves:
                    out.write(objname + '\n')
            plt.close()
            sys.exit()
        plt.close()
    with open(output_lightcurves_file, 'w') as out:
            for objname in output_lightcurves:
                out.write(objname + '\n')

        

if __name__=="__main__":
    sys.exit(main())
