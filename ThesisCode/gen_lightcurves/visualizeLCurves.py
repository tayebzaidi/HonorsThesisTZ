#!/usr/bin/env python
import matplotlib.pyplot as plt
import json
import os
import sys
import numpy as np
import random

def main():
    path = "./gp_smoothed/"
    output_lightcurves_file = 'selectedLightcurves'
    output_lightcurves = []

    filenames = os.listdir(path)

    #Randomize the file order to allow for fairer selection of the sub-sample
    filenames = np.random.permutation(filenames)

    for filename in filenames:
        objname = filename
        with open(os.path.join(path, filename), mode='r') as f:
            data_raw = json.load(f)

        for filt in data_raw:
            time = data_raw[filt]["modeldate"]
            mag = data_raw[filt]["modelmag"]
            mag_err = data_raw[filt]["modelerr"]
            old_time = data_raw[filt]["mjd"]
            old_mag = data_raw[filt]["mag"]
            old_mag_err = data_raw[filt]["dmag"]
            bspline_mag = data_raw[filt]["bsplinemag"]
            #print("Resampled Time", time)
            #print("Old time", old_time)
            #print("Old mag", old_mag)

            #Print outlier stats
            mag_range = np.ptp(mag)
            old_mag_range = np.ptp(old_mag)
            print(objname, filt)

            #out_data = map(np.array,[time, mag, old_time, old_mag])

            fig = plt.figure(figsize=(10, 10))
            ax0 = fig.add_subplot(1, 1, 1)
            ax0.errorbar(old_time, old_mag, fmt='r', yerr=old_mag_err,label='Original Data')
            ymin, ymax = ax0.get_ylim()
            ax0.plot(time, mag,'-k', label='Smoothed Data')
            ax0.plot(time, bspline_mag, '-g', label='Spline Smoothed Data')
            ax0.set_ylim(ymin, ymax)
            plt.draw()
            plt.pause(0.05) 
            input("<Hit Enter To Close>")
            plt.close()

        

if __name__=="__main__":
    sys.exit(main())
