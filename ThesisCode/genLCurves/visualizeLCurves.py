#!/usr/bin/env python2
import matplotlib.pyplot as plt
import json
import os
import sys
import numpy as np

def main():
    path = "./gp_smoothed/"

    filenames = os.listdir(path)
    for filename in filenames:
        objname = filename
        with open(os.path.join(path, filename), mode='r') as f:
            data_raw = json.load(f)

        for key in data_raw:
            time = data_raw[key]["smoothed"][0]
            mag = data_raw[key]["smoothed"][1]
            mag_err = data_raw[key]["smoothed"][2]
            old_time = data_raw[key]["original"][0]
            old_mag = data_raw[key]["original"][1]
            old_mag_err = data_raw[key]["original"][2]
            print("Resampled Time", time)
            print("Old time", old_time)
            print("Old mag", old_mag)

            #Print outlier stats
            mag_range = np.ptp(mag)
            old_mag_range = np.ptp(old_mag)
            print("Ranges", mag_range, old_mag_range)

            #out_data = map(np.array,[time, mag, old_time, old_mag])

            fig = plt.figure(figsize=(10, 10))
            ax0 = fig.add_subplot(1, 1, 1)
            ax0.errorbar(time, mag, yerr=mag_err, fmt='-k', label='Smoothed Data')
            ax0.errorbar(old_time, old_mag, fmt='r', yerr=old_mag_err,label='Original Data')
            plt.show()

        

if __name__=="__main__":
    sys.exit(main())
