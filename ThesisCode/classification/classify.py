"""A script to run wavelet decomposition on arbitrary objects"""
#!/usr/bin/env python
import sys
import json
import featureExtraction
import matplotlib.pyplot as plt
from matplotlib import cm

def main():

    filename = '../gen_lightcurves/CfA_CSP_lightcurves'
    with open(filename, 'r') as f:
        lightcurves = [line.rstrip('\n') for line in f]

    wavelet_coeffs = {}
    i = 1
    for lightcurve in lightcurves:
        i += 1
        lightcurve_path = '../gen_lightcurves/gp_smoothed/' + lightcurve
        with open(lightcurve_path, 'r') as f:
            file_data = json.load(f)

        #This hack removes the '_gpsmoothed.json' from the string to return the objname
        objname = lightcurve[:-16]
        print(objname)


        ## For now, take only filters starting with B_ (Assuming only one filter with B_)
        for k in list(file_data.keys()):
            if not k.startswith('B_'):
                del file_data[k]
        if len(file_data) == 0:
            continue
        wavelet_coeffs[objname] = {}

        filt = list(file_data.keys())[0]
        mjd = file_data[filt]['mjd']
        mag = file_data[filt]['mag']
        mag_err = file_data[filt]['dmag']
        model_phase = file_data[filt]['modeldate']
        model_mag = file_data[filt]['modelmag']
        bspline_mag = file_data[filt]['bsplinemag']
        goodstatus = file_data[filt]['goodstatus']
        stype = file_data[filt]['type']

        raw_coeffs = featureExtraction.general_wavelet_coeffs('bagidis', model_phase, bspline_mag)
        wavelet_coeffs[objname]['coeffs'] = list(map(float, raw_coeffs.split()))
        wavelet_coeffs[objname]['type'] = stype
        print(i)
        #if i > 8:
        #    break

    print(wavelet_coeffs)
    fig = plt.figure(figsize=(10, 10))
    feature1 = []
    feature2 = []
    feature3 = []
    feature4 = []
    type = []
    print(list(wavelet_coeffs.keys()))
    for obj in wavelet_coeffs:
        feature1.append(wavelet_coeffs[obj]['coeffs'][0])
        feature2.append(wavelet_coeffs[obj]['coeffs'][1])
        feature3.append(wavelet_coeffs[obj]['coeffs'][2])
        feature4.append(wavelet_coeffs[obj]['coeffs'][3])
        type.append(wavelet_coeffs[obj]['type'])
        print(wavelet_coeffs[obj]['type'])

    type = list(map(hash, type))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    ax1.scatter(feature1, feature2, c=type, cmap=cm.jet, marker='o')
    ax1.set_title('1 vs 2')
    ax2.scatter(feature2, feature3, c=type, cmap=cm.jet, marker='o')
    ax2.set_title('2 vs 3')
    ax3.scatter(feature1, feature3, c=type, cmap=cm.jet, marker='o')
    ax3.set_title('1 vs 3')
    ax4.scatter(feature2, feature4, c=type, cmap=cm.jet, marker='o')
    ax4.set_title('2 vs 4')
    plt.show()

    with open('waveletcoeffs.json', 'w') as out:
        json.dump(wavelet_coeffs, out)


if __name__ == "__main__":
    sys.exit(main())
