"""A script to run wavelet decomposition on arbitrary objects"""
#!/usr/bin/env python
import sys
import json
import featureExtraction

def main():

    filename = '../gen_lightcurves/CfA_CSP_lightcurves'
    with open(filename, 'r') as f:
        lightcurves = [line.rstrip('\n') for line in f]

    for lightcurve in lightcurves:
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

        filt = list(file_data.keys())[0]
        mjd = file_data[filt]['mjd']
        mag = file_data[filt]['mag']
        mag_err = file_data[filt]['dmag']
        model_phase = file_data[filt]['modeldate']
        model_mag = file_data[filt]['modelmag']
        bspline_mag = file_data[filt]['bsplinemag']
        goodstatus = file_data[filt]['goodstatus']
        type = file_data[filt]['type']

        wavelet_coeffs = featureExtraction.general_wavelet_coeffs('bagidis', model_phase, bspline_mag)
        print(wavelet_coeffs)

if __name__ == "__main__":
    sys.exit(main())