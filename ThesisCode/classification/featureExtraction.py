"""A module for extracting the wavelet features from a given lightcurve"""
#!/usr/bin/env python
import sys
import json
import numpy as np
import rpy2
import pywt
from rpy2.robjects.packages import importr
from sklearn.decomposition import PCA

def get_bagidis_coeffs(lightcurve, num_band_coeffs=10):
    """  Returns the first numCoeffs BAGIDIS coeffs of a given lightcurve
    lightcurve -- lightcurve dictionary containing all bands for single object
    num_band_coeffs -- number of coefficients to be returned, in decreasing order of importance

    Returns: 
        all_coeffs -- all bagidis coeffs (num_band_coeffs * num_bands) in np array
    """
    num_bands = len(list(lightcurve.keys()))
    all_coeffs = np.zeros((num_band_coeffs*num_bands,))
    lightcurve_keys = list(lightcurve.keys())
    #Make sure the bands are in order in the coefficients
    lightcurve_keys = sorted(lightcurve_keys)

    for i, filt in enumerate(lightcurve_keys):
        magnitude = lightcurve[filt]['modelmag']

        bagidis = importr('Bagidis')
        magnitude = rpy2.robjects.FloatVector(magnitude)
        bagidis_lcurve = bagidis.BUUHWE(magnitude)
        bagidis_coeffs = bagidis_lcurve.rx2('detail')[1:num_band_coeffs+1]
        bagidis_coeffs = np.array(bagidis_coeffs)

        all_coeffs[i::num_bands] = bagidis_coeffs
        #print(bagidis_coeffs.shape)
        #print(all_coeffs)

    return all_coeffs

def get_pywt_wav_coeffs(lightcurve, wav_type, num_levels, num_band_coeffs=10):
    """
    This particular function draws from Michelle Lochner's code in https://github.com/LSSTDESC/snmachine
    """

    num_points = len(lightcurve[list(lightcurve.keys())[0]]['modelmag'])
    num_bands = len(list(lightcurve.keys()))
    num_band_coeffs = num_points*num_levels
    all_coeffs = np.zeros((num_points*num_levels*num_bands,))

    lightcurve_keys = list(lightcurve.keys())
    #Make sure the bands are in order in the coefficients
    lightcurve_keys = sorted(lightcurve_keys)

    for i, filt in enumerate(lightcurve_keys):
        magnitude = lightcurve[filt]['modelmag']

        sc_detail_coeffs = pywt.swt(magnitude, wav_type, level=num_levels)
        #Extract only the detail coefficients from the the levels of wavelet_analysis
        detail_coeffs = np.zeros(len(magnitude)*num_levels)
        for j in range(num_levels):
            detail_coeffs[j*num_points:(j+1)*num_points] = sc_detail_coeffs[j][1]
        
        all_coeffs[i*num_band_coeffs:(i+1)*num_band_coeffs] = detail_coeffs

    return all_coeffs

def general_wavelet_coeffs(lightcurve, wavelet_type, num_levels=1, num_band_coeffs=10):
    """  Returns the wavelet decomposition for given types of wavelets
    wavelet_type -- 'bagidis' (Returns num_band_coeffs of BAGIDIS coefficients)
                    'haar' (Wavelet coefficients of the Haar wavelet decomposition)
    mjd -- phase in Modified Julian date (must be a uniform grid)
    magnitude -- magnitudes corresponding to the given phases
    """
    wavelet_types = pywt.wavelist()
    if wavelet_type == 'bagidis':
        return get_bagidis_coeffs(lightcurve, num_band_coeffs)
    elif wavelet_type in wavelet_types:
        return get_pywt_wav_coeffs(lightcurve, wavelet_type, num_levels, num_band_coeffs)
    else:
        pass


def main():
    pass

if __name__ == "__main__":
    sys.exit(main())


