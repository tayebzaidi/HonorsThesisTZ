"""A module for extracting the wavelet features from a given lightcurve"""
#!/usr/bin/env python
import sys
import json
import numpy as np
import rpy2
from rpy2.robjects.packages import importr

def get_bagidis_coeffs(mjd, magnitude, num_coeffs=10):
    """  Returns the first numCoeffs BAGIDIS coeffs of a given lightcurve
    mjd -- Modified Julian Date (must be a uniform grid)
    magnitude -- Corresponding magnitudes to the dates
    numCoeffs -- number of coefficients to be returned, in decreasing order of importance
    scriptPath -- path to the R script, autoset to be in the current directory
    """
    bagidis = importr('Bagidis')
    magnitude = rpy2.robjects.FloatVector(magnitude)
    bagidis_lcurve = bagidis.BUUHWE(magnitude)
    bagidis_coeffs = bagidis_lcurve.rx2('detail')[0:num_coeffs]
    bagidis_coeffs = np.array(bagidis_coeffs)

    return bagidis_coeffs

def general_wavelet_coeffs(wavelet_type, mjd, magnitude, num_coeffs=10):
    """  Returns the wavelet decomposition for given types of wavelets
    wavelet_type -- 'bagidis' (Returns num_coeffs of BAGIDIS coefficients)
                    'haar' (Wavelet coefficients of the Haar wavelet decomposition)
    mjd -- phase in Modified Julian date (must be a uniform grid)
    magnitude -- magnitudes corresponding to the given phases
    """
    if wavelet_type == 'bagidis':
        return get_bagidis_coeffs(mjd, magnitude, num_coeffs)


def main():
    pass

if __name__ == "__main__":
    sys.exit(main())


