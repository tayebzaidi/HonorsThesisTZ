"""A module for extracting the wavelet features from a given lightcurve"""
#!/usr/bin/env python
import sys
import json
import numpy as np
import subprocess

#Initialize R script
COMMAND = 'Rscript'
SCRIPT_PATH = './getBagidisCoeffs.R'



def get_bagidis_coeffs(mjd, magnitude, num_coeffs=10, script_path=SCRIPT_PATH):
    """  Returns the first numCoeffs BAGIDIS coeffs of a given lightcurve
    mjd -- Modified Julian Date (must be a uniform grid)
    magnitude -- Corresponding magnitudes to the dates
    numCoeffs -- number of coefficients to be returned, in decreasing order of importance
    scriptPath -- path to the R script, autoset to be in the current directory
    """
    arguments = list(map(str,list(magnitude) + [num_coeffs]))
    cmd = [COMMAND, script_path] + arguments
    bagidis_coeffs = subprocess.check_output(cmd, universal_newlines=True)

    return bagidis_coeffs

def general_wavelet_coeffs(wavelet_type, mjd, magnitude, num_coeffs=10, script_path=SCRIPT_PATH):
    """  Returns the wavelet decomposition for given types of wavelets
    wavelet_type -- 'bagidis' (Returns num_coeffs of BAGIDIS coefficients)
                    'haar' (Wavelet coefficients of the Haar wavelet decomposition)
    mjd -- phase in Modified Julian date (must be a uniform grid)
    magnitude -- magnitudes corresponding to the given phases
    """
    if wavelet_type == 'bagidis':
        return get_bagidis_coeffs(mjd, magnitude, num_coeffs, script_path)


def main():
    pass

if __name__ == "__main__":
    sys.exit(main())


