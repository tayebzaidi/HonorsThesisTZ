"""A module for extracting the wavelet features from a given lightcurve"""
#!/usr/bin/env python
import sys
import json
import numpy as np
import subprocess

#Initialize R script
command = 'Rscript'
bagidis_path = './getBagidisCoeffs.R'



def get_bagidis_coeffs(mjd, magnitude, num_coeffs=10, script_path=bagidis_path):
    """
    Returns the first numCoeffs BAGIDIS coeffs of a given lightcurve
    mjd -- Modified Julian Date (must be a uniform grid)
    magnitude -- Corresponding magnitudes to the dates
    numCoeffs -- number of coefficients to be returned, in decreasing order of importance
    scriptPath -- path to the R script, autoset to be in the current directory
    """
    arguments = [mjd, magnitude, num_coeffs]
    cmd = [command, script_path] + arguments
    bagidis_coeffs = subprocess.check_output(cmd, universal_newlines=True)

    return bagidis_coeffs

def generalWaveletCoeffs(waveletType, mjd, magnitude):
    pass


def main():
    return 1

if __name__ == "__main__":
    sys.exit(main())


