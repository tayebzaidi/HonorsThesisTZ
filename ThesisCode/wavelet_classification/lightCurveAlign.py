#!/usr/bin/env python
import file_handling
import sys
import os
import json
import numpy as np
from scipy.signal import find_peaks_cwt

def main():
    
    #open all of the files
    data = file_handling.loadfiles()

if __name__=="__main__":
    sys.exit(main())
