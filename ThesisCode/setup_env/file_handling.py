#!/usr/bin/env python
import sys
import json
from mpi4py import MPI

def loadfiles():
    """ Load limited set of gaussian process smoothed lightcurves for analysis
    No parameters as of yet

    Returns a dictionary with each object as an entry
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    procs_num = comm.Get_size()
    print(procs_num, rank)
    json_files = glob.glob('./gp_smoothed/*.json')
    nfiles = len(json_files)
    quotient = nfiles/procs_num + 1
    P = rank*quotient
    Q = (rank+1)*quotient
    if P > nfiles:
        P = nfiles
    if Q > nfiles:
        Q = nfiles
    print(procs_num, rank, nfiles, quotient, P, Q)

    obj_data = {}

    for f in json_files[P:Q]:
        data = json.load(j)
        objname = data['objname']
        obj_data[objname] = data

    return obj_data

def main():
    pass

if __name__ == "__main__":
    sys.exit(main())
