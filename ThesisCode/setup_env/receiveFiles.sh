#!/bin/bash
HOST='Antares03'
FILE_PATH='/home/antares/nfs_share/tzaidi/HonorsThesisTZ/ThesisCode/gen_lightcurves/gp_smoothed/all_lcurves.tar.gz'

rm ../gen_lightcurves/gp_smoothed/*
if ssh $HOST stat $FILE_PATH \> /dev/null 2\>\&1
    then
        scp Antares03:$FILE_PATH ../gen_lightcurves/gp_smoothed/
    else
        ssh Antares03 'cd ~/nfs_share/tzaidi/HonorsThesisTZ/ThesisCode/gen_lightcurves/gp_smoothed && tar -czvf all_lcurves.tar.gz ./*'
        scp Antares03:$FILE_PATH ../gen_lightcurves/gp_smoothed/
fi

tar -xzvf ../gen_lightcurves/gp_smoothed/all_lcurves.tar.gz
rm ../gen_lightcurves/gp_smoothed/all_lcurves.tar.gz
