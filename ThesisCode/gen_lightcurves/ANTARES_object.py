#!/usr/bin/env python

import sys
import os
import argparse
import glob
import numpy as np
import scipy
import scipy.linalg
import scipy.interpolate as scinterp
import scipy.optimize as op
import astropy
import astropy.stats
from gatspy import periodic
import george
#import antares.stage_algorithms.ransac as ransac
import matplotlib.pyplot as plt
from george import kernels
import warnings
import multiprocessing


warnings.simplefilter('once')


def nn2_diffs(x):
    return np.subtract.outer(x,x)[np.tril_indices(x.shape[0],k=-1)]



class GaussianProcessFit:
    """
    This class implements a basic model for a Gaussian process regression for one a TouchstoneObject
    It's intended to be used with RANSAC to generate a fit that is robust to outliers
    CURRENTLY NOT FULLY TESTED! DO NOT USE UNTIL VALIDATED!
    """
    def __init__(self, input_column, output_column, error_column, per=False):
        """
        The RANSAC implementation we're calling simply slices the 'data' structure to get trial points
        This is nice, but it means all the data has to be bundled together in an (ndata, ncol) structure
        This means we save the columns that we want for input and such in the constructor
        and use it to pull the right columns out when we fit
        Additionally, we save if the kernel is periodic or not here
        """
        self.input_column  = input_column
        self.output_column = output_column
        self.error_column  = error_column

        if per:
            kernel = kernels.ExpSine2Kernel(0.5, 1.0)
        else:
            kernel = kernels.ExpSquaredKernel(1.0)
        self.kernel = kernel
        self.gp = george.GP(kernel)    

    def fit(self, data):
        phase     = np.vstack(data[:,self.input_column]).T[0]
        mag       = np.vstack(data[:,self.output_column]).T[0]
        mag_err   = np.vstack(data[:,self.error_column]).T[0]
        gp = self.gp
        gp.compute(phase, ((mag_err)**2.+0.01**2.)**0.5) 
        return (gp, phase, mag, mag_err)

    def get_error(self, data, model):
        gp , compphase, compmag, comperr = model
        phase     = np.vstack(data[:,self.input_column]).T[0] 
        mag       = np.vstack(data[:,self.output_column]).T[0]
        mag_err   = np.vstack(data[:,self.error_column]).T[0]
        interpy, cov = gp.predict(compmag, phase)
        err_per_point = (mag - interpy)**2.
        return err_per_point




###########################################################################################################################3




class TouchstoneObject:
    """
    The touchstone is a distillation of astrophysical knowledge, built from light curves of known objects 
    This class defines functions to process each of these objects
    """

    def __init__(self, objectname, time, mag, mag_err, passband, flux=False, per=False, best_period=None, header=None):

        self.objectname = objectname
        self.time    = np.array(time).astype('f')
        self.mag     = np.array(mag).astype('f')
        self.mag_err = np.array(mag_err).astype('f')
        self.passband= passband
        self.filters = np.unique(passband)
        self.per     = per
        self.best_period = None
        if self.per: #only set the period if we explicitly defined the object to be periodic
            self.best_period = best_period

        self.amplitude = None
        self.stats = None #order is min, max, mean, unbiased var, skewness (biased), fisher kurtosis (biased)

        # computed feature if we use splines
        self.outtck  = None
        # computed feature if we use gaussian process
        self.outgp  = None

        # computed features (if per is True)
        self.model       = None
        self.periods     = None
        self.P_multi     = None

        # currently not used - the DES light curves are actually in flux, but we're converting to mags
        self.flux    = flux 

        # currently sort of used 
        self.header  = {}
        


    @classmethod
    def fromfile(cls, filename, flux=False):
        if os.path.basename(filename).startswith('DES_'):
            return cls.__read_DES_file(filename)
        elif 'Magdumps' in filename:
            return cls.__read_abi_text_file(filename)
        else:
            return cls.__read_text_file(filename)



    @classmethod
    def __read_text_file(cls, filename):
        d = np.recfromtxt(filename, names=True)
        objectname = os.path.basename(filename).replace('.dat','')
        thisobj = cls(objectname, d.HJD, d.mag, d.dmag, d.pb, flux=False)
        return thisobj



    @classmethod
    def __read_abi_text_file(cls, filename):
        dtype = 'f8'*11+'|S1'
        objectname = os.path.basename(filename).replace('.dat','')
        d = np.recfromtxt(filename, names=True, skip_footer=1, invalid_raise=False, dtype=dtype)
        mask = (np.abs(d.CONTAM) < 0.1)
        thisobj = cls(objectname, d.HJD[mask], d.MAG[mask], d.MAGERR[MASK], d.FILTS[mask], flux=False)
        return thisobj



    @classmethod
    def frombin(cls, binfilename):
        return NotImplemented



    @classmethod
    def __read_DES_file(cls, filename, flux=False):
        sntype = None
        mwebv  = None
        zgal   = None
        zgalerr = None
        oid = os.path.basename(filename).replace('DES_SN','').replace('.DAT','').lstrip('0')
        
        t  = []
        pb = []
        f  = []
        df = []
        
        data = None
        with open(filename, 'r') as infile:
            data = infile.readlines()
        for entry in data:
            if entry.startswith('MWEBV:'):
                mwebv = entry.split()[1]
            if entry.startswith('HOST_GALAXY_PHOTO-Z:'):
                zgal, crap, zgalerr = entry.split()[1:]
            if entry.startswith('SNTYPE:'):
                sntype = entry.split()[1]
            if not entry.startswith('OBS:'):
                continue
            (mjd, filt, crap, flux, fluxerr) = entry.lstrip('OBS:').split()
            t.append(mjd)
            pb.append(filt)
            f.append(flux)
            df.append(fluxerr)

        t = np.array(t).astype('float64')
        pb = np.array(pb)
        f = np.array(f).astype('float64')
        df = np.array(df).astype('float64')

        if flux==False:
            mask = (f > 0)
            m = -2.5*np.log10(f[mask]) + 25.
            dm = (2.5/np.log(10.))*np.abs(df/f)[mask]
            thisobj = cls(objectname, t, m, dm, pb, flux=flux)
        else:
            thisobj = cls(oid, t, f, df, pb, flux=flux)
        header  = {'mwebv':mwebv, 'sntype':sntype, 'zgal':zgal, 'zgalerr':zgalerr, 'lc':data}
        thisobj.header = header
        return thisobj



    def periodscan(self, min_p, max_p, periods=None):
        """
        Computes a LombScargle periodgram from minp to maxp, or alternatively on the array of periods
        sets model, periods, P_multi and best_period, overwriting if they already exist
        """
        if len(self.time) == 0:
            return None, None, None
        if self.time.max() - self.time.min() < max_p:
            return None, None, None
        try:
            model = periodic.LombScargleMultibandFast(fit_period=True)
            model.optimizer.period_range = (min_p, max_p)
            model.fit(self.time, self.mag, self.mag_err, self.passband)
            if periods is None:
                periods, P_multi = model.periodogram_auto()
            else:
                P_multi = model.periodogram(periods)
            self.model = model
            self.best_period = model.best_period
            self.periods = periods
            self.P_multi = P_multi
            return model, periods, P_multi
        except:
            return None, None, None



    def phase_for_period(self, period, phase_offset=None):
        """
        Returns the phase for some period
        Exposed directly to trial different periods, but ideally should be used by get_phase
        """
        if phase_offset is None:
            phase_offset = 0.
        phase = ((self.time - phase_offset )/ period) % 1
        return phase



    def get_phase(self, per=None, phase_offset=None):
        """
        Returns the phase of the object
        if periodic, and best_period is set, simply returns phase with it
        if periodic, nad best_period is not set, computes a periodogram with some defaults, and returns it
        if not periodic, just returns phase relative to first event
        TODO - return phase relative to some time argument, and accept arguments for periodscan (maybe with *args, **kwargs)
        """
        phase = None
        if per is None:
            per = self.per #default is False
        if per:
            if self.best_period is None:
                if self.model is None:
                    # we don't have a model - calculate it
                    max_p = 100.
                    if self.time.max() - self.time.min() < 100:
                        max_p = self.time.max() - self.time.min()
                    model, periods, P_multi = self.periodscan(0.1, max_p)
                    if model is None:
                        # we failed to get a model - give up
                        return self.get_phase(per=None, phase_offset=phase_offset)
                    # we got a model - set the period
                    self.best_period = model.best_period
                    best_period = self.best_period
                else:
                    # we have a model - just restore the period from it 
                    best_period = self.model.best_period
            else:
                # we have a period - just restore it 
                best_period = self.best_period
            phase = self.phase_for_period(best_period, phase_offset=phase_offset)
        else:
            if phase_offset is None:
                phase_offset = 0.
            phase = self.time - phase_offset
        return phase



    def gaussian_process_alt_smooth(self, per=None, minobs=10, phase_offset=None, recompute=False, scalemin=None, scalemax=None):
        """
        per = cheaty hackish thing to get a gaussian process with some continuity at the end points
        minobs = mininum number of observations in each filter before we fit
        uses george and emcee
        """
        outgp = {}
        if self.outgp is not None:
            if not recompute:
                outgp = self.outgp
                return outgp


        filters = self.filters
        phase = self.get_phase(per=per, phase_offset=phase_offset)

        if phase is None:
            return outgp

        for i in range(len(filters)):
            pb = filters[i]
            mask = (self.passband == pb)
            nobs = len(phase[mask])
            if nobs < minobs:
                continue #I guess we could do linear interpolation or something here, but that seems useless.
            
            m2 = phase[mask].argsort()
            minphase = phase[mask][m2].min()
            maxphase = phase[mask][m2].max()

            thisphase = phase[mask][m2]
            thismag   = self.mag[mask][m2]
            thismag_err=self.mag_err[mask][m2]
            if per:
                # I'd been replicating the data hackishly to try and force the period to 1.0, but I can just fix it
                #thisphase  = np.hstack((thisphase-1., thisphase, thisphase+1.))            
                #thismag    = np.hstack((thismag, thismag, thismag))
                #thismag_err= np.hstack((thismag_err, thismag_err, thismag_err))

                kernel = kernels.ExpSine2Kernel(0.5, 1.0)
            else:
                kernel = kernels.ExpSquaredKernel(100.) * kernels.DotProductKernel()
    
            gp = george.GP(kernel, mean=thismag.mean())    
            def nll(p):
                gp.kernel[:] = p
                ll = gp.lnlikelihood(thismag, quiet=True)
                return -ll if np.isfinite(ll) else 1e25

            # And the gradient of the objective function.
            def grad_nll(p):
                gp.kernel[:] = p
                return -gp.grad_lnlikelihood(thismag, quiet=True)
            gp.compute(thisphase, thismag_err)
            p0 = gp.kernel.vector
            if per:
                results = op.minimize(nll, p0, jac=grad_nll, bounds=[(scalemin,scalemax),(0.,0.)])
            else:
                results = op.minimize(nll, p0, jac=grad_nll, bounds=[(scalemin,scalemax)])
            gp.kernel[:] = results.x    
            # george is a little different than sklearn in that the prediction stage needs the input data
            outgp[pb] = (gp, thisphase, thismag, thismag_err)
        self.outgp = outgp
        return outgp



    def gaussian_process_alt_smooth_with_RANSAC(self, per=None, minobs=10, phase_offset=None, recompute=False):
        """
        # DO NOT USE THIS UNTIL GSN IS DONE TESTING
        per = cheaty hackish thing to get a gaussian process with some continuity at the end points
        minobs = mininum number of observations in each filter before we fit
         uses george and emcee
        uses george and emcee
        """
        outgp = {}
        if self.outgp is not None:
            if not recompute:
                outgp = self.outgp
                return outgp

        filters = self.filters
        phase = self.get_phase(per=per, phase_offset=phase_offset)
        if phase is None:
            return outgp
        for i in range(len(filters)):
            pb = filters[i]
            mask = (self.passband == pb)
            nobs = len(phase[mask])
            if nobs < minobs:
                continue
            m2 = phase[mask].argsort()
            minphase = phase[mask][m2].min()
            maxphase = phase[mask][m2].max()

            thisphase =phase[mask][m2]
            thismag   =self.mag[mask][m2]
            thismag_err=self.mag_err[mask][m2]
            #if per:
            #    thisphase  = np.hstack((thisphase-1., thisphase, thisphase+1.))            
            #    thismag    = np.hstack((thismag, thismag, thismag))
            #    thismag_err= np.hstack((thismag_err, thismag_err, thismag_err))

            # some stupid data reshaping for RANSAC
            thisphase = np.atleast_2d(thisphase).T
            thismag = np.atleast_2d(thismag).T
            thismag_err = np.atleast_2d(thismag_err).T
            all_data = np.hstack((thisphase, thismag, thismag_err))

            model = GaussianProcessFit(0, 1, 2,per=per)
            tol_threshold = 3.*np.mean(thismag_err)
            fit_good_thresh = np.max([minobs, np.floor(0.85*nobs)])
            ransac_fit, ransac_data = ransac.ransac(all_data, model, minobs, 100, tol_threshold, fit_good_thresh, return_all=True)

            # george is a little different than sklearn in that the prediction stage needs the input data
            outgp[pb] = (ransac_fit, thisphase, thismag, thismag_err, ransac_data)
        self.outgp = outgp
        return outgp



    def spline_smooth(self, per=None, minobs=15, phase_offset=None, recompute=False):
        """
        per = fit a periodic spline
        minobs = mininum number of observations in each filter before we fit
        """
        outtck = {}
        
        if self.outtck is not None:
            if not recompute:
                outtck = self.outtck
                return outtck


        outticks = np.arange(0.05, 1., 0.15)
        w = 1./self.mag_err
        filters = self.filters
        phase = self.get_phase(per=per, phase_offset=phase_offset)
        if phase is None:
            return outtck
        for i in range(len(filters)):
            pb = filters[i]
            mask = (self.passband == pb)
            nobs = len(phase[mask])
            print(nobs)
            if nobs < minobs:
                print("Not enough observations in {}".format(pb))
                continue
            m2 = phase[mask].argsort()
            minphase = phase[mask][m2].min()
            maxphase = phase[mask][m2].max()
            phase_range = maxphase - minphase

            #The current knot generation fails on non-periodic lightcurves
            #useticks = outticks[((outticks > minphase) & (outticks < maxphase))]

            #Knot generation based on the range of the dataset, with the same number as before
            useticks = (outticks * phase_range) + minphase

            #Test for monotonically increasing phase
            mono_bool = np.any(np.diff(phase[mask][m2]) == 0)

            mag = self.mag
            mag_err = self.mag_err

            #Fix repeated values
            if(mono_bool):
                diff_idx = np.where(np.diff(phase[mask][m2]) == 0)[0]
                repeat_phases = np.split(diff_idx, np.where(np.diff(diff_idx) != 1)[0]+1)
                #print("Repeat Phases", repeat_phases)

                delete_idx = []

                for repeat_phase in repeat_phases:
                    new_mag = np.mean(mag[mask][m2][repeat_phase])
                    new_err = np.sqrt(np.sum((mag_err[mask][m2][repeat_phase])**2))

                    #Replace first value from the magnitude array
                    mag[mask][m2][repeat_phase[-1]] = new_mag

                    #Replace first value from the magnitude error array
                    mag_err[mask][m2][repeat_phase[-1]] = new_err

                    #Add to the delete_idx array
                    delete_idx.extend(repeat_phase.tolist())

                #Delete values from mag, mag_err, and phase
                #print("DeleteIDX", delete_idx)
                phase_spline = np.delete(phase[mask][m2], np.array(delete_idx))
                mag = np.delete(mag[mask][m2], np.array(delete_idx))
                mag_err = np.delete(mag_err[mask][m2], np.array(delete_idx))
            else:
                mag = mag[mask][m2]
                mag_err = mag_err[mask][m2]
                phase_spline = phase[mask][m2]

            #Debug printing
            #print("Used knots", useticks)
            #print("Original Phase", phase[mask][m2])
            #print("Phase", phase_spline)
            #print("mag", mag)
            #print("Error", mag_err)
            #print(np.diff(phase_spline))
            #print(any(np.diff(phase_spline) <= 0))
            tck = scinterp.splrep(phase_spline, mag,\
                     w=1./mag_err,\
                     k=3, per=per)
            outtck[pb] = tck
        self.outtck = outtck
        return outtck 



    def get_smooth_model_for_passband_phase(self, passband, phase, per=None, gpr=True, phase_offset=None, ransac=False, recompute=False):
        """
        Convenience function - return the smoothed model for some passband at some array of phases  
        Generally, it'll just make sense to call one of the smooth model functions, and set outgp or outtck, and do whatever the heck you need with them
        """

        pb = passband
        if gpr:
            if ransac:
            # DO NOT USE THIS UNTIL GSN IS DONE TESTING
                outgp = self.gaussian_process_alt_smooth_with_RANSAC(per=per, phase_offset=phase_offset, recompute=recompute)
            else:
                outgp = self.gaussian_process_alt_smooth(per=per, phase_offset=phase_offset, recompute=recompute)
        else:
            outtck = self.spline_smooth(per=per, phase_offset=phase_offset, recompute=recompute)
                
        mask = (self.passband == pb)
        if gpr:
            thisgp = outgp.get(pb)
            if thisgp is None:
                return (None, None)
            if ransac:
                # DO NOT USE THIS UNTIL GSN IS DONE TESTING
                ransac_fit, thisphase, thismag, thismag_err, ransac_data = thisgp
                bestgp, compphase, compmag, compmag_err  = ransac_fit
            else:
                bestgp, thisphase, thismag, thismag_err = thisgp
        
            interpy, cov = bestgp.predict(thismag, phase)
            sigma = np.sqrt(np.diag(cov))
        else:
            thistck = outtck.get(pb)
            if thistck is None:
                return (None, None)
            interpy = scinterp.splev(phase, thistck)
            sigma = np.zeros(len(interpy))
        return interpy, sigma



    def model_and_align_objects(self, per=None, gpr=True, phase_offset=None, ransac=False, recompute=False):
        """
        Align splines to some particular phase
        """
        filters = self.filters
        phase = self.get_phase(per=per, phase_offset=phase_offset)

        if gpr:
            if ransac:
            # DO NOT USE THIS UNTIL GSN IS DONE TESTING
                outgp = self.gaussian_process_alt_smooth_with_RANSAC(per=per, phase_offset=phase_offset, recompute=recompute)
            else:
                outgp = self.gaussian_process_alt_smooth(per=per, phase_offset=phase_offset, recompute=recompute)
        else:
            outtck = self.spline_smooth(per=per, phase_offset=phase_offset, recompute=recompute)
                
        outdmdt = {}
        for i in range(len(filters)):
            pb = filters[i]
            mask = (self.passband == pb)

            outphase = np.linspace(phase[mask].min(), phase[mask].max(), num=1000, endpoint=True)
            if per or self.per:
                # temporary hack just to visualize periodic lightcurves
                outphase = np.linspace(0.,2.0,num=5000,endpoint=True)

            if gpr:
                thisgp = outgp.get(pb)
                if thisgp is None:
                    continue 
                if ransac:
                    # DO NOT USE THIS UNTIL GSN IS DONE TESTING
                    ransac_fit, thisphase, thismag, thismag_err, ransac_data = thisgp
                    bestgp, compphase, compmag, compmag_err  = ransac_fit
                else:
                    bestgp, thisphase, thismag, thismag_err = thisgp

                interpy, cov = bestgp.predict(thismag, outphase)
                sigma = np.sqrt(np.diag(cov))
                ax1.fill(np.concatenate([outphase, outphase[::-1]]),\
                     np.concatenate([interpy - 1.9600 * sigma,(interpy + 1.9600 * sigma)[::-1]]),\
                     alpha=.5, fc='grey', ec='None', label='95% confidence interval')
            else:
                thistck = outtck.get(pb)
                if thistck is None:
                    continue 
                interpy = scinterp.splev(outphase, thistck)

            dm = nn2_diffs(interpy)                            
            dt = nn2_diffs(outphase)                            
            outdtdm[pb] = (dt, dm)
        return outdmdt
        


    def get_amplitude(self, smoothed=True, per=None, gpr=True, ransac=False, phase_offset=None, recompute=False):
        outamp = {}
        if self.amplitude is not None:
            if not recompute:
                outamp = self.amplitude
                return outamp

        filters = self.filters
        phase = self.get_phase(per=per, phase_offset=phase_offset)

        if smoothed==True:
            # if smoothed, we compute the amplitude based on the smoothed curve
            if gpr:
                if ransac:
                    # DO NOT USE THIS UNTIL GSN IS DONE TESTING
                    outgp = self.gaussian_process_alt_smooth_with_RANSAC(per=per, phase_offset=phase_offset, recompute=recompute)
                else:
                    outgp = self.gaussian_process_alt_smooth(per=per, phase_offset=phase_offset, recompute=recompute)
            else:
                outtck = self.spline_smooth(per=per, phase_offset=phase_offset, recompute=recompute)
            
            for i in range(len(filters)):
                pb = filters[i]
                mask = (self.passband == pb)
                thispbphase = phase[mask]
                minph = thispbphase.min()
                maxph = thispbphase.max()

                if per or self.per or per is not None:
                    outphase = np.linspace(0.,2.0,num=200,endpoint=True)
                else:
                    npb = len(thispbphase)
                    if npb > 25:
                        outphase = np.linspace(minph, maxph, num=100, endpoint=True)
                    else:
                        outphase = thispbphase

                if gpr:
                    thisgp = outgp.get(pb)
                    if thisgp is None:
                        continue 
                    if ransac:
                        # DO NOT USE THIS UNTIL GSN IS DONE TESTING
                        ransac_fit, thisphase, thismag, thismag_err, ransac_data = thisgp
                        bestgp, compphase, compmag, compmag_err  = ransac_fit
                    else:
                        bestgp, thisphase, thismag, thismag_err = thisgp
                    interpy, cov = bestgp.predict(thismag, outphase)
                else:
                    thistck = outtck.get(pb)
                    if thistck is None:
                        continue 
                    interpy = scinterp.splev(outphase, thistck)
                amp = interpy.ptp()
                outamp[pb] = amp
        else:
            for i in range(len(filters)):
                pb = filters[i]
                mask = (self.passband == pb)
                nobs = len(phase[mask])
                if nobs == 1:
                    key = '%sRefMag' %pb
                    amp = self.mag[mask] - self.header.get(key, self.mag[mask])
                     #default value if header key is absent is to use the same value - i.e. we don't trigger and return amplitude = 0
                    outamp[pb] = amp
                else:
                    # the percentile formula is one way of trimming outliers, but in practice it seems to be a worse estimator
                    # than just trusting the photometry and making an error cut
                    # since the really large outliers often also have large errors
                    # 
                    #amp = (np.percentile(self.mag[mask],98)-np.percentile(self.mag[mask],2))*1.02
                    m2 = (self.mag_err[mask] < 0.2)
                    if len(self.mag[mask][m2]) > 1:
                        amp = self.mag[mask][m2].ptp()
                    else:
                        amp = 0.
                    outamp[pb] = amp
                    
        # make sure amplitudes are set to 0 for bands that do not get us a useful amplitude
        for i in range(len(filters)):
            pb = filters[i]
            if pb not in outamp:
                outamp[pb] = 0.
        self.amplitude = outamp
        return outamp
        


    def get_stats(self, smoothed=True, per=None, gpr=True, ransac=False, phase_offset=None, recompute=False):
        outstats = {}
        if self.stats is not None:
            if not recompute:
                outstats = self.stats
                return outstats

        filters = self.filters
        phase = self.get_phase(per=per, phase_offset=phase_offset)

        if smoothed==True:
            # if smoothed, we compute the amplitude based on the smoothed curve
            if gpr:
                if ransac:
                    # DO NOT USE THIS UNTIL GSN IS DONE TESTING
                    outgp = self.gaussian_process_alt_smooth_with_RANSAC(per=per, phase_offset=phase_offset, recompute=recompute)
                else:
                    outgp = self.gaussian_process_alt_smooth(per=per, phase_offset=phase_offset, recompute=recompute)
            else:
                outtck = self.spline_smooth(per=per, phase_offset=phase_offset, recompute=recompute)

            
            for i in range(len(filters)):
                pb = filters[i]
                mask = (self.passband == pb)

                outphase = np.linspace(phase[mask].min(), phase[mask].max(), num=100, endpoint=True)
                if per or self.per:
                    outphase = np.linspace(0.,2.0,num=1000,endpoint=True)

                if gpr:
                    thisgp = outgp.get(pb)
                    if thisgp is None:
                        nobs = len(phase[mask]) 
                        m2 = (self.mag_err[mask] < 0.1)
                        if len(self.mag_err[mask][m2]) == 0:
                            continue
                        thisstat = scipy.stats.describe(self.mag[mask][m2])
                        outstats[pb] = thisstat
                        continue 
                    if ransac:
                        # DO NOT USE THIS UNTIL GSN IS DONE TESTING
                        ransac_fit, thisphase, thismag, thismag_err, ransac_data = thisgp
                        bestgp, compphase, compmag, compmag_err  = ransac_fit
                    else:
                        bestgp, thisphase, thismag, thismag_err = thisgp
                    interpy, cov = bestgp.predict(thismag, outphase)
                else:
                    thistck = outtck.get(pb)
                    if thistck is None:
                        continue 
                    interpy = scinterp.splev(outphase, thistck)
                thisstat = scipy.stats.describe(interpy)
                outstats[pb] = thisstat
        else:
            for i in range(len(filters)):
                pb = filters[i]
                mask = (self.passband == pb)
                nobs = len(phase[mask])
                m2 = (self.mag_err[mask] < 0.1)
                if len(self.mag[mask][m2]) != 0:
                    thisstat = scipy.stats.describe(self.mag[mask][m2])
                    outstats[pb] = thisstat
                else:
                    continue
                    
        # unlike the amplitude, which has a reasonable default 
        # we don't have useful defaults for most things
        # min, max, mean, std, kurtosis, skewness
        self.stats = outstats
        return outstats


    def plot_object(self, per=None, smoothed=True, gpr=True, phase_offset=None, ransac=False,\
                     recompute=False, minobs=25, savefn=None, show=False):
        """
        Quick and dirty plot of the object
        per = force the object to be treated as periodic
        smoothed = fit a smoothed model - default is using a gaussian process regression
            gpr = use gaussian process regression to compute the smoothed model
                else use a spline (not a very good idea)
            ransac = use gaussian process regression + RANSAC for outlier filtering (NOT TESTED DO NOT USE)
        phase_offset = if you want to pass a phase_offset down to get_phase - shifts the phase curve/date range
        recompute - recompute the smoothing function
        minobs - passed to smoothing functions to decide what the minimum number of observations for smoothing is
        savefn - save filename for plot
        show - display the plot (false)
        returns matplotlib.Figure instance
        """
        filters = self.filters
        phase = self.get_phase(per=per, phase_offset=phase_offset)

        fig = plt.figure(figsize=(10,8))
        if per:
            if self.model is None:
                # if we have a model that means the lomb-scargle periodgram was run for this object
                # in that case plot the periodogram and the light curve, else just the light curve
                ax1 = fig.add_subplot(1,1,1)
            else:
                ax1 = fig.add_subplot(1,2,1)
                ax2 = fig.add_subplot(1,2,2)
                ax2.plot(self.periods, self.P_multi)
                ax2.set_xscale('log')
                ax2.set_title('Multiband Periodogram', fontsize=12)
                ax2.yaxis.set_major_formatter(plt.NullFormatter())
                ax2.set_xlabel('Period (days)')
                ax2.set_ylabel('Lomb-Scargle Power')
        else:
            # not a periodic object, so just plot the light curve
            ax1 = fig.add_subplot(1,1,1)

#        inset = fig.add_axes([0.78, 0.56, 0.15, 0.3])          
#        inset.errorbar(self.time, self.mag, self.mag_err, fmt='ok', elinewidth=1.5, capsize=0)
#        inset.invert_yaxis()
#        inset.set_xlabel('HJD')
#        inset.set_ylabel('Magnitude')

        if smoothed==True:
            # if we're overplotting the smoothed model on the data, check of the outgp has been computed
            if gpr:
                if ransac:
                    # DO NOT USE THIS UNTIL GSN IS DONE TESTING
                    outgp = self.gaussian_process_alt_smooth_with_RANSAC(per=per, phase_offset=phase_offset, recompute=recompute,\
                            minobs=minobs)
                else:
                    outgp = self.gaussian_process_alt_smooth(per=per, phase_offset=phase_offset, recompute=recompute,\
                     minobs=minobs)
            else:
                outtck = self.spline_smooth(per=per, phase_offset=phase_offset, recompute=recompute,minobs=minobs)
            

        lines = []
        for i in xrange(len(filters)):
            pb = filters[i]
            mask = (self.passband == pb)
            thisline = ax1.errorbar(phase[mask], self.mag[mask], yerr=self.mag_err[mask],\
                                    capsize=0, marker='.', linestyle='None', label=pb) 
            lines.append(thisline)
            if smoothed:
                outphase = np.linspace(phase[mask].min(), phase[mask].max(), num=100, endpoint=True)
                if per or self.per:
                    # temporary hack just to visualize periodic lightcurves
                    outphase = np.linspace(-1.0,2.0,num=500,endpoint=True)

                if gpr:
                    thisgp = outgp.get(pb)
                    if thisgp is None:
                        continue 
                    if ransac:
                        # DO NOT USE THIS UNTIL GSN IS DONE TESTING
                        ransac_fit, thisphase, thismag, thismag_err, ransac_data = thisgp
                        bestgp, compphase, compmag, compmag_err  = ransac_fit
                    else:
                        bestgp, thisphase, thismag, thismag_err = thisgp

                    interpy, cov = bestgp.predict(thismag, outphase)
                    sigma = np.sqrt(np.diag(cov))
                    ax1.fill(np.concatenate([outphase, outphase[::-1]]),\
                         np.concatenate([interpy - 1.9600 * sigma,(interpy + 1.9600 * sigma)[::-1]]),\
                         alpha=.5, fc='grey', ec='None', label='95% confidence interval')
                else:
                    thistck = outtck.get(pb)
                    if thistck is None:
                        continue 
                    interpy = scinterp.splev(outphase, thistck)
                ax1.plot(outphase, interpy, linestyle='-', marker='None',lw=0.5, color='k')
        #endfor    
        ax1.set(xlabel='Phase', ylabel='Magnitude')
        ax1.invert_yaxis()
        fig.legend(lines, filters, 'upper right')    
        plt.tight_layout()
        if savefn is not None:
            try:
                fig.savefig(savefn)
                plt.close(fig)
            except (OSError, IOError) as e:
                message = 'Could not save file %s - %s'%(savefn, str(e))
                warnings.warn(message, RuntimeWarning)
        if show:
            plt.ion()
            plt.show(fig)
            return fig




def get_options():
    types = 'rrlyr,acep,acv,cep,decep,dmcep,dn,dpv,dsct,ecl,ell,lpv,misc,pul,rcb,rcrb,socep,sxphe,t2cep,unclassified,wd,yso'
    threadct  = range(1, multiprocessing.cpu_count()+1)
    parser = argparse.ArgumentParser(description='Select class of variable stars to process through the touchstone algorithms')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-t', '--transient',  dest='transient', action='store_true', help='Process transients')
    group.add_argument('-v', '--varclass', metavar="CLASS", dest='varclass', nargs='+',\
                        choices=types.split(','), help='Specify the variable class(es) you want to (re)process through the Touchstone algorithms. Valid options are (%s)'%types ) 
    parser.add_argument('-n', '--nproc',   dest='nproc',   type=int,        choices=threadct,  default=5,   metavar='', help='specify   the number of processes to run (default=%(default)d)')
    args = parser.parse_args()
    return args
            

def crap(f):
    print(f)
    q = TouchstoneObject.fromfile(f)
    print(q)

def proc_transients():
    pass

def proc_variables(varclasses):
    path = '/data/antares/aux/OGLE/parsed'
    pjoin = os.path.join
    files = []
    types = []
    for varclass in varclasses:
        filepath = pjoin(path,varclass,'*.dat')
        thisclassfiles = glob.glob(filepath)
        types += [varclass]*len(thisclassfiles)
        files += thisclassfiles
    return files, types




    
def main():
    path = '/data/antares/aux/OGLE/parsed'
    args = get_options()

    if args.transient is True:
        pass    
        # do transient thing here
    else:   
        if args.varclass is None:
            message = 'Need at least one variable or transient lightcurve set to process. Specify -h to see options.'
            raise RuntimeError(message)

        files, types = proc_variables(args.varclass)


    processPool = multiprocessing.Pool(args.nproc)
    lock = multiprocessing.Lock()

    task = None
    # TODO define stagefunc
    stagefunc = crap
    for f  in files:
        task = processPool.apply_async(stagefunc, args=(f,))
        
    processPool.close()
    processPool.join()
    if task is not None:
        task.get()



    




if __name__=='__main__':
    sys.exit(main())
