"""
Main Structure of the bolometric lightcurve object
and bolometric Fit object.
"""

import os
#import csv
import pandas as pd
import numpy as np
import scipy as sp
import math as m
import warnings
import astropy
import pickle

from lightcurve_fitting.lightcurve import LC
from lightcurve_fitting.bolometric import calculate_bolometric
from astropy.table import Table
from matplotlib import pyplot as plt
import bolometric_modelling.models as mdl


class Bol_LC(LC):
    """
    A broadband and bolomteric light curve,
    Parent - LC from 'lightcurve_fitting.lightcurve'
    Grandparent - :class:`astropy.table.Table` a(see ``help(Table)`` for more details)
    

    Attributes
    ----------
    SN : string
        Name of the Astrnomical Transient of which this object corresponds to.
    z : float
        Redshift of Transient.
    coords : 2-Tuple
        Coordinates of the event in J2000 - (ra, dec).
    filters: dict
        Dictionary of exctinction values for this transient.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z = None
        self.coords = (None, None) #(ra, dec)        
        self.filters = None
        self.SN = None
        self.bolometric_data = dict()

    
    def __convert_to_erg_s(blackbody_data):
        """
        Internal Unit Converter

        Parameters
        ----------
        blackbody_data : Table / DataFrame
            In units of Watts (W)

        Returns
        -------
        blackbody_data : Table / Dataframe
            In units of erg/s

        """
        blackbody_data['Lum'] = blackbody_data['Lum'] * 1e7
        blackbody_data['dLum0'] = blackbody_data['dLum0'] * 1e7
        return blackbody_data


    def __convert_to_K(blackbody_data):
        """
        Internal Unit converter

        Parameters
        ----------
        blackbody_data : Table / Dataframe
            Units in KiloKelvin

        Returns
        -------
        blackbody_data : Table / Dataframe
            Units in Kelvin

        """
        blackbody_data['temp']=blackbody_data['temp']*1000
        blackbody_data['dtemp0']=blackbody_data['dtemp0']*1000
        return blackbody_data


    def __convert_to_cm(blackbody_data):
        """
        Internal Unit Converter 

        Parameters
        ----------
        blackbody_data : Table
            Units - Solar Radii
        Returns
        -------
        blackbody_data : Table
            Units - cm

        """
        blackbody_data['radius']= blackbody_data['radius']*(6.9634*10**13)
        blackbody_data['dradius0']= blackbody_data['dradius0']*(6.9634*10**13)
        return (blackbody_data)


    def __calc_bolo(blackbody_data):
        blackbody_data['Lum']= np.array(4 * (m.pi) * ((blackbody_data['radius'])**2) * (5.6703*10**(-5)) * ((blackbody_data['temp'])**4))
        blackbody_data['dLum0']= np.array((((8 * (m.pi) * (blackbody_data['radius']) * (5.6703*10**(-5)) * ((blackbody_data['temp'])**4)*(blackbody_data['dradius0']))**2)+((16 * (m.pi) * ((blackbody_data['radius'])**2) * (5.6703*10**(-5)) * ((blackbody_data['temp'])**3)*(blackbody_data['dtemp0']))**2))**0.5)
        blackbody_data['dLum1']= np.array((((8 * (m.pi) * (blackbody_data['radius']) * (5.6703*10**(-5)) * ((blackbody_data['temp'])**4)*(blackbody_data['dradius1']))**2)+((16 * (m.pi) * ((blackbody_data['radius'])**2) * (5.6703*10**(-5)) * ((blackbody_data['temp'])**3)*(blackbody_data['dtemp1']))**2))**0.5)
        return blackbody_data
    
    
    def __prep_MCMC(self, bin_num=1):
        
        #radius and temp units
        self.bolometric_data['mcmc'] = self.__convert_to_K(self.bolometric_data['mcmc'] )
        self.bolometric_data['mcmc']  = self.__convert_to_cm(self.bolometric_data['mcmc'] )
        #calc lum
        self.bolometric_data['mcmc']  = self.__calc_bolo(self.bolometric_data['mcmc'] )
        #binning
        bb_t_bin = self.__bin_data(self.bolometric_data['mcmc']['t_relative_to_peak'],
                                   self.bolometric_data['mcmc']['temp'],
                                   self.bolometric_data['mcmc']['dtemp0'],
                                   self.bolometric_data['mcmc']['dtemp1'], bin_num)
        bb_r_bin = self.__bin_data(self.bolometric_data['mcmc']['t_relative_to_peak'],
                                   self.bolometric_data['mcmc']['radius'],
                                   self.bolometric_data['mcmc']['dradius0'],
                                   self.bolometric_data['mcmc']['dradius1'], bin_num)
        bb_l_bin = self.__bin_data(self.bolometric_data['mcmc']['t_relative_to_peak'],
                                   self.bolometric_data['mcmc']['Lum'],
                                   self.bolometric_data['mcmc']['dLum0'],
                                   self.bolometric_data['mcmc']['dLum1'], bin_num)
        
        self.bolometric_data['binned_mcmc'] = (bb_t_bin, bb_r_bin, bb_l_bin)
        return #bb_t_bin, bb_r_bin, bb_l_bin


    def __prep_scipy(self, bin_num=1):
        
        #radius and temp units
        self.bolometric_data['scipy']  = self.__convert_to_K(self.bolometric_data['scipy'])
        self.bolometric_data['scipy'] = self.__convert_to_cm(self.bolometric_data['scipy'])
        self.bolometric_data['scipy'] = self.__convert_to_erg_s(self.bolometric_data['scipy'])
        #binning
        bb_t_bin = self.__bin_data(self.bolometric_data['scipy']['t_relative_to_peak'],
                                   self.bolometric_data['scipy']['temp'],
                                   self.bolometric_data['scipy']['dtemp0'],None, bin_num)
        bb_r_bin = self.__bin_data(self.bolometric_data['scipy']['t_relative_to_peak'],
                                   self.bolometric_data['scipy']['radius'],
                                   self.bolometric_data['scipy']['dradius0'],None, bin_num)
        bb_l_bin = self.__bin_data(self.bolometric_data['scipy']['t_relative_to_peak'],
                                   self.bolometric_data['scipy']['Lum'],
                                   self.bolometric_data['scipy']['dLum0'],None, bin_num)
        
        self.bolometric_data['binned_scipy'] = (bb_t_bin, bb_r_bin, bb_l_bin)
        return #bb_t_bin, bb_r_bin, bb_l_bin

    
    def __bin_data(x, y, e0, e1=None ,binsize=1):
        
        
        if e1 is None:
            e1 = e0
        # Sort:
        s = np.argsort(x)
        x = x[s]
        y = y[s]
        e0 = e0[s]
        e1 = e1[s]
        e = np.mean([e0,e1],0)
        if 0 in e:
            weight = np.tile(1,len(e))
        else:
            weight = 1/e**2
        # Do the binning using weighted averages:
        toaverage = [0]
        binned_x = np.array([])
        binned_y = np.array([])
        binned_e0 = np.array([])
        binned_e1 = np.array([])
        for i in range(1,len(x)):
            if abs(x[i]-x[toaverage[0]]) < binsize:
                toaverage.append(i)
            else:
                binned_x = np.append(binned_x,sum(weight[toaverage]*x[toaverage])/sum(weight[toaverage]))
                binned_y = np.append(binned_y,sum(weight[toaverage]*y[toaverage])/sum(weight[toaverage]))
                binned_e0 = np.append(binned_e0,sum(weight[toaverage]*e0[toaverage])/sum(weight[toaverage]))
                binned_e1 = np.append(binned_e1,sum(weight[toaverage]*e1[toaverage])/sum(weight[toaverage]))
                toaverage = [i]
        binned_x = np.append(binned_x,sum(weight[toaverage]*x[toaverage])/sum(weight[toaverage]))
        binned_y = np.append(binned_y,sum(weight[toaverage]*y[toaverage])/sum(weight[toaverage]))
        binned_e0 = np.append(binned_e0,sum(weight[toaverage]*e0[toaverage])/sum(weight[toaverage]))
        binned_e1 = np.append(binned_e1,sum(weight[toaverage]*e1[toaverage])/sum(weight[toaverage]))
        s = np.argsort(binned_x)
        return(binned_x[s],binned_y[s],binned_e0[s],binned_e1[s])

    @property
    def find_extinction(self):
        
        from astroquery.irsa_dust import IrsaDust
        
        ra = self.coords[0].replace(':', ' ')
        dec = self.coords[1].replace(':', ' ')
        coords = ra + ' ' + dec    
        t = IrsaDust.get_extinction_table(coords)
        types = ['CTIO', 'SDSS', 'UKIR']
        
        filt = dict()
        for i in t:
            if i['Filter_name'][0:4] in types:
                filt[i['Filter_name'][-1]] = i['A_SandF']
                
        self.filters = filt
        return

    @classmethod
    def read(cls, filepath, name, z, ra, dec,
             format='ascii', fill_values=None, **kwargs):
        """

        Parameters
        ----------
        filepath : string
            Filepath to document containing lightcurve data
        name : string
            Supernova/event Name
        z : float
            redshift of event
        ra : string
            ra coordinate (J2000). Format - \'hh:mm:ss.ss\'
        dec : string
            declination (J2000). Format - \'+hh:mm:ss.ss\'
        format : string, optional
            File format type. The default is 'ascii'.
        fill_values : tuple of strings, optional
            Specific characters or strings to interchange. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        t : Bol-LC
            Bol-LC object of data given.

        """
        t = Table.read(filepath,format=format)
        
        try:
            t['Error'].name = 'dmag'
        except KeyError:
            try:
                t['error'].name = 'dmag'
            except KeyError:
                print('Problem finding error column in data!')
                pass
        
        _, file_ext = os.path.splitext(filepath)
        
        t.write(filepath.replace(file_ext,'_pipe.txt'),format='ascii', overwrite=True, delimiter=' ')
        filepath = filepath.replace(file_ext,'_pipe.txt')
        del t
        
        t = super(Bol_LC, cls).read(filepath=filepath, format='ascii', fill_values=fill_values, **kwargs)
        
        t.z = z
        t.coords = (ra, dec)        
        t.SN = name
        
        return t


    def bolometric_extract(self, t0, outpath1):
        """
        
        Parameters
        ----------
        t0 : float
            Photometric Peak date of event (MJD / JD)
        outpath1 : string
            Path for saving MCMC plots in calculating
            the bolometric lightcurve
            
        Returns
        -------
        None.
        
        """
    
        from lightcurve_fitting.bolometric import calculate_bolometric
        import astropy.cosmology as cosmo
        from astropy.cosmology import WMAP9 as wmap
    
        if self.filters == None:
            try:
                self.find_extinction
            except:
                print('Unable to find extinction values - Coordinates not provided')
                pass
    
        self.meta['dm'] = cosmo.wCDM(wmap.H(0),0.27,0.73).distmod(self.z).value #Should Review
        self.meta['extinction'] = self.filters
        
        t = calculate_bolometric(self, self.z, outpath1, burnin_steps=300,
                                                    steps=500, nwalkers=20, res=1)
        t['t_relative_to_peak'] = (t['MJD']) - t0
        
        t.write(outpath1 + '\\Bolometric_table.txt', format='ascii.fixed_width', overwrite=True)
        
        df = t.to_pandas()
        
        self.bolometric_data['mcmc'] = df.copy()[['MJD',
                                                  't_relative_to_peak',
                                                  'dMJD0',
                                                  'dMJD1',
                                                  'temp_mcmc',
                                                  'dtemp0',
                                                  'dtemp1',
                                                  'radius_mcmc',
                                                  'dradius0',
                                                  'dradius1',
                                                  'Lum',
                                                  'dLum0',
                                                  'dLum1']]
        
        self.bolometric_data['scipy'] = df.copy()[['MJD',
                                                   't_relative_to_peak',
                                                   'dMJD0',
                                                   'dMJD1',
                                                   'temp',
                                                   'dtemp',
                                                   'radius',
                                                   'dradius',
                                                   'Lum',
                                                   'dLum0']]
        
        #units and bolometric luminosity
        self.__prep_MCMC()
        self.__prep_scipy()
                
        return 
    

    def write(self, path=''):
        
        try:
            for f in ['binned_mcmc', 'binned_scipy']:
                file = path + '/' + 'Bolometric_Data_' + f + '.csv'
                dats = np.array(self.bolometric_data[f]).transpose
                np.savetxt(file, dats, delimiter=',')
        except:
            pass
        
        file_path = path + '/' + 'Bolometric_Data.csv'
        
        super().write(file_path, format='csv')

        
        return

class bol_fit:
    def __init__(self):
        #priors and setup
        self.RD_priors = None
        self.RDCSM_priors = None
        self.time_prior = None
        
        self.bol_lc = None
        
        #results and samplers
        self.RD_mcmc = None
        self.CSM_mcmc = None 
        self.CSM_ns = None
        
        __spec__ = None
        
    
    def prep_data_model(self, data=None, path=None, delimiter=',', cutoff=0):
        
        try:
            if data == None and path == None:
                raise Exception(1)
        except:
            print('No data or path given. Terminating')
            raise 
        
        if isinstance(path, type(None)) == False:
            data = np.genfromtxt(path, delimiter=delimiter)
        
        t = data[1:,2]
        y = data[1:, 11] * 10e7
        yerr = ((data[1:, 11] + data[1:, 12])/2) * 1e7
    
        if cutoff != 0:
            mask = t < cutoff
            t = t[mask]
            y = y[mask]
            yerr = yerr[mask]
        
        self.bol_lc = (t, y, yerr)
        return t, y, yerr
            

    def RD_fit_mcmc(self, save_to='', priors=((0.001, 0.001, 0.01, 1e-5),(20, 20, 10, 1)),
               niter=5000, nwalkers=100, owarnings=False):
        #-----------------------------------------------------------
        #Extract Data from Lightcurve
        t, y, yerr = self.bol_lc
        t_prior = self.time_prior
        #-----------------------------------------------------------
        #Set up Time Priors
        try:
            if (not isinstance(t_prior, tuple)) or (not isinstance(t_prior[1], tuple)):
                raise TypeError
        except TypeError:
            print('invalid t_0 prior type given! Please insert as a 2-nested tuple ((t_min,), (t_max,))')
            return
        
        prilow = priors[0] + t_prior[0]
        prihigh = priors[1] + t_prior[1]
        priors = (prilow, prihigh)
        
        #-----------------------------------------------------------
        import scipy.optimize as opt
        print('Starting Scipy Curvefit...')
        out_mean, out_var = opt.curve_fit(mdl.RD_model, t, y, bounds=priors,
                                                  p0 = [1, 1, 1, 0.2, (t_prior[1][0] - 1)], maxfev=8000)
        print('Scipy Curvefit Success')
    
        sfig = plt.figure(dpi=1200)
        ax = sfig.subplots()
        ax.scatter(t, y)
    
        #------------------------------------------------------------
        #MCMC Fitting
        def lnlike(theta, t, y, yerr):
            m_ni, m_ej, v_ph, k_o, t_0 = theta
            model = mdl.RD_model(t, m_ni, m_ej, v_ph, k_o, t_0)
            #print('Model Val: ', model)
            likelihood =  -0.5 * np.sum(np.log(2 * np.pi * yerr ** 2) + ((y - model) / yerr) ** 2)
            if np.isnan(likelihood):
                return -np.inf
            else:
                return -0.5 * np.sum(np.log(2 * np.pi * yerr ** 2) + ((y - model) / yerr) ** 2)
    
    
        def lnprior(theta):
            m_ni, m_ej, v_ph, k_o, t_0 = theta
            if (priors[0][0] < m_ni < priors[1][0]
                and priors[0][1] < m_ej < priors[1][1]
                and priors[0][2] < v_ph < priors[1][2]
                and priors[0][3] < t_0 < priors[1][3]):
                return 0.0
            return -np.inf
    
    
        def lnprob(theta, t, y, yerr):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, t, y, yerr)
    
        ndim = len(out_mean)
        p0 = [out_mean + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
        
        import emcee
        from pathos.multiprocessing import ProcessingPool as Pool
        
        print('Starting MCMC fit...')
        #if __name__ == '__main__':  #The horror of windows...
        if True:
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, y, yerr), pool=pool)
    
                print("Running burn-in...")
                p0, _, _ = sampler.run_mcmc(p0, 150, progress=True)
                sampler.reset()
        
                print("Running production...")
                pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
        
                print('MCMC Analysis complete!')
                #--------------------------------------------------
                
                self.RD_mcmc = sampler
                
                
                #PLOTTING
                ts = np.linspace(t[0], t[-1], 100)
                samples = sampler.flatchain
                for theta in samples[np.random.randint(len(samples), size=100)]:
                    ax.plot(ts, mdl.RD_model(ts, theta[0], theta[1], theta[2], theta[3], theta[4]), color="r", alpha=0.1)
                print(theta)
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                ax.set_xlabel('Days from Peak')
                ax.set_ylabel('Luminosity')
    
                import corner
                fig = corner.corner(samples, show_titles=True, plot_datapoints=True,
                                    labels=["$M_{Ni}$", "$M_{ej}$", "$v_{ph}$", "$\kappa_{\gamma}$", "$t_{0}$"],
                                    quantiles=[0.16, 0.5, 0.84])
        
                #del self.RD_mcmc.pool
                if save_to:
                    print('saving sampler.flatchain...')
                    np.save(save_to + '\\flatchain_RD', sampler.flatchain)
                    print('Saving Cornerplot...')
                    fig.savefig(save_to + '\\Cornerplot_RD.pdf', dpi=1200)
                    print('Saving smapler plot...')
                    sfig.savefig(save_to + '\\Sampleplot_RD.pdf', dpi=1200)
                    #print('saving bol_it object...')
                    #filehandler = open(save_to + '\\Bolfit.obj',"wb")
                    #pickle.dump(self, filehandler)
                    #filehandler.close()
                    print('Saving Complete!')
                
                mcmc = []
                q = []
                for i in range(ndim):
                    mcmc.append(np.percentile(samples[:, i], [16, 50, 84]))
                    q.append(np.diff(mcmc))
                
                return mcmc, q
        
        
    def RDCSM_fit_mcmc(self, n=7, delt=0, s=0, save_to='',
                   priors = ((0.01, 1e-3, 1e-3, 1e-3, 0.01, 0.01, 1e-3, 1e-3, 1e-4),
                  (12, 20, 20, 20, 50, 20, 1, 1, 1)),
                   niter=2500, nwalkers=100, owarnings=False):
        
        import scipy.constants as cst
        import warnings 
        if not owarnings:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
        #----------------------------------------------------------------
        #parameters
        t, y, yerr = self.bol_lc
        t_prior = self.time_prior
        
        params = ['v_sn', 'm_ej', 'm_ni', 'm_csm', 'rho', 'r_in', 'e','x_0', 'k_o', 't_0']
        #-------------------------------------------------------------
        #Set up priors (full)
        try:
            if (not isinstance(t_prior, tuple)) or (not isinstance(t_prior[1], tuple)):
                raise TypeError
        except TypeError:
            print('invalid t_0 prior type given! Please insert as a 2-nested tuple ((t_min,), (t_max,))')
            return
        
        prilow = priors[0] + t_prior[0]
        prihigh = priors[1] + t_prior[1]
        priors = (prilow, prihigh)
        
        #-------------------------------------------------------------
        #Scipy Curvefit
        import scipy.optimize as opt
        times = np.linspace(t_prior[0][0], t[-1], 100)
        print('Starting Scipy Curvefit...')
        out_mean, out_var = opt.curve_fit(mdl.RDCSM_model, t, y, bounds=priors)
        print('Scipy Curvefit Success')
    
        sfig = plt.figure(dpi=1200)
        ax = sfig.subplots()
        ax.scatter(t, y) #Data
    
        for i in range(len(params)):
            print(params[i] + ': ', out_mean[i])
        
        mod = mdl.RDCSM_model(times, out_mean[0], out_mean[1], out_mean[2], out_mean[3], out_mean[4], out_mean[5], out_mean[6], out_mean[7], out_mean[8], out_mean[9])
        plt.ylim(bottom=0)
        plt.plot(times, mod)
        
        #----------------------------------------------------------
        #MCMC Fitting
    
        def lnlike(theta, t, y, yerr):
            v_sn, m_ej, m_ni, m_csm, rho, r_in, e, x_0, k_o, t_0 = theta
            model = mdl.RDCSM_model(t, v_sn, m_ej, m_ni, m_csm, rho, r_in, e, x_0, k_o, t_0)
            likelihood =  -0.5 * np.sum(np.log(2 * np.pi * yerr ** 2) + ((y - model) / yerr) ** 2)
            if np.isnan(likelihood):
                return -np.inf
            else:
                return -0.5 * np.sum(np.log(2 * np.pi * yerr ** 2) + ((y - model) / yerr) ** 2)
    
    
        def lnprior(theta):
            v_sn, m_ej, m_ni, m_csm, rho, r_in, e, x_0, k_o, t_0 = theta
            if (    priors[0][0] < v_sn < priors[1][0]
                and priors[0][1] < m_ej < priors[1][1]
                and priors[0][2] < m_ni < priors[1][2]
                and priors[0][3] < m_csm < priors[1][3]
                and priors[0][4] < rho < priors[1][4]
                and priors[0][5] < r_in < priors[1][5]
                and priors[0][6] < e < priors[1][6]
                and priors[0][7] < x_0 < priors[1][7]
                and priors[0][8] < k_o < priors[1][8]
                and priors[0][9] < t_0 < priors[1][9]
                and m_ni < m_ej
                ):
                return 0.0
            return -np.inf
    
        def lnprob(theta, t, y, yerr):
            lp = lnprior(theta)
            #print(lp)
            if not np.isfinite(lp):
                return -np.inf
            return lp + lnlike(theta, t, y, yerr)
    
    
        ndim = len(out_mean)
        p0 = [out_mean + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    
        import emcee
        from pathos.multiprocessing import ProcessingPool as Pool
         
        #if __name__ == '__main__':  #The horror of windows...
        if True:
            with Pool() as pool:
                #MCMC simulation starts here...
                print('\nStarting MCMC fit...')
                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, y, yerr),
                                                pool=pool,
                                                moves=[
                                                        (emcee.moves.DEMove(), 0.8),
                                                        (emcee.moves.DESnookerMove(), 0.2)])
                print("Running burn-in...\n")
                p0, _, _ = sampler.run_mcmc(p0, 100, progress=True)
                sampler.reset()
            
                print("----------------\nRunning production...\n----------------")
                pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
            
                print('MCMC Analysis complete!')
                
                self.CSM_mcmc = sampler
                
                samples = sampler.flatchain
                for theta in samples[np.random.randint(len(samples), size=100)]:
                    plt.plot(times, mdl.RDCSM_model(times, theta[0], theta[1], theta[2], theta[3], theta[4],
                                              theta[5], theta[6], theta[7], theta[8], theta[9]),
                             color="r", alpha=0.1)
                #print(theta)
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                ax.set_xlabel('Days from Peak')
                ax.set_ylabel('Luminosity')
                
        
                import corner
                labels = ["$v_{ph}$", "$M_{ej}$", "$M_{Ni}$", '$m_{csm}$', '$rho$',
                          '$r_{csm}$', '$e$','$x_{0}$', '$k_{o}$', "$t_{0}$"]
                fig = corner.corner(samples, show_titles=True, plot_datapoints=True,
                                    labels=labels)
                if save_to:
                    print('Saving sampler flatchain...')
                    np.save(save_to + '\\flatchain_RDCSM', sampler.flatchain)
                    print('Saving Cornerplots...')
                    fig.savefig(save_to + '\\Cornerplot_RDCSM.pdf', dpi=1200,
                                bbox_inches='tight')
                    print('Saving Sampleplot...')
                    sfig.savefig(save_to + '\\Sampleplot_RDCSM.pdf', dpi=1200,
                                 bbox_inches='tight')
                    print('saving bol_it object...')
                    #filehandler = open(save_to + '\\Bolfit.obj',"wb")
                    #pickle.dump(self, filehandler)
                    #filehandler.close()
                    print('Saving Complete!')
    
                mcmc = []
                q = []
                for i in range(ndim):
                    mcmc.append(np.percentile(samples[:, i], [16, 50, 84]))
                    q.append(np.diff(mcmc))
                
                return mcmc, q
            
           
    def RDCSM_fit_ns(self, n=7, delt=0, s=0, save_to='',
                   priors = ((0.01, 1e-3, 1e-3, 1e-3, 0.01, 0.01, 1e-3, 1e-3, 1e-4),
                  (12, 20, 20, 20, 50, 20, 1, 1, 1)),
                   owarnings=False, qs=5):
        
        if not owarnings:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
        #----------------------------------------------------------------
        #GLOBAL VARIABLES
        t, y, yerr = self.bol_lc
        t_prior = self.time_prior
        
        params = ['v_sn', 'm_ej', 'm_ni', 'm_csm', 'rho', 'r_in', 'e','x_0', 'k_o', 't_0']
        #-------------------------------------------------------------
        #Set up priors (full)
        try:
            if (not isinstance(t_prior, tuple)) or (not isinstance(t_prior[1], tuple)):
                raise TypeError
        except TypeError:
            print('invalid t_0 prior type given! Please insert as a 2-nested tuple ((t_min,), (t_max,))')
            return
        
        prilow = priors[0] + t_prior[0]
        prihigh = priors[1] + t_prior[1]
        priors = (prilow, prihigh)
        
        #-------------------------------------------------------------
        #Scipy Curvefit
        
        import scipy.optimize as opt
        times = np.linspace(t_prior[0][0], t[-1], 100)
        print('Starting Scipy Curvefit...')
        out_mean, out_var = opt.curve_fit(mdl.RDCSM_model, t, y, bounds=priors)
        print('Scipy Curvefit Success')
    
        sfig = plt.figure(dpi=1200)
        ax = sfig.subplots()
        ax.scatter(t, y) #Data
    
        for i in range(len(params)):
            print(params[i] + ': ', round(out_mean[i], 3))
        
        mod = mdl.RDCSM_model(times, out_mean[0], out_mean[1], out_mean[2], out_mean[3], out_mean[4], out_mean[5], out_mean[6], out_mean[7], out_mean[8], out_mean[9])
        ax.set_ylim(bottom=0)
        ax.plot(times, mod)
        
        
        #----------------------------------------------------------
        #Nested sampling
    
        def prior_transform(u):
        
            x = np.zeros(len(u))    
            #For any definite positive or negative independent priors (numbers in list).
            for i in range(len(u)):
                x[i] = priors[0][i] + (priors[1][i] - priors[0][i]) * u[i]
    
            #for dependent prior, m_ni 
            x[2] = priors[0][2] + (x[1] - priors[0][2]) * u[2]
            return x
    
    
        def lnlike(theta):
            v_sn, m_ej, m_ni, m_csm, rho, r_in, e, x_0, k_o, t_0 = theta
            model = mdl.RDCSM_model(t, v_sn, m_ej, m_ni, m_csm, rho, r_in, e, x_0, k_o, t_0)
            likelihood =  -0.5 * np.sum(np.log(2 * np.pi * yerr ** 2) + ((y - model) / yerr) ** 2)
            #print('preprint')
            if np.isnan(likelihood):
                #print('Oh yay!')
                return -np.inf
            else:
                #print(likelihood)
                return (-0.5 * np.sum(np.log(2 * np.pi * yerr ** 2) + (((y - model)**2) / (yerr**2))))
    
        
        ndim=len(out_mean)
        #------------------------------------------------------
        #Sampling
        import dynesty
        from pathos.multiprocessing import ProcessingPool as Pool
        
        print('Loglike of Scipy: ')
        print(lnlike(out_mean))
            
        print('\nStarting Nested Sampler (Dynamic)...')
        sampler = dynesty.NestedSampler(loglikelihood=lnlike,
                                        prior_transform=prior_transform,
                                        ndim=ndim, nlive=1000,
                                        sample='rwalk', pool=Pool(), queue_size=qs)
        
        dsampler = dynesty.DynamicNestedSampler(loglikelihood=lnlike,
                                                prior_transform=prior_transform,
                                                ndim=ndim,
                                                pool=Pool(), queue_size=qs,
                                                sample='rwalk')
        
        dsampler.run_nested(nlive_init=1000, nlive_batch=1000)
        self.CSM_ns = dsampler
        results = dsampler.results
        print('Nested Sampling Complete!')
        from dynesty import plotting as dyplot
        from dynesty import utils as dyfunc
        
        # Extract sampling results.
        samples = results.samples  # samples
        weights = np.exp(results.logwt - results.logz[-1])  # normalized weights
        
        mean, cov = dyfunc.mean_and_cov(samples, weights)
        
        
        labels = ["$v_{ph}$", "$M_{ej}$", "$M_{Ni}$", '$M_{csm}$', '$rho_{in}$',
                          '$r_{in, csm}$', '$\epsilon$','$x_{0}$', '$k_{o}$', "$t_{0}$"]
        print('Plotting...')
        # Plot a summary of the run.
        rfig, raxes = dyplot.runplot(results)
        # Plot traces and 1-D marginalized posteriors.
        tfig, taxes = dyplot.traceplot(results)
        # Plot the 2-D marginalized posteriors cornerplot.
        cfig, caxes = dyplot.cornerplot(results, color='slategray', show_titles=True,
                                          labels=labels, quantiles=[0.16, 0.5, 0.84])
        #Plot the Curve of the mean.
        ys = mdl.RDCSM_model(times, *mean)
        ax.plot(times, ys, c='navy')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.set_xlabel('Days from Peak')
        ax.set_ylabel('Luminosity')
        
        
        if save_to:
            print('Saving sampler samples and weights...')
            np.save(save_to + '/samples', samples)
            np.save(save_to + '/weights', weights)
            np.save(save_to + '/params', mean)
            print('Saving Cornerplots...')
            cfig.savefig(save_to + '/Cornerplot_RDCSM.pdf', dpi=1200,
                         bbox_inches='tight')
            print('Saving Sampleplot...')
            sfig.savefig(save_to + '/Sampleplot_RDCSM.pdf', dpi=1200,
                         bbox_inches='tight')
            print('saving bol_it object...')
            #filehandler = open(save_to + '\\Bolfit.obj',"wb")
            #pickle.dump(self, filehandler)
            #filehandler.close()
            print('Saving Complete!')
                
        return results, dsampler

            
