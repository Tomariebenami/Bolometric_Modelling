import os
import pickle
import numpy as np
from dynesty import utils as dyfunc

#import Param_Functions as pf
import bolometric_modelling.utils as utils
import bolometric_modelling.bolometric_lightcurve as lc


#%% --- SNE Object, General Params ---

class supernova:
    def __init__(self):
        self.res_obj = None
        self.ver = None
        self.name = None
        self.s = 0
        self.d = 0
        
        #empty params
        self.mean = None
        self.error = []
        self.cov = None
        self.quant = None
        self.kl_div_mean = None
        self.kl_div_std = None
        self.resampled_res = None
        self.resampled_mean = None
        self.resampled_cov = None
        self.resampled_quant = None
        self.resampled_error = None
        
    #Use this to load function
    def load(path, name, version):
        
        path = path + '/' + name + '/' + version + '__result.obj'
        
        sne = supernova()
        
        file = open(path, 'rb')
        sne.res_obj = pickle.load(file)
        sne.ver = version
        sne.name = name
        
        sne.s = int(version[-1])
        sne.d = int(version[3])
        
        file.close()
        
        return sne
        
    def calc_mean(self):
        """ Calculates mean of Parameter """
        samples = self.res_obj.samples  # samples
        weights = np.exp(self.res_obj.logwt - self.res_obj.logz[-1])  # normalized weights

        # Compute 16%-84% quantiles.
        self.quant = [dyfunc.quantile(samps, [0.16, 0.5, 0.84], weights=weights)
                     for samps in samples.T]

        # Compute weighted mean and covariance.
        self.mean, self.cov = dyfunc.mean_and_cov(samples, weights)
        
        # Compute STD_Dev
        self.stdev = np.sqrt(np.diag(self.cov))
        
        for p in range(len(self.mean)):
            self.error.append(self.cov[p,p])
        
    def posterior_uncertainty(self, Nrepeat=100):
        """ Calculate Posterior Mean """
        from scipy.stats import gaussian_kde
            
        # Compute KL divergences
        klds = []
        for i in range(Nrepeat):
            kld = dyfunc.kld_error(self.res_obj)#, error='simulate')
            klds.append(kld)
            
        # Compute KLD kernel density estimate
        kl_div = [kld[-1] for kld in klds]
        kde = gaussian_kde(kl_div)
                
        self.kl_div_mean, self.kl_div_std = np.mean(kl_div), np.std(kl_div)        
    
    def simulate_uncertainty(self):
        
        #ln(evidence) error of combined statistical and sampling uncertainties
        lnzs = np.zeros((100, len(self.CSM_ns.logvol)))
        for i in range(100):
            dres_s = dyfunc.simulate_run(self.CSM_ns)
            lnzs[i] = np.interp(-self.CSM_ns.logvol, -dres_s.logvol, dres_s.logz)
        self.lnzerr = np.std(lnzs, axis=0)
    
        return self.lnzerr       
    
    
    def resample(self):
        
        self.resampled_res = dyfunc.resample_run(self.res_obj)
        #get mean and error
        samples = self.resampled_res.samples  # samples
        weights = np.exp(self.resampled_res.logwt - self.resampled_res.logz[-1])

        # Compute 10%-90% quantiles.
        self.resampled_quant = [dyfunc.quantile(samps, [0.16, 0.84], weights=weights)
                     for samps in samples.T]

        # Compute weighted mean and covariance.
        self.resampled_mean, self.resampled_cov = dyfunc.mean_and_cov(samples, weights)

    
    def plot_all(self):
        print('I still need to do this!')
        return



#%% --- CSM SN Object, csm analyis for SNs. ---

class csm(supernova):  
    
    def __init__(self):
        super().__init__()
        
        self.mass = 0 #M * 1.99e33
        self.r_in = 0 #r_in * 10**(14)
        self.rho_in = 0 #rho * 10**(-12)
        
        self.v_w = np.array([100e5,1050e5, 2000e5]) #In cm s^-1
        
        self.q = 0.0 #self.rho_in * (self.r_in ** self.s)
          
        #empty parameters
        self.width = None   #CSM width
        self.r_out = None   # outer CSM Radius
        self.t_forward = None #Forward Shock time
        self.t_backward = None #Backward Shock time
        
        return
    
    def load(path, name, version):    
    
        path = path + '/' + name + '/' + version + '__result.obj'
        
        sne = csm()
        
        #with open(path, 'rb') as file:
        
        file = open(path, 'rb')
        sne.res_obj = pickle.load(file)
        sne.ver = version
        sne.name = name
        
        sne.s = int(version[-1])
        sne.d = int(version[-3])
        
        file.close()
        return sne
    
    def calc_mean(self):
        super().calc_mean()
        
        self.mass = self.mean[3] * 1.99e33
        self.r_in = self.mean[5] * 10**(14)
        self.rho_in = self.mean[4] * 10**(-12)

        self.q = self.rho_in * (self.r_in ** self.s)
        
        return

    
    def calc_width(self, silent=True):
        
        if self.s == 0:
            self.r_out = (self.r_in**3 + (3*self.mass) / (4 *np.pi * self.rho_in))**(1/3)
        elif self.s == 2:
            self.r_out = (self.mass / (4 * np.pi * self.q)) + self.r_in
        else:
            print('Polynomial order invalid!')
            return
        self.width = (self.r_out - self.r_in) * 1e-3
        if not silent:
            print(self.name + ' ', self.width)
        return
    
    
    def calc_mdot(self, verbose=False):
        
        #For s=2 case(wind):
        if self.s == 2:
            self.M_dot = 4 * np.pi * self.v_w * self.q
            self.M_dot = (self.M_dot / 1.989e33) * 3.154e7 # Unit Conversion
            
            self.t_2 = (self.mass / 1.989e33) / self.M_dot #Wind time calc
        
        #For s=0 case(constant shell):
        if self.s == 0:
            self.t_0f = (self.r_out / self.v_w) / 3.154e7
    
            self.t_0i = (self.r_in / self.v_w) / 3.154e7
            
            self.M_dot = (self.mass) / (self.t_0f - self.t_0i)
            self.M_dot = (self.M_dot / 1.989e33)# * 3.154e+7
        return

        

    def shock_times(self):
        #!!! Write(?)
        return


#%% ---------- Bin ---------------
"""
    #use to set up csm object on an exsiting sne
    def extract(sne):
        
        try:
            dtype = int(sne.ver[-1])    
        except:
            print('Ivalid version string given')
        cloud = csm(M=sne.mean[3], r_in=sne.mean[5], rho=sne.mean[4], dtype=dtype)
        cloud.name = sne.name
        cloud.ver = sne.ver
        return cloud
"""



    