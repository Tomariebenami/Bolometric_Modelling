
import os
import json
import numpy as np
import pickle

from astroquery.irsa_dust import IrsaDust
from bolometric_modelling.Bolometric_Functions import bol_fit

def find_extinction(ra, dec):
    
    ra = ra.replace(':', ' ')
    dec = dec.replace(':', ' ')
    coords = ra + ' ' + dec    
    t = IrsaDust.get_extinction_table(coords)
    types = ['CTIO', 'SDSS', 'UKIR']
    filt = dict()
    for i in t:
        if i['Filter_name'][0:4] in types:
            filt[i['Filter_name'][-1]] = i['A_SandF']     

    return filt



def blc_fit():
    #Explanation of System and Data Choices!
    print("""
          ▄▄▄▄·       ▄▄▌  • ▌ ▄ ·.       ·▄▄▄▄  
          ▐█ ▀█▪▪     ██•  ·██ ▐███▪▪     ██▪ ██ 
          ▐█▀▀█▄ ▄█▀▄ ██▪  ▐█ ▌▐▌▐█· ▄█▀▄ ▐█· ▐█▌
          ██▄▪▐█▐█▌.▐▌▐█▌  ██ ██▌▐█▌▐█▌.▐▌██. ██ 
          ·▀▀▀▀  ▀█▄▀▪.▀   ▀▀  █▪▀▀▀ ▀█▄▀▪▀▀▀▀▀• 
          """)
    print('Welcome to the bolometric modelling shell control module!\n')
    print('This function can accept one of multiple bolometric lightcurves\n')
    print('This function accepts and already computed bolometric lightcurve,')
    print('and not the original spectral lightcurves\n')
    
    print("""
In order to use the multiple event fit option, please supply a
directory including the bolometric lighturve data, with the file
names as th event names.
          """)
          
    #Define whether one event or multiple events
    while True:
        amount = input('Fit a single event or multiple events? (single/multi):').lower()
        if amount == 'single' or amount == 'multi':
            break
        else:
            print('Please enter a valid answer...(single/multi)')
            
    #Define Path for event or multievent
    if amount == 'single':
        while True:
            rpath = input('Enter filepath:')
            if os.path.isfile(rpath):
                break
            else:
                print('Please enter a valid path')

    elif amount == 'multi':
        while True:
            rpath = input('Enter directory path:')
            if os.path.exists(rpath):
                break
            else:
                print('Please enter a valid path')
                
    #Map out events and get data
    if amount == 'single':
        names = input('Name of the Event to be analysed:')
    
    elif amount == 'multi':
        files = os.listdir(rpath)
        names = []
        for f in files:
            sn, _ = os.path.splitext(f)
            names.append(sn)
        
        names = ["""names of events"""]
    #Ask for writepath
    
    while True:
        wpath = input('Result write directory:')
        if os.path.exists(wpath):
            break
        else:
            try:
                os.makedirs(wpath)
                break
            except:
                print('Invalid Directory Path given! Please supply a valid path')
    #create file system in writepath if needed
    for s in names:
        f_path = os.path.join(wpath, s)
        try:
            os.mkdir(wpath)
        except FileExistsError():
            pass
        
    #Decide on the model
    while True:
        model = input('Choose model to fit to (RD/RDCSM):').upper()
        if model == 'RD' or model == 'RDCSM':
            break
        else:
            print('Please choose a valid model')
    if model == 'RDCSM':
        while True:
            try:
                cores = int(input('Amount of cores used for Parallelisation(0 = Default)'))
                if cores == 0:
                    #Get maximun number of cores from machine
                    break
                elif """height cores""": # !!!
                    print('Warning - Number of threads exceeds number of cores')
                    print('on this computer, this may slow down the process')
                    break
                else:
                    break
            except:
                print('Please insert a valid integer')
        while True:
            fittype = input('Please choose CSM fitting algorithm(MCMC/DNS):').upper()
            if fittype == 'MCMC' or fittype == 'DNS':
                break
            else:
                print('Please insert a valid prefrence')
        if fittype == 'DNS':
            print('r')
        # !!! Set up n, delta, s options
        
    #Details for configurations
    print('Certain values required for the fit may be inserted via a json file,')
    print('or manually. Please insert preference below (json/manual)')
    while True:
        pref = input('').lower()
        if pref == 'json' or pref == 'manual':
            break
        else:
            print('Please supply a valid argument')
    if pref == 'json':
        while True:
            config_p = input('JSON file path:')
            if os.path.isfile(config_p):
                try:
                    configfile = open('positions.yaml')
                    snes_dat = json.load(configfile)
                    break
                except:
                    pass
            print('Please enter a valid path')
                
    if pref == 'manual':
        while True:
            print('Too bad! need to make this still...')
            break
    
    #Read data
    
    #Start fitting!
   # print('Starting DNS Fitting - n=7, delta=',delta, ', s=',ss,'...') 
    
    
    #-------------------------------------------------------------
    #Setup time priors
    tpriors = dict()
    for s in names:
        # !!! Add end prior from data...
        tpriors[s] = ((eval(snes_dat[s]['nondet']) - 2,), 0) 


    # !!! move to choice section
    if fittype == 'DNS':
        delta = int(input('Please select delta value (0, 1, 2):'))
        ss = int(input('Please select s value (0, 2):'))
      
    """
    #------------------------------------------------------------
    #RD Analysis - MCMC

    if amount == 'multi' and model == 'RD':
        mcmc_RD = dict()
        q_RD = dict()
        for s in names:
            sn_path = s_path + '/' + s
            if len(datas[s][0]) >= 10:
                print('\nNow Fitting: ', s)
                #if __name__ == '__main__':
                mcmc_RD[s], q_RD[s] = bf.RD_fit(datas[s][0], datas[s][1], datas[s][2],
                                                tpriors[s], save_to=sn_path)

    #--------------------------------------------------------------
    #RD+CSM Analysis - MCMC

    if amount == 'multi' and model == 'RDCSM' and fittype == 'MCMC':
        mcmc_CSM = dict()
        q_CSM = dict()
        for s in names:
            sn_path = s_path + '/' + s
            if len(datas[s][0]) >= 10:
                #if __name__ == '__main__':
                mcmc_CSM[s], q_CSM[s] = bf.RD_CSM_fit(datas[s][0], datas[s][1], datas[s][2],
                                                      tpriors[s], save_to=sn_path)
    #--------------------------------------------------------------
    #Single SNe - RD - MCMC
    if amount == 'single' and model == 'RD':
        s = SNE
        sn_path = s_path + '/' + SNE
        #if __name__ == '__main__':
        mcmc_RD, q_RD = bf.RD_fit(datas[s][0], datas[s][1], datas[s][2],
                                                tpriors[s], save_to=sn_path, niter=100)

#--------------------------------------------------------------
#Single SNe - CSM - MCMC

    if amount == 'single' and model == 'RDCSM' and fittype == 'MCMC':
        s = SNE
        sn_path = s_path + '/' + SNE
        #if __name__ == '__main__':
        mcmc_CSM, q_CSM = bf.RD_CSM_fit(datas[s][0], datas[s][1], datas[s][2],
                                        tpriors[s], n=7, delt=0, save_to=sn_path, niter=2500, nwalkers=150,
                                        priors = ((1, 0.1, 1e-4, 1e-4, 0.01, 0.01, 0.1, 0.1, 1e-2),
                                                  (12, 5, 1, 1, 1, 20, 1, 1, 1)))

    #-------------------------------------------------------------
    #Single SNe - CSM - DNS
    if amount == 'single' and model == 'RDCSM' and fittype == 'DNS':
        s = SNE
        sn_path = spath_2
        print(sn_path)
        print(s_path2)
        print('\nNow Fitting: ', s)
        #if __name__ == '__main__':
        res2 = bf.RD_CSM_NS(datas[s][0],  datas[s][1], datas[s][2],
                            tpriors[s], n=7, delt=delta, s=ss, save_to=sn_path, qs=qs,
                            priors = ((1.0, 0.1, 1e-4, 1e-4, 0.01, 0.01, 0.1, 0.1, 1e-2),
                                      (12.0, 5.0, 1.0, 1.6, 1.0, 20.0, 1.0, 1.0, 1.0)))

    #---------------------------------------------------------------
    #Multiple SNe - CSM - DNS
    if amount == 'multi' and model == 'RDCSM' and fittype == 'DNS':
        results = dict()
        samplers = dict()
        for s in names:
            sn_path = s_path2 + '/' + s
            if len(datas[s][0]) >= 10 and (s not in []):
                print('\nNow Fitting: ', s)
                #if __name__ == '__main__':
                results[s], samplers[s] = bf.RD_CSM_NS(datas[s][0],  datas[s][1], datas[s][2],
                            tpriors[s], n=7, delt=delta, s=ss, save_to=sn_path, qs=qs,
                            priors = ((1.0, 0.1, 1e-4, 1e-4, 0.01, 0.01, 0.1, 0.1, 1e-2),
                                      (12.0, 5.0, 1.0, 1.6, 1.0, 20.0, 1.0, 1.0, 1.0)))
                
    """           
    #TODO: Save data using pickle