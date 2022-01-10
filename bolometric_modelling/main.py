
import os
import sys
import json
import numpy as np
import argparse
#import pickle

from bolometric_modelling.Bolometric_Functions import bol_fit

def prompt(ptype, text, error='Invalid input', options = [], limit=[]):
    if ptype == 'choice':
        print(text, options)
        while True:
            c = input()
            if c not in options:
                print(error, options)
            else:
                return c
    if ptype == 'int':
        print(text, limit)
        while True:
            try:
                i = int(input())
                return i
            except:
                print(error)
    if ptype == 'float':
        print(text, limit)
        while True:
            try:
                f = float(input())
                return f
            except:
                print(error)
    if ptype == 'str':
            print(text)
            while True:
                try:
                    s = str(input())
                    return s
                except:
                    print(error)
    if ptype == 'file':
        print(text)
        while True:
            f = input()
            if os.path.isfile(f):
                return f
            else:
                print(error)
    if ptype == 'dir':
        print(text)
        while True:
            d = input()
            if os.path.exists(d):
                return d
            else:
                print(error)
    print('Unknown ptype!')
    return

def set_parser():
    parser = argparse.ArgumentParser(prog='bolmod',
                                    description='Fit bolometric lightcurves')  
    
    
    #Define whether one event or multiple events
    g1 = parser.add_mutually_exclusive_group() #Single or Multi
    g1.add_argument(
                    '--single',
                    '-s',
                    dest='amount',
                    #nargs='+',
                    action='store_const',
                    const='single',
                    help='Fit a model to a single event')
    
    g1.add_argument(
                    '--multi',
                    '-m',
                    dest='amount',
                    #nargs='+',
                    action='store_const',
                    const='multi',
                    help='Fit a model to multiple events.')
    
    #ask for event name
    parser.add_argument(
                    '--Name',
                    '-N',
                    dest='name',
                    default='',
                    help='Name(s) of SNe to be fitted.')                  
    
    # Define Path for event or multievent
    parser.add_argument(
                    '--path',
                    '-p',
                    dest='path',
                    default='',
                    #nargs='+',
                    help='Bolometric lightucurve data path.')  
             
    #Ask for writepath
    
    parser.add_argument(
                    '--write',
                    '-w',
                    dest='wpath',
                    default='',
                    #nargs='+',
                    help='The path for writing results.')
    
    #Decide on the model
    
    g2 = parser.add_mutually_exclusive_group()  #Model Group
    
    g2.add_argument(
                    '--RD',
                    '-rd',
                    dest='model',
                    #nargs='+',
                    action='store_const',
                    const='RD',
                    help='Fit a radioactive decay model.') 
    
    g2.add_argument(
                    '--RDCSM',
                    '-rdcsm',
                    dest='model',
                    #nargs='+',
                    action='store_const',
                    const='RDCSM',
                    help='Fit a radioactive decay + circumstellar interaction model.')
    
    #Set up n, delta, s options
    
    parser.add_argument(
                    '--poly_n',
                    '-n',
                    dest='n',
                    default=7,
                    type=int,
                    help='degree of ejecta external polynomial.')
    
    parser.add_argument(
                    '--delta',
                    '-d',
                    dest='delta',
                    default=0,
                    type=int,
                    help='degree of ejecta internal polynomial.')
    
    parser.add_argument(
                    '--poly_s',
                    '-ss',
                    dest='s',
                    default=0,
                    type=int,
                    help='degree of CSM density polynomial.\n s=0 - shell, s=2 - wind')
 
    #Cores
    parser.add_argument(
                    '--cores',
                    '-c',
                    dest='cpu',
                    default=int(os.cpu_count() / 2),
                    type=int,
                    help='Number of cores to utilise in NS fit. (Default: half)')
    
    #Choose fit algorithm
    g3 = parser.add_mutually_exclusive_group()  #Model Group
    
    g3.add_argument(
                    '--MCMC',
                    '-MC',
                    dest='algorithm',
                    action='store_const',
                    const='MCMC',
                    help='Use an MCMC sampler to fit the model.')
    
    g3.add_argument(
                    '--DNS',
                    '-D',
                    dest='algorithm',
                    action='store_const',
                    const='DNS',
                    help='Use a Dynamic Nested Sampler to fit the model.')
       
    #json or manual
    g4 = parser.add_mutually_exclusive_group()  #config Group
    
    g4.add_argument(
                    '--json',
                    '-J',
                    dest='config_set',
                    action='store_const',
                    const='json',
                    help='use a json file to config required values')
    
    g4.add_argument(
                    '--manual',
                    '-M',
                    dest='config_set',
                    action='store_const',
                    const='manual',
                    help = 'Manually configure required fit values')
   
    #Details for configurations
    
    parser.add_argument(
                    '--config',
                    #'-r',
                    dest='config',
                    default=0.0,
                    type=float,
                    help='Configuration json file path.')
    
    parser.add_argument(
                    '--redshift',
                    #'-r',
                    dest='r',
                    default=0.0,
                    type=float,
                    help='Event redshift')
    
    #last non det
    
    parser.add_argument(
                    '--nondet',
                    #'-nd',
                    dest='nd',
                    default=0.0,
                    type=float,
                    help='Latest non-detection limit')
    
    return parser

def main():
#if True:    
    my_parser = set_parser()
    args = my_parser.parse_args()

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
    print('This function accepts already computed bolometric lightcurve,')
    print('and not the original spectral lightcurves\n')
    
    #my_parser.print_help()
    
    print("""
In order to use the multiple event fit option, please supply a
directory including the bolometric lighturve data, with the file
names as th event names.
          """)
         
    #Define whether one event or multiple events
    if args.amount == None:
        args.amount = prompt('choice',
                             text = 'Fit a single event or multiple events?',
                             error = 'Please choose a valid option',
                             options =['single', 'multi'])
            
    #Define Path for event or multievent
    if args.path == None or not os.path.exists(args.path):
        if args.amount == 'single':
            args.path = prompt('file',
                               text = 'Choose Data filepath',
                               error = 'Please choose a valid path')
        
        elif args.amount == 'multi':
            args.path = prompt('dir',
                               text = 'Choose Data directory path',
                               error = 'Please choose a valid path')
    #Map out events and get data
    if args.amount == 'single' and args.name == '':
        args.name = [prompt('str',
                           text = 'Please eneter event name',
                           error = 'Invalid name given')]
    
    elif args.amount == 'multi':
        files = os.listdir(args.path)
        args.name = []
        for f in files:
            sn, _ = os.path.splitext(f)
            args.name.append(sn)
        
    #Ask for writepath
    if args.wpath == None or not os.path.exists(args.wpath):
        args.wpath = prompt('dir',
                            text = 'Choose path to save results',
                            error = 'Invlid path given')

    #Create file system in writepath if needed
    if args.amount == 'multi':
        for s in args.name:
            f_path = os.path.join(args.wpath, s)
            try:
                os.mkdir(f_path)
            #except FileExistsError():
            #    pass
            except:
                pass
    elif args.amount == 'single':
        f_path = os.path.join(args.wpath, args.name[0])
        try:
            os.mkdir(f_path)
        except:
            pass
    #Decide on the model
    if args.model == None:
        args.model = prompt('choice',
                            text = 'Please choose which model to fit to',
                            error = 'Please choose a valid model',
                            options = ['RD', 'RDCSM'])
        
    if args.model == 'RDCSM':
        if args.cpu == int(os.cpu_count() / 2):
            args.cpu = prompt('int', limit = [0, os.cpu_count()],
                              text = 'Choose amount of cores to use, 0 - default.',
                              error = 'Invalid value given')
            if args.cpu == 0:
                args.cpu = int(os.cpu_count() / 2)
        if args.algorithm == None:
            args.algorithm = prompt('choice',
                                    text = 'Please choose fitting algorithm',
                                    error = 'Invalid algorithm option given',
                                    options = ['MCMC', 'DNS'])
        
    #Details for configurations
    print('Certain values required for the fit may be inserted via a json file')
    print('or manually. Please insert preference below (json/manual)')
    if args.config_set == None:
        args.config_set = prompt('choice',
                                 text = 'Please choose configuration setup prference',
                                 error = 'Invalid setup chosen',
                                 options = ['json', 'manual'])
    if args.config_set == 'json':
        args.config = prompt('file', 
                             text = 'Please eneter cofig file path',
                             error = 'Invalid filepath given')
    
        configfile = open(args.config)
        snes_dat = json.load(configfile)
                
    if args.config_set == 'manual':
        print("""
This method only works for a single fit. 
Multiple events are passed via the json format.
             """)
    #redshift
        args.r = dict()
        args.r[args.name[0]] = prompt('float',
                                   text = 'Insert event redshift',
                                   error = 'Invalid redshift inserted')
    #last non det
        args.nd = dict()
        args.nd[args.name[0]] = prompt('float',
                                    text = 'Insert event non-det limit',
                                    error = 'Invalid nondet limit inserted')

    #Read data
    if args.amount == 'multi':
        paths = [args.path + '/' + f for f in files]
    elif args.amount == 'single':
        paths = [args.path]
    
    #setup objects
    events = dict()
    for s in range(len(paths)):
        events[args.name[s]] = bol_fit()
        _, _, _ = events[args.name[s]].prep_data_model(path=paths[s])

    #Start fitting!
    #-------------------------------------------------------------
    #Setup time priors
    for s in args.name:
        events[s].time_prior = ((args.nd[s] - 2,), (0,)) 

      
    
    #------------------------------------------------------------
    #RD Analysis - MCMC

    if args.amount == 'multi' and args.model == 'RD':
        mcmc_RD = dict()
        q_RD = dict()
        for s in args.name:
            sn_path = args.wpath + '/' + s
            print('\nNow Fitting: ', s)
            #if __name__ == '__main__':
            mcmc_RD[s], q_RD[s] = events[s].RD_fit_mcmc(save_to=sn_path)

    #--------------------------------------------------------------
    #RD+CSM Analysis - MCMC

    if args.amount == 'multi' and args.model == 'RDCSM' and args.algorithm == 'MCMC':
        mcmc_CSM = dict()
        q_CSM = dict()
        for s in args.name:
            sn_path = args.wpath + '/' + s
            #if __name__ == '__main__':
            mcmc_CSM[s], q_CSM[s] = events[s].RDCSM_fit_mcmc(save_to=sn_path)
    #--------------------------------------------------------------
    #Single SNe - RD - MCMC
    if args.amount == 'single' and args.model == 'RD':
        sn_path = args.wpath + '/' + args.name[0]
        #if __name__ == '__main__':
        mcmc_RD, q_RD = events[args.name[0]].RD_fit_mcmc(save_to=sn_path, niter=500)

    #--------------------------------------------------------------
    #Single SNe - CSM - MCMC

    if args.amount == 'single' and args.model == 'RDCSM' and args.algorithm == 'MCMC':
        sn_path = args.wpath + '/' + args.name[0]
        #if __name__ == '__main__':
        mcmc_CSM, q_CSM = events[args.name[0]].RDCSM_fit_mcmc(n=args.n, delt=args.delta,
                                                       save_to=sn_path,
                                                       niter=5000, nwalkers=150,
                                                       priors = ((1, 0.1, 1e-4, 1e-4, 0.01, 0.01, 0.1, 0.1, 1e-2),
                                                                 (12, 5, 1, 1, 1, 20, 1, 1, 1)))

    #-------------------------------------------------------------
    #Single SNe - CSM - DNS
    if args.amount == 'single' and args.model == 'RDCSM' and args.algorithm == 'DNS':
        sn_path = args.wpath + '/' + args.name[0]
        print('\nNow Fitting: ', s)
        #if __name__ == '__main__':
        res2 = events[args.name[0]].RDCSM_fit_ns(n=args.n, delt=args.delta, s = args.s,
                                           save_to=sn_path, qs = args.cpu,
                                           priors = ((1.0, 0.1, 1e-4, 1e-4, 0.01, 0.01, 0.1, 0.1, 1e-2),
                                                     (12.0, 5.0, 1.0, 1.6, 1.0, 20.0, 1.0, 1.0, 1.0)))

    #---------------------------------------------------------------
    #Multiple SNe - CSM - DNS
    if args.amount == 'multi' and args.model == 'RDCSM' and args.algorithm == 'DNS':
        for s in args.name:
            sn_path = args.wpath2 + '/' + s
            print('\nNow Fitting: ', s)
            #if __name__ == '__main__':
            res2 = events[s].RDCSM_fit_ns(n=args.n, delt=args.delta, s = args.s,
                                                          save_to=sn_path, qs = args.cpu,
                                                          priors = ((1.0, 0.1, 1e-4, 1e-4, 0.01, 0.01, 0.1, 0.1, 1e-2),
                                                                    (12.0, 5.0, 1.0, 1.6, 1.0, 20.0, 1.0, 1.0, 1.0)))
                
             
