"""
Bolometric functions plotting interface. 
Author: Tom A. Ben-Ami
"""

from matplotlib import pyplot as plt
import numpy as np
import math as m

def plot_temp(handle, data, label, c='k', marker='o', alpha=0.6, log=False, xlabel=True):
    
    if len(data) == 3:
        print('scipy!')
        error = data[2]
    elif len(data) == 4:
        print('mcmc!')
        error = [data[2], data[3]]
    else:
        print('Data length Error')
        return
    #convert to log scale
    if log:    
        for i in range(len(error)):
            error[i] = (1/m.log(10))*(error[i]/data[1])
        y_dat = np.log10(data[1])
    else:
        y_dat = data[1]
        
    handle.errorbar(x=data[0],
                y=y_dat,
                yerr=error,
                label=label,
                marker=marker,
                markersize=6,
                fillstyle='full',
                lw=0.8,
                color=c,
                alpha=alpha)

    handle.set_ylabel('Temperature(K)')
    if xlabel:
        handle.set_xlabel('Rest-frame days relative to peak')
    handle.set_title('Temperature', fontsize=14)


def plot_radius(handle, data, label, c='k', marker='o', alpha=0.6, log=False,
                xlabel=True):
    
    if len(data) == 3:
        error = data[2]
    elif len(data) == 4:
        error = [data[2], data[3]]
    else:
        print('Data length Error')
        return
    if log:    
        for i in range(len(error)):
            error[i] = (1/m.log(10))*(error[i]/data[1])
        y_dat = np.log10(data[1])
    else:
        y_dat = data[1]
    
    handle.errorbar(x=data[0],
                y=y_dat,
                yerr=error,
                label=label,
                marker=marker,
                markersize=6,
                fillstyle='full',
                lw = 0.8,
                color=c, 
                alpha=alpha)

    handle.set_ylabel('Radius(cm)')
    if xlabel:
        handle.set_xlabel('Rest-frame days relative to peak')
    handle.set_title('Radius', fontsize=14)
    

def plot_lum(handle, data, label, c='k', marker='o', alpha=0.6, log=False,
             xlabel=True):
    
    if len(data) == 3:
        error = data[2]
    elif len(data) == 4:
        error = [data[2], data[3]]
    else:
        print('Data length Error')
        return
    if log:    
        for i in range(len(error)):
            error[i] = (1/m.log(10))*(error[i]/data[1])
        y_dat = np.log10(data[1])
    else:
        y_dat = data[1]
    
    handle.errorbar(x=data[0],
                y=y_dat,
                yerr=error,
                label=label,
                marker=marker,
                markersize=6,
                fillstyle='full',
                lw = 0.8,
                color=c, 
                alpha=alpha)

    handle.set_ylabel('Bolometric Luminosity (erg/s)')
    if xlabel:
        handle.set_xlabel('Rest-frame days relative to peak')
    handle.set_title('Bolometric Luminosity', fontsize=14)