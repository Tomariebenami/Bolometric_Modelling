
import os
import sys
import json
import numpy as np
import math as m
from astroquery.irsa_dust import IrsaDust

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


def convert_to_erg_s(blackbody_data):
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


def convert_to_K(blackbody_data):
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


def convert_to_cm(blackbody_data):
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

def calc_bolo(blackbody_data):
    blackbody_data['Lum']= np.array(4 * (m.pi) * ((blackbody_data['radius'])**2) * (5.6703*10**(-5)) * ((blackbody_data['temp'])**4))
    blackbody_data['dLum0']= np.array((((8 * (m.pi) * (blackbody_data['radius']) * (5.6703*10**(-5)) * ((blackbody_data['temp'])**4)*(blackbody_data['dradius0']))**2)+((16 * (m.pi) * ((blackbody_data['radius'])**2) * (5.6703*10**(-5)) * ((blackbody_data['temp'])**3)*(blackbody_data['dtemp0']))**2))**0.5)
    blackbody_data['dLum1']= np.array((((8 * (m.pi) * (blackbody_data['radius']) * (5.6703*10**(-5)) * ((blackbody_data['temp'])**4)*(blackbody_data['dradius1']))**2)+((16 * (m.pi) * ((blackbody_data['radius'])**2) * (5.6703*10**(-5)) * ((blackbody_data['temp'])**3)*(blackbody_data['dtemp1']))**2))**0.5)
    return blackbody_data
