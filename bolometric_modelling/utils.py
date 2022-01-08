
import os
import sys
import json
import numpy as np

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
