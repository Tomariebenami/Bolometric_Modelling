"""
Model options for fitting bolometric lightcurves
Author: Tom A. Ben-Ami
"""

from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import scipy.constants as cst
import warnings 


def RD_model(t, m_ni, m_ej, v_ph, k_o, t_0, owarnings=False):
    
 
    if not owarnings:
        warnings.filterwarnings("ignore", category=RuntimeWarning)
    #-----------------------------------------------------------
    #GLOBAL FUNCTION VARIABLES
    tau_ni = 8.77*3600*24    #decay time of Ni56(s)
    tau_co = 111.3*3600*24    #decay time of Co56(s)
    e_ni = 3.9e10       #Energy Produced Ni valenti
    e_co = 6.78e9      #Energy Produced Co valenti 
    beta = 13.8
    #k_o = 0.19
    k_g = 0.027
    s_m = 1.989e33  #sun mass in grams
    #------------------------------------------------------------
    #Integrand of first integral, A(z)
    def integrand1(z, tau_m, tau_ni):
        y = tau_m / (2 * tau_ni)
        return 2 * z * np.exp(-2 * z * y + z**2)

    #Integrand of second integral, B(z)
    def integrand2(z, tau_m, tau_ni, tau_co):
        y = tau_m / ( 2 * tau_ni)
        s = (tau_m * (tau_co - tau_ni)) / (2 * tau_co * tau_ni)
        return 2 * z * np.exp(-2 * z * y + 2 * z * s + z**2)
    
    #whole function, constructed.
    def photospheric(t, m_ni, m_ej, v_ph, k_o, t_0):
    
        m_ni = m_ni * s_m
        m_ej = m_ej * s_m
        v_ph = v_ph * 1e8
    
        t = (t - t_0)*3600*24
        
        tau_m = np.sqrt(k_o / (beta * cst.c*100)) * np.sqrt(m_ej / v_ph) * (20/3)**(1/4)
        x = t / tau_m
        dt = x / 5000
        tt = np.arange(0, x, dt)    
        arr1 = integrand1(tt, tau_m, tau_ni) #array for integration A
        arr2 = integrand2(tt, tau_m, tau_ni, tau_co)  #array for integration B
        return m_ni*np.exp(-1 * x**2) * ((e_ni - e_co)*np.trapz(arr1, dx=dt, axis=-1) + e_co*np.trapz(arr2, dx=dt, axis=-1)) * (1 - np.exp(-(3*k_g*m_ej) / (4 * 3.14 * v_ph**2 * t**2)))

    
    f = lambda s: photospheric(s, m_ni, m_ej, v_ph, k_o, t_0)
    vec_model = list(map(f, t))
    return vec_model


def RDCSM_model(t, v_sn, m_ej, m_ni, m_csm, rho, r_in, e, x_0, k_o, t_0,
                n=7, delt=0, s=0):
    

    #----------------------------------------------------------------
    #GLOBAL VARIABLES
    
    beta_f = { 0: {6: 1.256, 7: 1.181, 
                   8: 1.154, 9: 1.140,
                   10: 1.131, 12: 1.121,
                   14: 1.116       
                   },  # 1982 (Source 3)
               2: {6: 1.377, 7: 1.299, 
                   8: 1.267, 9: 1.250,
                   10: 1.239, 12: 1.226,
                   14: 1.218       
                   }
              }
    
    beta_r = {0: {6: 0.906, 7: 0.935,
                 8: 0.950, 9: 0.960,
                 10: 0.966, 12: 0.974,
                 14: 0.979       
                 }, # 1982 (Source 3)
              2: {6: 0.958, 7: 0.970,
                 8: 0.976, 9: 0.981,
                 10: 0.984, 12: 0.987,
                 14: 0.990       
                 }
              }
    
    A = {0: {6: 2.4, 7: 1.2,
             8: 0.71, 9: 0.47,
             10: 0.33, 12: 0.19,
             14: 0.12      
             }, # 1982 (Source 3)
         2: {6: 0.62, 7: 27,
             8: 0.15, 9: 0.096,
             10: 0.067, 12: 0.038,
             14: 0.025      
             }
         }
    
    e_co = 6.78e9
    e_ni = 3.9e10 
    t_ni = 8.77*3600*24    #decay time of Ni56(s) ref at asassn14
    t_co = 111.3*3600*24   #decay time of Ni56(s) ref at asassn14
    s_m = 1.989e33  #solar mass in grams
    
    params = ['v_sn', 'm_ej', 'm_ni', 'm_csm', 'rho', 'r_in', 'e','x_0', 'k_o', 't_0']
    #---------------------------------------------------------------
    #MODEL SETUP
    
    #See Eq. 4 in Liu - Checked
    def q(rho, r_in):
        return rho * (r_in ** s)
    
    #See Eq. 3 in Liu - Checked
    def E_sn(m_ej, v_sn, x_0):
        return (((3-delt)*(n-3)) / ((2*(5-delt)*(n-5)))) * m_ej * (x_0 * v_sn)**2
    
    #See Eq. 2 in Liu - Checked
    def g(n, m_ej, v_sn, x_0):
        return (1/(4*np.pi*(n-delt))) * ((2*(5-delt)*(n-5)*E_sn(m_ej, v_sn, x_0))**((n-3)/2)) / (((3-delt)*(n-3)*m_ej)**((n-5)/2))
    
    #t_interaction below Eq.14 Chatzopoulos - Checked
    def t_i(v_sn, r_in):
        return r_in / v_sn   
    
    #Reverse shock timescale (Eq. 16 Chatzopoulos) - Checked
    def t_rs(v_sn, m_ej, r_in, rho, x_0):
        
        p1 = (v_sn) / (beta_r[s][n] * ((A[s][n] * g(n, m_ej, v_sn, x_0) / q(rho, r_in))**(1/(n-s))))
        
        p2 = 1 - ((3-n) * m_ej) / (4 * np.pi * v_sn**(3-n) * g(n, m_ej, v_sn, x_0))
        
        return (p1 * p2**(1/(3-n)))**((n-s) / (s-3))
    
    
    """
    r_out, and r_ph are found by calculating the integrals 16 and 17 in Liu.
    I assume the star radius is much smaller compared to the CSM radius.
    thus, m_th_csm is found using eq. 15.
    
    One should note that in Chatzopoulos eq. 15, m_csm refers to the optically
    thick mass, and not the overall mass... (See Eq. 20 and 21, for explanation)
    Thus, m_csm is calculated using the equations below:
    """
    
    #Eq. 19 Chatzopoulos - Checked
    def r_out(m_csm, r_in, rho):
        if s == 0:
            return (r_in**3 + (3*m_csm) / (4 *np.pi * rho))**(1/3)
        elif s == 2:
            return (m_csm / (4 * np.pi * q(rho, r_in))) + r_in
    
    #Eq. 18 Chatzopoulos - Checked
    def r_ph(m_csm, r_in, rho, k_o):
        if s == 0:
            return r_out(m_csm, r_in, rho) + 2/(3 * k_o * q(rho, r_in))
        elif s == 2:
            return ((1/r_out(m_csm, r_in, rho)) - (2 / (3 * k_o * q(rho, r_in)))) ** (-1)
            
    #Eq. 17 Chatzopoulos - Checked
    def m_th(m_csm, r_in, rho, k_o):
        if s == 0:
            return (4 * np.pi * q(rho, r_in) / 3) * ((r_ph(m_csm, r_in, rho, k_o)**3) - r_in**3)
        elif s == 2:
            return (4 * np.pi * q(rho, r_in)) * (r_ph(m_csm, r_in, rho, k_o) - r_in)
        
    
    #Calculated Via liu Eq. 20 or/and below Eq. 20 Chatzopoulos - Checked
    def t_d(m, r_in, rho, k_o):
        beta = 4 * (np.pi**3) / 9
        return (k_o * m) / (beta * cst.c*100 * r_ph(m, r_in, rho, k_o))
    
    #Forward shock timescale (Eq. 14 Liu) - Checked
    def t_fs(v_sn, m_ej, m_csm, r_in, rho, k_o, x_0):
    
        inner = ((3-s) * q(rho, r_in)**((3-n)/(n - s)) * (A[s][n] * g(n, m_ej, v_sn, x_0))**((s-3)/(n-s))) / (4 * np.pi * beta_f[s][n]**(3-s))
        
        return abs(inner)**((n-s) / ((n-3)*(3-s))) * m_th(m_csm, r_in, rho, k_o) **((n-s) / ((n-3)*(3-s)))
    
    
    #First integrand of Chatzopoulos Eq. 21. CSM interaction - Checked
    def integrand1(t, m_ej, m_csm, v_sn, rho, r_in, k_o, x_0, e):
    
        alpha_i = (2*n + 6*s - n * s - 15) / (n - s)    
    
        p1 = (2*np.pi / (n-s)**3) * g(n, m_ej, v_sn, x_0)**((5-s)/(n-s)) * q(rho, r_in)**((n-5)/(n-s)) * ((n-3)**2) * (n-5) * beta_f[s][n]**(5-s)
        
        p2 = A[s][n]**((5-s)/(n-s)) * ((t + t_i(v_sn, r_in))**alpha_i) * np.heaviside(t_fs(v_sn, m_ej, m_csm, r_in, rho, k_o, x_0) - t, 0.5)
    
        p3 = 2*np.pi * ((A[s][n] * g(n, m_ej, v_sn, x_0))/ q(rho, r_in))**((5-n)/(n-s)) * beta_r[s][n]**(5-n) * g(n, m_ej, v_sn, x_0) * ((3-s)/(n-s))**3
    
        p4 = ((t + t_i(v_sn, r_in))**alpha_i) * np.heaviside(t_rs(v_sn, m_ej, r_in, rho, x_0) - t, 0.5)
    
        retarr = []
        for i in range(len((p1 * p2 + p3 * p4))):
            if (p1 * p2 + p3 * p4)[i] == 0:
                retarr.append(0.0)
            else:
                retarr.append(e * np.exp(t[i]/t_d(m_th(m_csm, r_in, rho, k_o), r_in, rho, k_o)) * (p1 * p2 + p3 * p4)[i])
        return retarr
    
    
    #2nd Integral of Chatzopoulos Eq. 21 - Ni-Co Decay - Checked
    def integrand2(z, m_ni, m_ej, m_csm, r_in, rho, k_o):
    
        p1 = (e_ni - e_co) * np.exp(-z/t_ni)
        
        p2 = e_co * (np.exp(-z/t_co))
    
        return np.exp(z/t_d(m_th(m_csm, r_in, rho, k_o) + m_ej, r_in, rho, k_o)) * m_ni * (p1 + p2)
    
    #Chatzopoulos Eq. 21 - Checked
    def model(t, v_sn, m_ej, m_ni, m_csm, rho, r_in, e, x_0, k_o, t_0):
        
        m_ej = m_ej * s_m
        m_csm = m_csm * s_m
        m_ni = m_ni * s_m
        v_sn = v_sn * 1e8
        r_in = r_in * 1e14
        rho = rho * 1e-12
        
        t_d1 = t_d(m_th(m_csm, r_in, rho, k_o), r_in, rho, k_o)
        t_d2 = t_d(m_th(m_csm, r_in, rho, k_o) + m_ej, r_in, rho, k_o)

        t = (t - t_0)*3600*24
        dt = t/1000
        #print('t: ', t)
        tt = np.arange(0, t, dt)
        arr1 = integrand1(tt, m_ej, m_csm, v_sn, rho, r_in, k_o, x_0, e)
        arr2 = integrand2(tt, m_ni, m_ej, m_csm, r_in, rho, k_o)
        return (np.exp(-t/t_d1) / t_d1) * np.trapz(arr1, dx=dt, axis=-1) + (np.exp(-t/t_d2) / t_d2) * np.trapz(arr2, dx=dt, axis=-1)
 
    #Manual vectorization of model (np.traps does not like arrays...)
            
    f = lambda s: model(s, v_sn, m_ej, m_ni, m_csm, rho, r_in, e, x_0, k_o, t_0)
    vec_model = list(map(f, t))
    return vec_model
        