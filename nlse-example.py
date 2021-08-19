# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:15:52 2021

@author: Firulais
"""
import numpy as np
import matplotlib.pylab as plt
from optocommon import FFT, sechPulse, c, plotevol, dB
from optosimnlse import GNLSE

#####################################################################
# Example: Figure 3 of Dudley, Genty & Coen, 
#          Reviews of Modern Physics 78, 2006.
#####################################################################

#SAMPLING AND TIME SPAN
N    = 2**13
Tmax = 6.25 # ps

#CENTRAL WAVELENGTH
lambda0 = 835
omega0  = 2*(np.pi)*c/lambda0

#DISPERSION
#betas in ps^m/m
betas = [-11.8300e-03, 8.1038e-05, -9.5205e-08, 2.0737e-10, 
          -5.3943e-13, 1.3486e-15, -2.5495e-18, 3.0524e-21, -1.7140e-24]

#NONLINEARITY
gamma0   = 0.11 #1/(W m)
satgamma = 3e6
#gamma1   = gamma0/omega0 #Self-steepening parameter s = 1
gamma1   = gamma0*0.56e-3 #Shock time = 0.56 fs

#FIBER LENGTH
distance = 0.15 # m
Nz       = 201
Z        = np.linspace(0,distance,Nz)

#PULSE PARAMETERS
TFWHM   = 50e-3     #ps
P0      = 10e3      #W


#%% GNLSE

fibra1  = GNLSE(lambda0=lambda0,N=N,Tmax=Tmax,
                betas=betas,gammas=[gamma0,gamma1],satgamma=satgamma)
AT0     = sechPulse(P0,fibra1.T,TFWHM=TFWHM)
AW0     = FFT(AT0) #initial field

fibra1.rtol = 1e-3
fibra1.atol = 1e-6
fibra1.integrateIP(AW0, Z, progress=True)
        
plt.figure()
plotevol(dB(fibra1.A/np.sqrt(P0)),Z,fibra1.W/(2*np.pi),fftshft=True,vmin=-5)
