# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:53:33 2021

@author: Firulais
"""
import numpy as np
from optocommon import FFT, sechPulse, c, plotevol, dB
from optosimnlse import GNLSE
from optosimpcnlse import pcGNLSE
import matplotlib.pyplot as plt

#DATOS DE MUESTREO
N    = 2**17
Tmax = 95 # ps

#FRECUENCIA CENTRAL
lambda0 = 850
omega0  = 2*(np.pi)*c/lambda0

#DISPERSIÓN Y ZDW
zdw          = 780
Omegazd      = 2*np.pi*c/zdw-omega0
beta3        = 0.12e-3 #0.55/1.2 # ps^3/m 1/Omegaz#
beta2        = -beta3*Omegazd
print(f'beta2: {beta2*1e3: .5f} ps^2/km')
print(f'ZDW: {zdw:.0f} nm')
print(f'WZD: {Omegazd:.3f} Trad/s')


#NO-LINEALIDAD Y ZNW
gamma0   = 0.018353
satgamma = 3e6
znw      = 860
Omegaz   = 2*np.pi*c/znw-omega0
gamma1   = -gamma0/Omegaz

#SOLITÓN
TFWHM   = 200e-3 #ps
P0      = 6000
#LONGITUD DE LA FIBRA
distance = 0.05 # m
Nz       = 201
Z        = np.linspace(0,distance,Nz)

#%% GNLSE

fibra1  = GNLSE(lambda0=lambda0,N=N,Tmax=Tmax,betas=[beta2,beta3],gammas=[gamma0,gamma1],satgamma=satgamma)
AT0     = sechPulse(P0,fibra1.T,TFWHM=TFWHM)
AW0     = FFT(AT0) #initial field


fibra1.integrateIP(AW0, Z, progress=True)
        
plt.figure()
plotevol(dB(fibra1.A/np.sqrt(P0)),Z,fibra1.W/(2*np.pi),fftshft=True,vmin=-5)

#%% pcGNLSE

fibra2  = pcGNLSE(lambda0=lambda0,N=N,Tmax=Tmax,betas=[beta2,beta3],gammas=[gamma0,gamma1],satgamma=satgamma)
AT0     = sechPulse(P0,fibra2.T,TFWHM=TFWHM)
AW0     = FFT(AT0) #initial field


fibra2.integrateIP(AW0, Z, progress=True)
        
plt.figure()
plotevol(dB(fibra2.A/np.sqrt(P0)),Z,fibra2.W/(2*np.pi),fftshft=True,vmin=-5)
