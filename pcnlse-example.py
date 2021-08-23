# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:15:52 2021

@author: Firulais
"""
import numpy as np
import matplotlib.pylab as plt
from optocommon import FFT, iFFT, FFTSHIFT, fundamentalSoliton, PhysConst, plotevol, dB
from optosimnlse import GNLSE
from optosimpcnlse import pcGNLSE

#####################################################################
# Example: Figures 3 and 4 of Bonetti et al., 
#          JOSA B 37(2), 2020.
#####################################################################

#Physical Constants
cnst = PhysConst()

#SAMPLING AND TIME SPAN
N    = 2**13
Tmax = 6 # ps

#CENTRAL WAVELENGTH
lambda0 = 835
omega0  = 2*(np.pi)*cnst.cwave/lambda0

#DISPERSION
#betas in ps^m/m
betas = [-38.3e-03,+25.0e-05]

# #RAMAN
# fR      = 0.18
# tau1    = 0.0155 #ps
# tau2    = 0.2305 #ps

#NONLINEARITY
gamma0   = 0.11 #1/(W m)
satgamma = 3e6
gamma1   = 0#gamma0/omega0 #Self-steepening parameter s = 1
gamma1   = -2.25e-4

#FIBER LENGTH
distance = 0.20 # m
Nz       = 201
Z        = np.linspace(0,distance,Nz)

#PULSE PARAMETERS
T0      = 10e-3     #ps


#%% GNLSE

fibra1  = GNLSE(lambda0=lambda0,N=N,Tmax=Tmax,
                betas=betas,gammas=[gamma0,gamma1],satgamma=satgamma)#,
                # fR=fR,tau1=tau1,tau2=tau2)
AT0     = fundamentalSoliton(fibra1.T,T0=T0,beta2=betas[0],gamma0=gamma0)
AW0     = FFT(AT0) #initial field

fibra1.rtol = 1e-3
fibra1.atol = 1e-6
fibra1.method = 'RK45'
fibra1.integrateIP(AW0, Z, progress=True)
        


#%% pcGNLSE

fibra2  = pcGNLSE(lambda0=lambda0,N=N,Tmax=Tmax,
                  betas=betas,gammas=[gamma0,gamma1],satgamma=satgamma)#,
                # fR=fR,tau1=tau1,tau2=tau2)
AT0     = fundamentalSoliton(fibra2.T,T0=T0,beta2=betas[0],gamma0=gamma0)
AW0     = FFT(AT0) #initial field

fibra2.rtol = 1e-3
fibra2.atol = 1e-6
fibra2.method = 'RK45'
fibra2.integrateIP(AW0, Z, progress=True)
        

#%% FIGURES

f1 = plt.figure()
plotevol(dB(fibra1.A),Z,fibra1.W/(2*np.pi),fftshft=True,vmin=40)
ax = f1.gca()
ax.set_title('Spectral evolution - GNLSE')
ax.set_ylim(-100,10)

f1 = plt.figure()
plotevol(dB(fibra2.A),Z,fibra2.W/(2*np.pi),fftshft=True,vmin=40)
ax = f1.gca()
ax.set_title('Spectral evolution - pcGNLSE')
ax.set_ylim(-100,10)

f1 = plt.figure()
ax = f1.gca()
plt.plot(FFTSHIFT(fibra1.W)/(2*np.pi),FFTSHIFT(np.abs(fibra1.A[:,0])**2),label='Input')
plt.plot(FFTSHIFT(fibra1.W)/(2*np.pi),FFTSHIFT(np.abs(fibra1.A[:,200])**2),label='GNLSE')
plt.plot(FFTSHIFT(fibra2.W)/(2*np.pi),FFTSHIFT(np.abs(fibra2.A[:,200])**2),label='pcGNLSE')
ax.set_xlim(-200,+200)
ax.set_xlabel('Frequency [THz]')
ax.set_ylabel('Spectral Density [a.u.]')
plt.legend()

f1 = plt.figure()
ax = f1.gca()
plt.plot(fibra2.T,np.abs(iFFT(fibra2.A[:,0]))**2,label='Input')
plt.plot(fibra1.T,np.abs(iFFT(fibra1.A[:,200]))**2,label='GNLSE')
plt.plot(fibra2.T,np.abs(iFFT(fibra2.A[:,200]))**2,label='pcGNLSE')
ax.set_xlim(-0.5,+3)
ax.set_xlabel('Time [ps]')
ax.set_ylabel('Power [W]')
plt.legend()


ph1 = np.zeros_like(Z)
ph2 = np.zeros_like(Z)

for k in range(len(Z)):
    ph1[k] = np.sum(np.abs(fibra1.A[:,k])**2/(omega0+fibra1.W))
    ph2[k] = np.sum(np.abs(fibra2.A[:,k])**2/(omega0+fibra2.W))    

ph1 = ph1/ph1[0]*100
ph2 = ph2/ph2[0]*100

f1 = plt.figure()
ax = f1.gca()
plt.plot(Z,ph1,label='GNLSE')
plt.plot(Z,ph2,label='pcGNLSE')
# ax.set_xlim(-0.5,+3)
ax.set_xlabel('Distance [m]')
ax.set_ylabel('No. of photons [%]')
plt.legend()
