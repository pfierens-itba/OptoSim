# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:15:52 2021

@author: Firulais
"""
import numpy as np
import matplotlib.pylab as plt
from optocommon import FFT, iFFT, FFTSHIFT, gaussianPulse, c, hbar, plotevol, dB
from optosimnlsefc import GNLSEFC
from optosimpcnlse import pcGNLSE

#####################################################################
# Example: Figures 3 and 4 of Yin & Agrawal, 
#          Optics Letters 32(14), 2007.
#####################################################################

#SAMPLING AND TIME SPAN
N    = 2**12
Tmax = 600 # ps

#CENTRAL WAVELENGTH
lambda0 = 1550                  #nm
omega0  = 2*(np.pi)*c/lambda0   #Trad/s
k0      = 2*np.pi/(lambda0*1e-9)   #1/m

#DISPERSION
#betas in ps^m/m
betas = [-1]

#LOSS
alpha = 10**(1e-2/10)

#RAMAN (see Lin, Painter & Agrawal, Optics Express)
# fR      = 0.048
fR      = 0.00
tau1    = 0.01 #ps
tau2    = 3.00 #ps

#NONLINEARITY
n2          = 6e-18         #m**2/W
gammaKerr0  = n2*k0         #1/(W m)
satgammaKerr= 3e6

#TPA - TWO PHOTON ABSORPTION
betaTPA     = 5e-12                   #m/W
gammaTPA0   = 0.1*gammaKerr0          #1/(W m) 
satgammaTPA = 3e6

#FCA/FCR - FREE CARRIER ABSORPTION/REFRACTION
sigmaFCA    = 1.45e-21              #m**2
sigmaFCR    = 2*k0*1.35e-27         #

tauFC       = 1e3       #ps
nlinearFC   = betaTPA/(2.0*hbar*omega0)

#FIBER LENGTH
distance = 0.02 # m
Nz       = 100
Z        = np.linspace(0,distance,Nz)

#PULSE PARAMETERS
T0      = 10        #ps
P0      = 1.2e9/1e-4        #W/m2


#%% GNLSE

#Neither TPA nor FCA/FCR
fibra1  = GNLSEFC(lambda0=lambda0,N=N,Tmax=Tmax,
                 betas=betas,alpha=alpha,
                 gammaskerr=[gammaKerr0],satgammakerr=satgammaKerr,
                 gammastpa=[0],satgammatpa=satgammaTPA,
                 sigmafca=0,sigmafcr=0,taufc=tauFC,nlinearfc=nlinearFC,
                 fR=fR,tau1=tau1,tau2=tau2)

#Only TPA
fibra2  = GNLSEFC(lambda0=lambda0,N=N,Tmax=Tmax,
                 betas=betas,alpha=alpha,
                 gammaskerr=[gammaKerr0],satgammakerr=satgammaKerr,
                 gammastpa=[gammaTPA0],satgammatpa=satgammaTPA,
                 sigmafca=0,sigmafcr=0,taufc=tauFC,nlinearfc=nlinearFC,
                 fR=fR,tau1=tau1,tau2=tau2)

#Full
fibra3  = GNLSEFC(lambda0=lambda0,N=N,Tmax=Tmax,
                 betas=betas,alpha=alpha,
                 gammaskerr=[gammaKerr0],satgammakerr=satgammaKerr,
                 gammastpa=[gammaTPA0],satgammatpa=satgammaTPA,
                 sigmafca=sigmaFCA,sigmafcr=sigmaFCR,taufc=tauFC,nlinearfc=nlinearFC,
                 fR=fR,tau1=tau1,tau2=tau2)


AT0     = gaussianPulse(P0,fibra1.T,T0=T0)
AW0     = FFT(AT0) #initial field


fibra1.integrateIP(AW0, Z, progress=True)
fibra2.integrateIP(AW0, Z, progress=True)
fibra3.integrateIP(AW0, Z, progress=True)
    
Pw1 = np.abs(fibra1.A)**2
Pw2 = np.abs(fibra2.A)**2
Pw3 = np.abs(fibra3.A)**2



#%% Figure

lambdas = FFTSHIFT(2*np.pi*c/(omega0+fibra1.W))
f = plt.figure()
plt.plot(lambdas,FFTSHIFT(Pw1[:,-1])/lambdas**2,label='No TPA/No FC')
plt.plot(lambdas,FFTSHIFT(Pw2[:,-1])/lambdas**2,label='TPA Only')
plt.plot(lambdas,FFTSHIFT(Pw3[:,-1])/lambdas**2,label='Full')
ax = f.gca()
ax.set_xlim(1549,1551)
ax.set_ylabel('Spectral Density [a.u./nm]')
ax.set_xlabel('Wavelength [nm]')
plt.legend()
