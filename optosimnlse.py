# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:02:27 2021

@author: Firulais
"""
import numpy as np
from optocommon import FFT, iFFT, FFTSHIFT
from optosimbare import SingleModePE

#GNLSE estándar: Raman con un solo pico. Ver Agrawal NLFO Capítulo 2.
class GNLSE(SingleModePE):
    
    def __init__(self,
                 lambda0=1550,N=2**13,Tmax=10,
                 betas=[-20],alpha=0.0,gammas=[0.1],satgamma=3e6,
                 fR=0.18,tau1=0.0122,tau2=0.032):
        
        #Type of equation
        self.type = "GNLSE"
        
        #Common definitions for all cases
        self._initcommon(lambda0,N,Tmax)

        #Definitions specific to this equation
        self.linop(betas,alpha)
        self.gammaw(gammas,satgamma)
        self.raman(fR,tau1,tau2)
      
    def raman(self,fR=0.18,tau1=0.0122,tau2=0.032):
        RamT = np.zeros(self.N)
        RamT[self.T>=0] = (tau1**2+tau2**2)/tau1/tau2**2*np.exp(-self.T[self.T>=0]/tau2)*np.sin(self.T[self.T>=0]/tau1)
        RamT[self.T<0] = 0 # heaviside step function
        RamT = FFTSHIFT(RamT)/np.sum(RamT) # normalizamos (el fftshift está para q la rta arranque al inicio de la ventana temporal)
        RamW = FFT(RamT) # Rta. Raman en la frecuencia
        self.RW = fR*RamW + (1-fR)*np.ones(len(RamW)) # Raman + resp. "instantanea" (i.e., electrónica)

    def NonlinearOp(self,z,A):
        AT = iFFT(A)
        return 1j * self.Gamma * FFT(AT * iFFT( self.RW * FFT(AT*np.conj(AT)) ) )

