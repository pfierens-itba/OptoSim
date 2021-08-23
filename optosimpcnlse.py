# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:02:27 2021

@author: Firulais
"""
import numpy as np
from math import factorial
from optocommon import FFT, iFFT, FFTSHIFT, PhysConst
from optosimbare import SingleModePE

#pcGNLSE
class pcGNLSE(SingleModePE):
    
    def __init__(self,
                 lambda0=1550,N=2**13,Tmax=10,
                 betas=[-20],alpha=0.0,gammas=[0.1],satgamma=3e6,
                 fR=0.18,tau1=0.0122,tau2=0.032,
                 cnst=PhysConst()):
        
        #Type of equation
        self.type = "NLSE"
        
        #Common definitions for all cases
        self._initcommon(lambda0,N,Tmax,cnst)

        #Definitions specific to this equation
        self.linop(betas,alpha)
        self.gammaw(gammas,satgamma)
        self.raman(fR,tau1,tau2)

    def gammaw(self,gammas=[0.1],satgamma=3e6):
        g = 0
        for i in range(len(gammas)):        # Taylor de beta(Omega)
            g = g + gammas[i]/factorial(i) * self.W**i
        g[g>+satgamma] = +satgamma
        g[g<-satgamma] = -satgamma
        self.r        = ((g+1j*0)/(self.W+self.omega0))**(1/4)
        self.cr       = np.conj(self.r) 
        self.gammaeff = 0.5*(((g+1j*0.0)*(self.W+self.omega0)**3)**(1/4))
        self.cgammaeff= np.conj(self.gammaeff)        
        
    def raman(self,fR=0.18,tau1=0.0122,tau2=0.032):
        RamT = np.zeros(self.N)
        RamT[self.T>=0] = (tau1**2+tau2**2)/tau1/tau2**2*np.exp(-self.T[self.T>=0]/tau2)*np.sin(self.T[self.T>=0]/tau1)
        RamT[self.T<0] = 0 # heaviside step function
        RamT = FFTSHIFT(RamT)/np.sum(RamT) # normalizamos (el fftshift está para q la rta arranque al inicio de la ventana temporal)
        RamW = FFT(RamT) # Rta. Raman en la frecuencia
        #agregado por PIF para hacer un poco más rápida la pcGNLSE
        self.RWpc = 2.0*fR*(RamW-np.ones(len(RamW)))

    def NonlinearOp(self,z,A):
        BT  = iFFT(self.r  * A) 
        CT  = iFFT(self.cr * A)
        cBT = np.conj(BT)
        cCT = np.conj(CT)
            
        return 1j * (    self.gammaeff * FFT(cCT * BT**2)  \
                     +  self.cgammaeff * FFT(cBT * CT**2 + \
                                        iFFT(self.RWpc*FFT(cBT*BT)) * BT ) )

