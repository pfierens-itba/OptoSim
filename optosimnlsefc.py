# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:02:27 2021

@author: Firulais
"""
import numpy as np
from scipy.integrate import simpson
from math import factorial
from optocommon import FFT, iFFT, FFTSHIFT
from optosimnlse import GNLSE


#GNLSE estándar: Raman con un solo pico. Ver Agrawal NLFO Capítulo 2.
#Includes two-photon absorption and free carriers effects according to
#Lin, Painter and Agrawal, Optics Express 15, 2007
class GNLSEFC(GNLSE):
    
    def __init__(self,
                 lambda0=1550,N=2**13,Tmax=10,
                 betas=[-20],alpha=0.0,
                 gammaskerr=[0.1],satgammakerr=3e6,
                 gammastpa=[0.1],satgammatpa=3e6,
                 sigmafca=1.45e-21,sigmafcr=5.3e-27,taufc=3e3,nlinearfc=0,
                 fR=0.048,tau1=0.01,tau2=3.00):
        
        #Type of equation
        self.type = "NLSE-Free Carriers"
        
        #Common definitions for all cases
        self._initcommon(lambda0,N,Tmax)

        #Raman & and Kerr nonlinearity
        self.raman(fR,tau1,tau2)
        self.gammaw(gammaskerr,satgammakerr)

        #Dispersion
        self.linop(betas,alpha)
      
        #Free carriers
        self.nlinearfc  = nlinearfc
        self.sigmafc    = -sigmafca/2-1j*sigmafcr/2
        self.taufc      = taufc
        self.intconst1  = 1-self.dT/self.taufc
        self.intconst2  = self.nlinearfc*self.dT
        
        #TPA
        self.Gamma = self.Gamma+self.gammaTPA(gammastpa,satgammatpa)


    def gammaTPA(self,gammas=[0.1],satgamma=3e6):
        g = 0
        for i in range(len(gammas)):        # Taylor de beta(Omega)
            g = g + gammas[i]/factorial(i) * self.W**i
        g[g>+satgamma] = +satgamma
        g[g<-satgamma] = -satgamma

        return 1j*g        

    def NonlinearOp(self,z,A):
        AT = iFFT(A)
        AT2= AT*np.conj(AT)
        P2 = AT2**2
        NT = np.zeros_like(P2)
        for k in range(1,self.N):
            NT[k] = NT[k-1]*self.intconst1+P2[k-1]*self.intconst2

        dA = 1j * self.Gamma * FFT(AT * iFFT( self.RW * FFT(AT2) ) )
        dA = dA + FFT(self.sigmafc*NT*AT)
                
        return dA

