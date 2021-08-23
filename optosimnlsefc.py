# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:02:27 2021

@author: Firulais
"""
import numpy as np
from math import factorial
from optocommon import FFT, iFFT, PhysConst
from optosimnlse import GNLSE


#GNLSE standard: Raman with a single Lorentzian frequency.
#See Agrawal NLFO Capítulo 2.
#It includes two-photon absorption and free carriers effects according to
#Lin, Painter and Agrawal, Optics Express 15, 2007
class GNLSEFC(GNLSE):
    
    def __init__(self,
                 lambda0=1550,N=2**13,Tmax=10,
                 betas=[-20],alpha=0.0,
                 gammaskerr=[0.1],satgammakerr=3e6,
                 gammastpa=[0.1],satgammatpa=3e6,
                 sigmafca=1.45e-21,sigmafcr=5.3e-27,taufc=3e3,nlinearfc=0,
                 fR=0.048,tau1=0.01,tau2=3.00,
                 cnst=PhysConst()):
        
        #Type of equation
        self.type = "NLSE-Free Carriers"
        
        #Common definitions for all cases
        self._initcommon(lambda0,N,Tmax,cnst)

        #Raman & and Kerr nonlinearity
        self.raman(fR,tau1,tau2)
        self.gammaw(gammaskerr,satgammakerr)

        #Dispersion
        self.linop(betas,alpha)
 
        #TPA
        self.GammaTPA = self.gammaTPA(gammastpa,satgammatpa)
        self.updategamma()
        
 
        #Free carriers
        self.freecarriers(nlinearfc,sigmafca,sigmafcr,taufc)
        
        
    def gammaTPA(self,gammas=[0.1],satgamma=3e6):
        g = 0
        for i in range(len(gammas)):        # Taylor de beta(Omega)
            g = g + gammas[i]/factorial(i) * self.W**i
        g[g>+satgamma] = +satgamma
        g[g<-satgamma] = -satgamma

        return 1j*g        

    def updategamma(self):
        self.Gamma = self.Gamma+self.GammaTPA

    def freecarriers(self,nlinearfc,sigmafca,sigmafcr,taufc):
        self.nlinearfc  = nlinearfc
        self.sigmafc    = -sigmafca/2-1j*sigmafcr/2
        self.taufc      = taufc
        self.greenfc    = self.nlinearfc/(1/self.taufc-1j*self.W)

    def NonlinearOp(self,z,A):
        AT = iFFT(A)
        AT2= AT*np.conj(AT)       
        NT = np.real(iFFT(FFT(AT2**2)*self.greenfc))

        dA = 1j * self.Gamma * FFT(AT * iFFT( self.RW * FFT(AT2) ) )
        dA = dA + FFT(self.sigmafc*NT*AT)
                
        return dA

