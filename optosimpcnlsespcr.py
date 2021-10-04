# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:01:25 2021

@author: Firulais
"""
import numpy as np
from math import factorial
from optocommon import FFT, iFFT, PhysConst
from optosimpcnlse import pcGNLSE
from numba import njit

@njit(fastmath=True)
def loop(NTG,AT2,Nsat,sp1,sp2):
    NTG[0] = 0.0
    for k in range(1,len(NTG)):
        if NTG[k-1] >= Nsat:
            NTG[k-1] = Nsat
            NTG[k] = sp2*NTG[k-1]
        else:                    
            NTG[k] = sp2*NTG[k-1] + \
                     sp1*(1-NTG[k-1]/Nsat)*AT2[k-1]    
    return NTG
    
#GNLSE standard: Raman with a single Lorentzian frequency.
#See Agrawal NLFO Capítulo 2.
#It includes free carriers effects according to
#Lin, Painter and Agrawal, Optics Express 15, 2007
#It includes two-photon absorption according to 
#Linale et al., JOSA B 37, 2020.
#It includes saturable photoexcited-carrier refraction (SPCR) according to
#Vermeulen et al., Nature Communications 9, 2018. 
class pcNLSESPCR(pcGNLSE):
    
    def __init__(self,
                 lambda0=1550,N=2**13,Tmax=10,
                 betas=[-20],alphas=[0.0],
                 gammaskerr=[0.1],satgammakerr=3e6,
                 gammastpa=[0.1],satgammatpa=3e6,
                 sigmafca=1.45e-21,sigmafcr=5.3e-27,taufc=3e3,nlinearfc=0,
                 sigmasp=[-1e-5],Nsat=3e8,tausp=200,
                 fR=0.048,tau1=0.01,tau2=3.00,ramanw=[],
                 cnst=PhysConst()):
        
        #Type of equation
        self.type = "NLSE-Free Carriers"
        
        #Common definitions for all cases
        self._initcommon(lambda0,N,Tmax,cnst)

        #Raman & and Kerr nonlinearity
        self.raman(fR,tau1,tau2,ramanw)
        self.gammaw(gammaskerr,satgammakerr)

        #Dispersion
        self.linop(betas,alphas)
 
        #TPA
        self.gammaTPA(gammastpa,satgammatpa)
 
        #Free carriers in the waveguide
        self.freecarriersW(nlinearfc,sigmafca,sigmafcr,taufc)
        
        #Free carriers in decorating graphene
        self.freecarriersG(alphas,sigmasp,Nsat,tausp)
        
    def gammaTPA(self,gammas=[0.1],satgamma=3e6):
        #If there are as many values of gammas as N, it assumes that it provides
        #the values as a function of W (already FFTSHIFTED)
        if len(gammas) == self.N:
            g = gammas
        else:
            g = 0
            for i in range(len(gammas)):        # Taylor de gamma(Omega)
                g = g + gammas[i]/factorial(i) * self.W**i
            g[g>+satgamma] = +satgamma
            g[g<-satgamma] = -satgamma
        self.GammaTPAeffa = np.sqrt(np.sqrt((g+1j*0)/(self.omega0+self.W)))
        self.GammaTPAeffb = (self.omega0+self.W)*self.GammaTPAeffa


    def freecarriersW(self,nlinearfc,sigmafca,sigmafcr,taufc):
        self.nlinearfc  = nlinearfc
        self.sigmafc    = -sigmafca/2-1j*sigmafcr/2
        self.taufc      = taufc
        self.greenfc    = self.nlinearfc/(1/self.taufc-1j*self.W)

    def freecarriersG(self,alphas,sigmasp,Nsat,tausp):
        #If sigmasp has N elements, it is assume that it corresponds to
        #the frequency dependent value FFTSHIFTed
        if len(sigmasp) == self.N:
            g = sigmasp
        else:
            g = 0
            for i in range(len(sigmasp)):        # Taylor de gamma(Omega)
                g = g + sigmasp[i]/factorial(i) * self.W**i
        self.SigmaSPCR      = g
        self.tausp          = tausp
        self.Nsat           = Nsat
        self.spcrintconst1  = alphas[0]/(self.cnst.hbar*self.omega0)*self.dT
        self.spcrintconst2  = 1.0-self.dT/tausp
        self.NTG            = np.zeros_like(self.T)


    
    def NonlinearOp(self,z,A):
        AT  = iFFT(A)
        AT2 = np.real(AT*np.conj(AT))
        BT  = iFFT(self.r  * A) 
        CT  = iFFT(self.cr * A)
        DT  = iFFT(self.GammaTPAeffa * A)
        cBT = np.conj(BT)
        cCT = np.conj(CT)
        cDT = np.conj(DT)
        
        #Kerr + Raman
        dA  = 1j * (    self.gammaeff  * FFT(cCT * BT**2)  \
                     +  self.cgammaeff * FFT(cBT * CT**2 + \
                                        iFFT(self.RWpc*FFT(cBT*BT)) * BT ) )
        #TPA
        dA  = dA - self.GammaTPAeffb * FFT(cDT * DT**2)

        #Free Carriers
        NTW = np.real(iFFT(FFT(AT2**2)*self.greenfc))
        dA  = dA + FFT(self.sigmafc*NTW*AT)
        
        #SPCR
        # self.NTG[0] = 0.0
        # for k in range(1,self.N):
        #     if self.NTG[k-1] >= self.Nsat:
        #         self.NTG[k-1] = self.Nsat
        #         self.NTG[k] = self.spcrintconst2*self.NTG[k-1]
        #     else:                    
        #         self.NTG[k] = self.spcrintconst2*self.NTG[k-1] + \
        #                       self.spcrintconst1*(1-self.NTG[k-1]/self.Nsat)*AT2[k-1]
        self.NTG = loop(self.NTG,AT2,self.Nsat,self.spcrintconst1,self.spcrintconst2)
        dA = dA + 1j * self.SigmaSPCR * FFT(self.NTG*AT)
         
        return dA

