# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 06:59:13 2021

@author: tiopi
"""

import numpy as np
from scipy.signal import firls, freqz, lfilter
from optocommon import PhysConst
import matplotlib.pyplot as plt



def filterdesign(fs,Numtaps=127,
                 lambda0=1600,lambdaa=1499,lambdab=1526,
                 transla=0.1,f0=0,fa=0,fb=0,transfr=0,cnst=PhysConst()):
    
    if f0 == 0:
        f0  = cnst.cwave/lambda0
        f1  = cnst.cwave/(lambdab+2*transla)-f0
        f2  = cnst.cwave/(lambdab+1*transla)-f0
        f3  = cnst.cwave/lambdab-f0
        f4  = cnst.cwave/lambdaa-f0
        f5  = cnst.cwave/(lambdaa-1*transla)-f0
        f6  = cnst.cwave/(lambdaa-2*transla)-f0
        f0  = cnst.cwave/lambda0
        f1  = cnst.cwave/(lambdab+2*transla)-f0
        f2  = cnst.cwave/(lambdab+1*transla)-f0
        f3  = cnst.cwave/(lambdab+0*transla)-f0
        f4  = cnst.cwave/(lambdaa-0*transla)-f0
        f5  = cnst.cwave/(lambdaa-1*transla)-f0
        f6  = cnst.cwave/(lambdaa-2*transla)-f0
    else:
        f1  = fa-2*transfr
        f2  = fa-1*transfr
        f3  = fa
        f4  = fb
        f5  = fb+1*transfr
        f5  = fb+2*transfr
    
    
    desired = [0,0,0,1,1,0,0,0]
    bands   = [0,f1,f2,f3,f4,f5,f6,fs/2]
    fir     = firls(Numtaps, bands, desired, fs=fs)
    
    freq, response = freqz(fir)
    m = max(np.abs(response))
    fir = fir/m
    freq, response = freqz(fir)
    # plt.plot(0.5*fs*freq/np.pi, 20*np.log10(np.abs(response)))
    plt.plot(cnst.cwave/(0.5*fs*freq/np.pi+f0), 20*np.log10(np.abs(response)))
    
    return fir

def filtersignal(fir,x,ax=0):
    return lfilter(fir, 1.0, x, axis=ax)
