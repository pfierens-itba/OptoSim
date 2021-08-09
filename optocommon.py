# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:00:35 2021

@author: Firulais
"""
import scipy.fft as fft
import numpy as np
import matplotlib.pyplot as plt

###################################
# CONSTANTES
###################################

#Speed of light
c = 299792458*1e9/1e12 # nm/ps    
#Planck constant
h = 6.62607015e-34*1e12 #J*ps
hbar = h/(2.0*np.pi)

###################################
#FFT
###################################

FFT         = lambda X:fft.ifft(X)*len(X)
iFFT        = lambda X:fft.fft(X)/len(X)
FFTSHIFT    = lambda X:fft.fftshift(X)
FFTSHIFT2D  = lambda X,ax:fft.fftshift(X,axes=ax)    

###################################
# MISC
###################################
def sech(t):
    return 2/(np.exp(t)+np.exp(-t))

def dB(x):
    return 20*np.log10(np.abs(x))

def inten(x):
    return np.abs(x)**2

def var2dat(x,y,filename):
    fo = open(filename, 'w')
    for a,b in zip(x,y):
        #fo.write(str(a)+'\t'+str(b)+'\n')
        fo.write('{c1:.10f}\t{c2:.10f}\n'.format(c1=a,c2=b))
    fo.close()
    

###################################
# PULSOS
###################################

def sechPulse(P0,T,T0 = 1,TFWHM = 0,C = 0):
    if TFWHM != 0:
        T0 = TFWHM/(2.0*np.log(1+np.sqrt(2)))
    t = T/T0
    return np.sqrt(P0)*sech(t)*np.exp((-1j*C/2.0)*t**2)

def gaussianPulse(P0,T,T0 = 1,C = 0):
    t = T/T0
    return np.sqrt(P0)*np.exp(((-1-1j*C)/2.0)*t**2)
    
def supergaussianPulse(P0,T,T0 = 1,C = 0,m = 1):
    t = T/T0
    return np.sqrt(P0)*np.exp(((-1-1j*C)/2.0)*t**(2*m))

def fundamentalSoliton(T,T0 = 1,TFWHM = 0,beta2 = 1,gamma0 = 1):
    if TFWHM != 0:
        T0 = TFWHM/(2.0*np.log(1+np.sqrt(2)))
    t = T/T0
    P0 = np.abs(beta2)/(gamma0*T0**2)
    return np.sqrt(P0)*sech(t)
    
##############################################
### GRÁFICOS
##############################################

def plotevol(Data,X,Y,fftshft=False,vmin=[],ylim=None,ax=None):
    if ax is None:
        ax=plt.gca()
        
    if fftshft:
        Data=fft.fftshift(Data,axes=0)
        Y=fft.fftshift(Y)
    if not vmin:
        ax.imshow(Data,aspect='auto',extent=(X[0],X[-1],Y[-1],Y[0]))
    else:
        ax.imshow(Data,aspect='auto',extent=(X[0],X[-1],Y[-1],Y[0]), vmin=vmin)
    if ylim:
        ax.set_ylim((ylim[0],ylim[1]))
              
    plt.xlabel(r'Distancia [m]')
    if fftshft:
        plt.ylabel(r'Frecuencia [THz]')
    else:
        plt.ylabel(r'Tiempo [ps]')
    plt.show()
    return(ax)

def plotanim(Data,X,step=0.02,ylim=[]):
    plt.ion()
    fig, ax = plt.subplots(1,1)
    if ylim: # ylim no vacío
        plt.ylim(ylim)
    line, = ax.plot(X,Data[:,0])
    fig.canvas.draw()
    for i in range(Data.shape[1]):
        plt.pause(step)
        line.set_ydata(Data[:,i])
        fig.canvas.draw()
    plt.ioff()

def xfrog(A,T,twidth,tstep,N,vmin=-80):
    
    t0 = twidth/(2*np.sqrt(np.log(2.0)))
    n = int((T[-1]-T[0])/tstep+1)
    t = np.linspace(T[0],T[-1],n)
    S = np.zeros((A.shape[0],n))
    for k in range(len(t)):
        B = A*np.exp(-0.5*((T-t[k])/t0)**2)
        s = FFT(B)
        S[:,k] = fft.fftshift(np.abs(s)**2)
    M = np.max(S)
    ax=plt.gca()
    F = np.arange(-N/2,N/2) / (len(T)*(T[1]-T[0]))
    ax.imshow(10*np.log10(S/M),aspect='auto',extent=(T[0],T[-1],F[-1],F[0]), vmin=vmin, vmax=0)
    
    return S,F,t

