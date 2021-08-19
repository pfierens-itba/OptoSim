# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:30:47 2021

@author: pfierens
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from RK45 import RK45

# try:
#     import pyfftw.interfaces.numpy_fft as fft
#     modoFFTW = True
#     print('Modo FFTW')
# except:
#     import numpy.fft as fft
#     modoFFTW = False
#     print('Modo numpy.fft')

# import numpy.fft as fft
import scipy.fft as fft

modoFFTW = False
print('Modo numpy.fft')

FFT =lambda X:fft.ifft(X)*len(X)
iFFT=lambda X:fft.fft(X)/len(X)
FFTSHIFT =lambda X:fft.fftshift(X)



#Speed of light
c = 299792458*1e9/1e12 # nm/ps

# === Raman response
def Raman(T,fR=0.18,tau1=0.0122,tau2=0.032):
  RamT = np.zeros(T.size)
  RamT[T>=0] = (tau1**2+tau2**2)/tau1/tau2**2*np.exp(-T[T>=0]/tau2)*np.sin(T[T>=0]/tau1)
  RamT[T<0] = 0 # heaviside step function
  RamT = fft.fftshift(RamT)/np.sum(RamT) # normalizamos (el fftshift está para q la rta arranque al inicio de la ventana temporal)
  RamW = FFT(RamT) # Rta. Raman en la frecuencia

  RW = fR*RamW + (1-fR)*np.ones(len(RamW)) # Raman + resp. "instantanea" (i.e., electrónica)

  #agregado por PIF para hacer un poco más rápida la pcGNLSE
  RWpc = 2.0*fR*(RamW-np.ones(len(RamW)))

  return RW, RWpc

def gammaw(W,gamma0=0.1,gamma1=0,lambda0=1550,satgamma=10):
  omega0   = 2.0*np.pi*c/lambda0
  gamma    = gamma0+gamma1*W # en 1/(W-m)
  gamma[gamma>+satgamma] = +satgamma
  gamma[gamma<-satgamma] = -satgamma
  r        = ((gamma+1j*0)/(W+omega0))**(1/4)
  cr       = np.conj(r) 
  gammaeff = 0.5*(((gamma+1j*0.0)*(W+omega0)**3)**(1/4))
  cgammaeff= np.conj(gammaeff)

  return gamma, r, cr, gammaeff, cgammaeff

def integrateIP(N, L, A0, Z, verbose=False, progress=True):
    """Integra una Ec. diferencial (metodo rk45) en el "interaction picture" (IP)
    Parametros:
    N: funcion (operador) que determina la evolucion (tipicamente la perturbación no lineal al Hamiltoniano)
    L: operador lineal que define la transformación al IP mediante la transformación e^(L z)
    A0: campo inicial (en el dominio de la frecuencia)
    Z: vector de distancias para los cuales se obtiene el campo [m]
    verbose: si verdadero imprime info que devuelve el solver (default es False).
    progress: (default: True) imprime el progreso en términos de % de distancia
    """
    rhs = lambda z, A: np.exp(L*z)*N(z, np.exp(-L*z)*A)
    if progress:
        A = np.zeros((len(A0),len(Z)),dtype='complex128')
        A[:,0] = A0
        for i in range(len(Z)-1):
            R = solve_ivp(rhs, (Z[i],Z[i+1]), A[:,i], t_eval=Z[i:i+1],rtol=1e-6, atol=1e-7)
            A[:,i+1] = np.exp(-L*Z[i+1])*R.y[:,-1]
            print("\r Progress: [","#"*int(40*Z[i+1]/Z[-1])+"."*int(40*(Z[-1]-Z[i+1])/Z[-1]),f'] {Z[i]/Z[-1]*100:.2f} %', end="")
            # if verbose:
            #     print("\n Number of evaluations of the right-hand side: "+str(R.nfev),end="")
            #     print("\n"+R.message,end="")
        return Z, A
    else:
        R = solve_ivp(rhs, (Z[0],Z[-1]), A0, t_eval=Z,method='DOP853')#, rtol=1e-5, atol=1e-8)  #A0 = A0*np.exp(L*0)
        if verbose:
            print("Number of evaluations of the right-hand side: "+str(R.nfev))
            print(R.message)
        return Z, np.exp(-(np.array([L]).T).dot(np.array([Z[0:R.y.shape[1]]])))*R.y

def integrateIPRK45(N, L, A0, Z, verbose=False, progress=True):
    """Integra una Ec. diferencial (metodo rk45) en el "interaction picture" (IP)
    Parametros:
    N: funcion (operador) que determina la evolucion (tipicamente la perturbación no lineal al Hamiltoniano)
    L: operador lineal que define la transformación al IP mediante la transformación e^(L z)
    A0: campo inicial (en el dominio de la frecuencia)
    Z: vector de distancias para los cuales se obtiene el campo [m]
    verbose: si verdadero imprime info que devuelve el solver (default es False).
    progress: (default: True) imprime el progreso en términos de % de distancia
    """
    rhs = lambda z, A: np.exp(L*z)*N(z, np.exp(-L*z)*A)
    if progress:
        verbose = True
    A = RK45(rhs,A0,Z,min_step=1e-7,rtol=1e-3, atol=1e-6, verbose=verbose)
    return Z, np.exp(-(np.array([L]).T).dot(np.array([Z])))*A

def linop(betas,W,alpha=0):
    """Operador lineal para una fibra óptica.
    Parametros:
    betas: lista o vector de valores de los coeficientes del desarrollo en serie de la dispersion a partir
           de beta_2 (GVD)
    alpha: atenuación (por defecto, nula)."""
    B = 0
    for i in range(len(betas)):        # Taylor de beta(Omega)
        B = B + betas[i]/math.factorial(i+2) * W**(i+2)
    return -1j*B + alpha   
    
def plotevol(Data,X,Y,fftshft=False,vmin=[],ylim=None,ax=None):
    if ax is None:
        ax=plt.gca()
        
    if fftshft:
        Data=fft.fftshift(Data,axes=0)
        Y=fft.fftshift(Y)
    if not vmin:
        ax.imshow(Data,aspect='auto',extent=(X[0],X[-1],Y[-1],Y[0]))
    else:
        ax.imshow(Data,aspect='auto',extent=(X[0],X[-1],Y[-1],Y[0]), vmin=vmin,cmap='hot')
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

# operador no lineal para ser pasado al integrador (integrateIP)
# caso NLSE sin SS
def N_NLSE_nss(z,A,gamma0):
    AT = iFFT(A)
    return 1j * gamma0 * FFT(np.abs(AT)**2 * AT)

# caso NLSE
def N_NLSE(z,A,gamma):
    AT = iFFT(A)
    return 1j * gamma * FFT(np.abs(AT)**2 * AT)

# operador no lineal para la GNLSE
def N_GNLSE(z,A,gamma,RW):
    AT = iFFT(A)
    return 1j * gamma * FFT(AT * iFFT( RW * FFT(AT*np.conj(AT)) ) )

# caso pcNLSE
def N_pcNLSE(z,A,r,cr,gammaeff,cgammaeff):
    BT = iFFT(r  * A)
    CT = iFFT(cr * A)
    return 1j * (gammaeff * FFT(np.conj(CT) * BT**2) + cgammaeff * FFT(np.conj(BT) * CT**2) )

# caso pcGNLSE
def N_pcGNLSE(z,A,r,cr,gammaeff,cgammaeff,RWpc):
    BT  = iFFT(r  * A) 
    CT  = iFFT(cr * A)
    cBT = np.conj(BT)
    cCT = np.conj(CT)
        
    return 1j * (    gammaeff * FFT(cCT * BT**2)  \
                 +  cgammaeff * FFT(cBT * CT**2 + \
                                    iFFT(RWpc*FFT(cBT*BT)) * BT ) )
