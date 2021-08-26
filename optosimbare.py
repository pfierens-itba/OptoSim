# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 11:04:01 2021

@author: Firulais
"""
import numpy as np
from math import factorial
from tqdm import tqdm
from scipy.integrate import solve_ivp
from optocommon import FFT, iFFT, FFTSHIFT, PhysConst

class SingleModePE():
    
    def __init__(self,
                 lambda0=1550,N=2**13,Tmax=10,
                 betas=[-20],alphas=[0.0],gammas=[0.1],satgamma=0.0,
                 cnst=PhysConst()):
        
        #Type of equation
        self.type = "NLSE"

        #Common definitions for all cases
        self._initcommon(lambda0,N,Tmax,cnst)

        #Definitions specific to this equation
        self.linop(betas,alphas)
        self.gammaw(gammas,satgamma)
        
                            
    def linop(self,betas=[-20],alphas=[0]):
        """Operador lineal para una fibra óptica.
        Parametros:
        betas: lista o vector de valores de los coeficientes del desarrollo en serie de la dispersion a partir
               de beta_2 (GVD)
        alpha: atenuación (por defecto, nula)."""
        
        #If there are as many values of betas as N, it assumes that it provides
        #the values as a function of W (already FFTSHIFTED)
        if len(betas) == self.N:
            B = betas
        else:
            B = 0
            for i in range(len(betas)):        # Taylor de beta(Omega)
                B = B + betas[i]/factorial(i+2) * self.W**(i+2)
        #If there are as many values of alpha as N, it assumes that it provides
        #the values as a function of W (already FFTSHIFTED)
        if len(alphas) == self.N:
            A = alphas
        else:
            A = 0
            for i in range(len(alphas)):        # Taylor de beta(Omega)
                A = A + alphas[i]/factorial(i) * self.W**(i)
                
        self.LinOP = -1j*B + A/2.0

    def gammaw(self,gammas=[0.1],satgamma=3e6):

        #If there are as many values of gammas as N, it assumes that it provides
        #the values as a function of W (already FFTSHIFTED)
        if len(gammas) == self.N:
            g = gammas
        else:
            g = 0
            for i in range(len(gammas)):        # Taylor de beta(Omega)
                g = g + gammas[i]/factorial(i) * self.W**i
            g[g>+satgamma] = +satgamma
            g[g<-satgamma] = -satgamma

        self.Gamma = g
      
    def NonlinearOp(self,z,A):
        AT = iFFT(A)
        return (1j * self.Gamma * FFT(np.abs(AT)**2 * AT))
    
    #########################################################
    ## NOTHING SHOULD CHANGE FROM THIS POINT ON
    #########################################################
    
    #COMMON DEFINITIONS FOR ALL CASES
    def _initcommon(self,lambda0,N,Tmax,cnst):
        #Physical constants
        self.cnst = cnst
        
        #WAVELENGTH [nm]
        self.lambda0 = np.abs(float(lambda0))
        self.omega0 = 2.0*np.pi*self.cnst.cwave/lambda0
        
        #NUMBER OF POINTS
        #We use a power of 2
        N = int(2**(np.ceil(np.log2(N))))
        self.N = N
        
        #TIME
        self.Tmax = float(Tmax)
        
        self.dW   = np.pi/Tmax
        self.dT   = 2.0*Tmax/N
        self.fs   = 1/self.dT
        self.T    = np.arange(-N/2,N/2)*self.dT
        self.W    = FFTSHIFT(np.pi * np.arange(-N/2,N/2) / Tmax)
        
        self.Omega= self.omega0 + np.pi * np.arange(-N/2,N/2) / Tmax
        
        if np.any(self.Omega < 0):
            print("Waveguide Init: Negative frequencies")

        #Results of the simulation
        self.A = []         #Fourier transform of the envelope
        self.Z = []         #Distance
        
        #Details of the integration method
        self.method = 'RK45'
        self.rtol = 1e-3
        self.atol = 1e-6
    
    #La convención es que con  _ es una función o variable "privada"
    def _RHS(self,z,A):
        elinop = np.exp(self.LinOP*z)
        return elinop * self.NonlinearOp(z,A/elinop)

    def _RHSprogress(self,z,A,pbar,state): 
        last_z, dz = state
        n = int((z - last_z)/dz)
        pbar.update(n)
        state[0] = last_z + dz * n
        elinop = np.exp(self.LinOP*z)
        return elinop * self.NonlinearOp(z,A/elinop)

    def integrateIP(self, A0, Z, verbose=False, progress=True):
        """Integrates a differential equation in the "interaction picture" (IP)
        The IP is partially implemented in the function RHS.
        Parametros:
        A0: Fourier transform of the initial envelope
        Z: distance for which the field is required [m]
        verbose: (default: False)if it is true, it prints the information returned by the solver
        progress: (default: True) prints the progress as a % of the total length
        """
        if progress:
            with tqdm(total=1000, unit="‰") as pbar:
                R = solve_ivp(self._RHSprogress,
                    (Z[0],Z[-1]), A0, t_eval=Z,
                    method=self.method, rtol=self.rtol, atol=self.atol,
                    args=[pbar, [Z[0], (Z[-1]-Z[0])/1000]]
                    )
        else:
            R = solve_ivp(self._RHS, 
                (Z[0],Z[-1]), A0, t_eval=Z, 
                method=self.method, rtol=self.rtol, atol=self.atol
                ) 
        if verbose:
            print("Number of evaluations of the right-hand side: "+str(R.nfev))
            print(R.message)
        A = np.exp(-(np.array([self.LinOP]).T).dot(np.array([Z[0:R.y.shape[1]]])))*R.y
        self.Z = Z[0:R.y.shape[1]]
        self.A = A


