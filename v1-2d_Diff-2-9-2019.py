#  In the name of God
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 2 13:59:48 2019
@author: samieeme
"""

import numpy as np
#from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, optimum_prnt
#from Outputs import Output_Corr


def forcef(x,y):
    return -4.0*np.sin(2.0*x)*np.cos(3.0*y)-9.0*np.sin(2.0*x)*np.cos(3.0*y)

def solV(x,y):
    return np.sin(2.0*x)*np.cos(3.0*y)

def deriv_x(Nnod,Vhat):
    divhat = np.zeros((Nnod,Nnod,Nnod),dtype = complex)
    kx = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           kx[i1,:,:] = complex(0,i1)
        else:
           kx[i1,:,:] = complex(0,i1-Nnod)   
    
    divhat = kx * Vhat
    diverx_V = np.real(np.fft.ifftn(divhat))
    return diverx_V

def diff_x(Nnod,Vhat):
#    ky = np.zeros((Nnod,Nnod,Nnod))
   divhat = np.zeros((Nnod,Nnod))
    kx = np.zeros((Nnod,Nnod))
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
            kx[i1,:] = i1
        else:
            kx[i1,:] = i1-Nnod
    alpha = 1;
    frac_L = -(kx[:]**2)**(alpha)       
    divhat = frac_L * Vhat
    diverz_V = np.real(np.fft.ifft2(divhat))
    return diverz_V

def diff_eq(Nnod,fhat):
    divhat = np.zeros((Nnod,Nnod))
    kx = np.zeros((Nnod,Nnod))
    ky = np.zeros((Nnod,Nnod))
   # kx[0,0]=1.0/10000000.0
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
            kx[i1,:] = i1
        else:
            kx[i1,:] = (i1-Nnod)
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
            ky[:,i2] = i2
        else:
            ky[:,i2] = (i2-Nnod)
    alpha = 1;
    frac_L = -(kx[:]**2+ky[:]**2)**(alpha)
    frac_L[0,0]=1.0
    frac_R=1.0/frac_L
    divhat = frac_R * fhat
    diverz_V = np.real(np.fft.ifft2(divhat))
    diverz_V[0,0] = 0.0
    return diverz_V

Nnod = 10
meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)

V = np.zeros([Nnod,Nnod])
f = np.zeros([Nnod,Nnod])

for i in range(Nnod):
    for j in range(Nnod):
        V[i,j] = solV(meshX[i],meshX[j])
        f[i,j] = forcef(meshX[i],meshX[j])
#V = solV(meshX) 
#f = forcef(meshX)

Vhat = np.fft.fft2(V)
fhat = np.fft.fft2(f)

vs = diff_eq(Nnod,fhat)
dxV = diff_x(Nnod,Vhat)

#err = dxV-f
err = vs-V


#solver = Output_Corr(filename,Rs,time)
#
#corr_smg = solver.SMG_Model()
#np.savetxt(fileout+'output-time-'+time+'-R'+Rs+'-SMG.csv', corr_smg)
#
#nu = 0.000185 #1.0/1600
#Re = solver.Resolved_Re_Nektar (int(Rs),nu)
