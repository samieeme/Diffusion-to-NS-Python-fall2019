#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 01:41:07 2019

@author: akhavans
"""

import numpy as np
from numpy import linalg as LA


def get_sphere_waven(res):
    
    sz=res**2
    
    u_w=np.zeros((sz,1))
    
    wave_n=np.array([0.0,0.0])
    max_wave=int(res/2)
    ndx=0
    
    for j in range(0,res):
        for i in range(0,res):
            
            wave_n[0]=i
            if i > max_wave:
                wave_n[0]=i-res
            wave_n[1]=j
            if j > max_wave:
                wave_n[1]=j-res
            
            u_w[ndx]=LA.norm(wave_n, ord=2)
                            
            ndx +=1
            
    K_sh = u_w.reshape(res,res)
    K_sh2 = K_sh**2
    K_sh4 = K_sh2**2

    return K_sh, K_sh2, K_sh4

def get_stats_eng(uhat,vhat,nu,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc):
    
    
    E_k = 0.5*((uhat*np.conj(uhat) + vhat*np.conj(vhat)).real)
    
    TKE = np.sum(E_k)
    
    u_rms = TKE**0.5
    
    Enst = np.sum(K_sh2*E_k)
    
    eta = 2.0*nu*np.sum(K_sh4*E_k)
    
    Diss = 2.0*nu*Enst
    
    K_eta = (eta/nu**3)**(1./6.)
    
    int_lscale = TKE**0.5/eta**(1.0/3.0)
    
    mic_lscale = (nu*Enst/eta)**0.5
    
    Re_L = TKE/(nu*eta**(1.0/3.0))
    
    Re_l = Enst**1.5/eta
    
    T_L = int_lscale/u_rms
    
#    Computing energy of the E.C. modes to use in artificial forcing
    TKE_EngC = 0.0
    Uhat_EC = np.zeros((sz_frc,2),dtype=complex)
    
    for i in range(0,sz_frc):
        
        TKE_EngC += E_k[ndx_frc[i,0],ndx_frc[i,1]]
        
        Uhat_EC[i,0] = uhat[ndx_frc[i,0],ndx_frc[i,1]]
        Uhat_EC[i,1] = vhat[ndx_frc[i,0],ndx_frc[i,1]]
    
    TKE_EngC *= 2.0
    
    Uhat_EC *= Diss/TKE_EngC
    
    return TKE,Enst,eta,Diss,K_eta,int_lscale,mic_lscale,Re_l,Re_L,T_L,Uhat_EC

def deriv_(Vhat,K):
           
    divhat = - K * Vhat
    diver_V = np.real(np.fft.ifft2(divhat))
    
    return diver_V

def Moments_dVdX(sz,uhat,vhat,kx,ky):
    
    du1dx=deriv_(uhat,kx)
    du1dx=du1dx.reshape(sz,1)
    
    du2dy=deriv_(vhat,ky)
    du2dy=du2dy.reshape(sz,1)
    
    M1=np.array([])
    M2=np.array([])
    
    std1=np.mean(du1dx**2)
    std2=np.mean(du2dy**2)
    
    M1=np.append(M1,std1)
    M2=np.append(M2,std2)
    
    M1=np.append(M1,np.mean(du1dx**3)/std1**1.5)
    M2=np.append(M2,np.mean(du2dy**3)/std2**1.5)
    
    M1=np.append(M1,np.mean(du1dx**4)/std1**2)
    M2=np.append(M2,np.mean(du2dy**4)/std2**2)
    
    return M1, M2  