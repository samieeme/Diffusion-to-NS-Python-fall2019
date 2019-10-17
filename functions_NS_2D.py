# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:57:08 2019

@author: samieeme
"""

import numpy as np
def forcef(x,y):
    return -4.0*np.sin(2.0*x)*np.cos(3.0*y)-9.0*np.sin(2.0*x)*np.cos(3.0*y)
def forcef_t(x,y,t,alpha,nu):
    po = 3.0  
    return np.sin(2.0*x)*np.cos(3.0*y)*(po)*t**(po-1)-alpha*(t**(po))*(-4.0*np.sin(2.0*x)*np.cos(3.0*y)-9.0*np.sin(2.0*x)*np.cos(3.0*y))-Ux_t(x,y,t,nu)*2.0*np.cos(2.0*x)*np.cos(3.0*y)*t**(po)+Uy_t(x,y,t,nu)*3.0*np.sin(2.0*x)*np.sin(3.0*y)*t**(po)


def solV(x,y):
    return np.sin(2.0*x)*np.cos(3.0*y)
def phi_t(x,y,t):
#    return np.sin(2.0*x)*np.cos(3.0*y)*np.exp(-2*t)
    return np.sin(2.0*x)*np.cos(3.0*y)*t**(3.0) #np.sin(2.0*x)*np.exp(-10*t)
 #   return np.sin(2.0*x)*np.cos(3.0*y)


def Ux_t(x,y,t,nu):
    return -np.cos(1.0*x)*np.sin(1.0*y)*np.exp(-2.0*t)#(1.0+2.0*nu*t) #np.sin(2.0*x)*np.exp(-10*t)
def Uy_t(x,y,t,nu):
    return np.sin(1.0*x)*np.cos(1.0*y)*np.exp(-2.0*t)#(1.0+2.0*nu*t) #np.sin(2.0*x)*np.exp(-10*t)
def press_t(x,y,t,nu):
    return -1.0/4.0*(np.cos(2.0*x)+np.cos(2.0*y))*np.exp(-4.0*t)#(1.0+2.0*nu*t)**2.0



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

def diff_FE(Nnod,fhat,vhat,diffusivity,dt):
    #frac_L = np.zeros((Nnod,Nnod))
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
    bb = 0.5
    identity = np.ones((Nnod,Nnod))
    operator_diff = (identity[:]-diffusivity*dt*(1.0-bb)*(kx[:]**2+ky[:]**2)**(alpha))/(identity[:]+diffusivity*dt*bb*(kx[:]**2+ky[:]**2)**(alpha))
    operator_force = dt*(identity[:])/(identity[:]+diffusivity*dt*bb*(kx[:]**2+ky[:]**2)**(alpha))
    
    solution = operator_diff * vhat + operator_force * fhat 
    return solution
    
    
    
def diff_AB(Nnod,fhat,fhat_old,vhat,vhat_old,diffusivity,dt):
    #frac_L = np.zeros((Nnod,Nnod))
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
    bb = 0.5
    identity = np.ones((Nnod,Nnod))
    operator_diff = (identity[:]-diffusivity*dt*(1.0-bb)*(kx[:]**2+ky[:]**2)**(alpha))/(identity[:]+diffusivity*dt*bb*(kx[:]**2+ky[:]**2)**(alpha))
    operator_force = dt*(identity[:])/(identity[:]+diffusivity*dt*bb*(kx[:]**2+ky[:]**2)**(alpha))
    
    solution = operator_diff * vhat + 3.0/2.0*operator_force * fhat - 1.0/2.0*operator_force * fhat_old 
    return solution    


def adv_FE(Nnod,fhat,vhat,adv_velx_hat,adv_vely_hat,diffusivity,dt):
    #frac_L = np.zeros((Nnod,Nnod))
    kxx = np.zeros((Nnod,Nnod))
    kyy = np.zeros((Nnod,Nnod))
   # kx[0,0]=1.0/10000000.0
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
            kxx[i1,:] = i1
        else:
            kxx[i1,:] = (i1-Nnod)
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
            kyy[:,i2] = i2
        else:
            kyy[:,i2] = (i2-Nnod)
    
    kx = np.zeros((Nnod,Nnod),dtype=complex)
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           kx[i1,:] = complex(0,i1)
        else:
           kx[i1,:] = complex(0,i1-Nnod)   
    ky = np.zeros((Nnod,Nnod),dtype=complex)
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
           ky[:,i2] = complex(0,i2)
        else:
           ky[:,i2] = complex(0,i2-Nnod)                  
    alpha = 1;
    bb = 0.5
    identity = np.ones((Nnod,Nnod))
    operator_diff = (identity[:]-diffusivity*dt*(1.0-bb)*(kxx[:]**2+kyy[:]**2)**(alpha))/(identity[:]+diffusivity*dt*bb*(kxx[:]**2+kyy[:]**2)**(alpha))
    operator_force = dt*(identity[:])/(identity[:]+diffusivity*dt*bb*(kxx[:]**2+kyy[:]**2)**(alpha))
    operator_adv = dt*(kx[:]*adv_velx_hat[:]+ky[:]*adv_vely_hat[:])/(identity[:]+diffusivity*dt*bb*(kxx[:]**2+kyy[:]**2)**(alpha))
    
    solution = operator_diff * vhat + operator_force * fhat + 1.0*operator_adv
    return solution    
    
def adv_AB(Nnod,fhat,fhat_old,vhat,adv_velx_hat,adv_vely_hat,adv_velx_hatold,adv_vely_hatold,diffusivity,dt):
    #frac_L = np.zeros((Nnod,Nnod))
    kxx = np.zeros((Nnod,Nnod))
    kyy = np.zeros((Nnod,Nnod))
    #kx[0,0]=1.0/10000000.0
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
            kxx[i1,:] = i1
        else:
            kxx[i1,:] = (i1-Nnod)
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
            kyy[:,i2] = i2
        else:
            kyy[:,i2] = (i2-Nnod)
    
    kx = np.zeros((Nnod,Nnod),dtype=complex)
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           kx[i1,:] = complex(0,i1)
        else:
           kx[i1,:] = complex(0,i1-Nnod)   
    ky = np.zeros((Nnod,Nnod),dtype=complex)
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
           ky[:,i2] = complex(0,i2)
        else:
           ky[:,i2] = complex(0,i2-Nnod)    
    alpha = 1;
    bb = 0.5
    identity = np.ones((Nnod,Nnod))
    operator_diff = (identity[:]-diffusivity*dt*(1.0-bb)*(kxx[:]**2+kyy[:]**2)**(alpha))/(identity[:]+diffusivity*dt*bb*(kxx[:]**2+kyy[:]**2)**(alpha))
    operator_force = dt*(identity[:])/(identity[:]+diffusivity*dt*bb*(kxx[:]**2+kyy[:]**2)**(alpha))
    operator_adv = dt*(kx[:]*adv_velx_hat[:]+ky[:]*adv_vely_hat[:])/(identity[:]+diffusivity*dt*bb*(kxx[:]**2+kyy[:]**2)**(alpha))
    operator_adv_old = dt*(kx[:]*adv_velx_hatold[:]+ky[:]*adv_vely_hatold[:])/(identity[:]+diffusivity*dt*bb*(kxx[:]**2+kyy[:]**2)**(alpha))
    
    solution = operator_diff * vhat + 3.0/2.0*operator_force * fhat - 1.0/2.0*operator_force * fhat_old + 3.0/2.0*operator_adv - 1.0/2.0*operator_adv_old
    return solution 
    
    
def diff_cont(Nnod,uhat,vhat):
    divhat = np.zeros((Nnod,Nnod))
    kxx = np.zeros((Nnod,Nnod))
    kyy = np.zeros((Nnod,Nnod))
   # kx[0,0]=1.0/10000000.0
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
            kxx[i1,:] = i1
        else:
            kxx[i1,:] = (i1-Nnod)
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
            kyy[:,i2] = i2
        else:
            kyy[:,i2] = (i2-Nnod)
    kx = np.zeros((Nnod,Nnod),dtype=complex)
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           kx[i1,:] = complex(0,i1)
        else:
           kx[i1,:] = complex(0,i1-Nnod)   
    ky = np.zeros((Nnod,Nnod),dtype=complex)
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
           ky[:,i2] = complex(0,i2)
        else:
           ky[:,i2] = complex(0,i2-Nnod)    
           
    alpha = 1;
    frac_L = (kxx[:]**2+kyy[:]**2)**(alpha)
    frac_L[0,0]=1.0
    frac_R=1.0/frac_L
    fhat = kx * uhat + ky * vhat
    
    divhat = frac_R * fhat
   # diverz_V = np.real(np.fft.ifft2(divhat))
    divhat[0,0] = 0.0
    return divhat
    
def corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt):
    kx = np.zeros((Nnod,Nnod),dtype=complex)
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           kx[i1,:] = complex(0,i1)
        else:
           kx[i1,:] = complex(0,i1-Nnod)   
    ky = np.zeros((Nnod,Nnod),dtype=complex)
    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2:
           ky[:,i2] = complex(0,i2)
        else:
           ky[:,i2] = complex(0,i2-Nnod)     
    
    uhatnew = Uhat_tilde[:] -  (-1.0*kx[:]) * phat[:]
    vhatnew = Vhat_tilde[:] -  (-1.0*ky[:]) * phat[:]
    
    return uhatnew, vhatnew
    

def dealiasing(cut_off, Nnod):
    
    Nhalf=int(Nnod/2)
    cf=int(np.round(cut_off*Nnod/2.))
    w = np.ones((Nhalf+1,1))
    cut = np.zeros((Nhalf-cf,1))
    
    w[cf+1::]=cut
    w_fliped=np.flip(w[1::])
    
    w=np.append(w,w_fliped)
    
    cutoff = np.zeros((Nnod,Nnod))
    
    for i2 in range(0,Nnod):
        for i1 in range(0,Nnod):
            cutoff[i1,i2] = w[i1]*w[i2]
                
    return cutoff    
    
    
    
    
    
    
    
