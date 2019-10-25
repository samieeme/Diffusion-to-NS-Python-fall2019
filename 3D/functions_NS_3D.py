#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:23:26 2019

@author: aliakhavan
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('text', usetex=True)
   
##################################################################

def derivatives(Nnod):
    
    kxx = np.zeros((Nnod,Nnod,Nnod))
    kyy = np.zeros((Nnod,Nnod,Nnod))
    kzz = np.zeros((Nnod,Nnod,Nnod))
     
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2.0:
            kxx[i1,:,:] = i1
        else:
            kxx[i1,:,:] = (i1-Nnod)

    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2.0:
            kyy[:,i2,:] = i2
        else:
            kyy[:,i2,:] = (i2-Nnod)
            
    for i3 in range(Nnod):
        if i3 <= (Nnod-1)/2.0:
            kzz[:,:,i3] = i3
        else:
            kzz[:,:,i3] = (i3-Nnod)

    kx = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    ky = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    kz = np.zeros((Nnod,Nnod,Nnod),dtype=complex)
    
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2.0:
            kx[i1,:,:] = complex(0,i1)
        else:
            kx[i1,:,:] = complex(0,i1-Nnod)   

    for i2 in range(Nnod):
        if i2 <= (Nnod-1)/2.0:
            ky[:,i2,:] = complex(0,i2)
        else:
            ky[:,i2,:] = complex(0,i2-Nnod)
            
    for i3 in range(Nnod):
        if i3 <= (Nnod-1)/2.0:
            ky[:,i2,:] = complex(0,i3)
        else:
            ky[:,i3,:] = complex(0,i3-Nnod)

    return kxx, kyy, kzz, kx, ky, kz
   

def get_diffusion_opt(alpha,dt,nu,Nnod,kxx,kyy,kzz):
    
    identity = np.ones((Nnod,Nnod,Nnod))
    
    frac_L = (kxx**2 + kyy**2 + kzz**2)**alpha
    
    den = identity + 0.5*dt*nu*frac_L
    
    num = identity - 0.5*dt*nu*frac_L
    
    operator_diff = num/den
    
    frac_L[0,0] = 1.0
    frac_R = 1.0/frac_L
    
    return operator_diff, den, frac_R
    

def adv_FE(Nnod, vhat, adv_u1_hat, adv_u2_hat, adv_u3_hat, dt, kx, ky, kz,
           operator_diff, den, af, ndx_f, sz_f):
    
    operator_adv = kx * adv_u1_hat + ky * adv_u2_hat + kz * adv_u3_hat
    
    for i in range(0,sz_f):
        operator_adv[ndx_f[i,0],ndx_f[i,1],ndx_f[i,2]] += af[i]

    solution = operator_diff * vhat + dt*operator_adv/den
    
    return solution    
    
def adv_AB(Nnod, vhat, adv_u1_hat, adv_u2_hat, adv_u3_hat,
           adv_u1_hatold, adv_u2_hatold, adv_u3_hatold, dt, kx, ky, kz,
           operator_diff, den, af, af_old, ndx_f, sz_f):

    operator_adv = kx*adv_u1_hat + ky*adv_u2_hat + kz*adv_u3_hat
    operator_adv_old = kx*adv_u1_hatold + ky*adv_u2_hatold + kz*adv_u3_hatold
    
    for i in range(0,sz_f):
        operator_adv[ndx_f[i,0],ndx_f[i,1],ndx_f[i,2]] += af[i]
        operator_adv_old[ndx_f[i,0],ndx_f[i,1],ndx_f[i,2]] += af_old[i]
    
    solution = operator_diff * vhat + dt*(1.5*operator_adv - 0.5*operator_adv_old)/den
    
    return solution 
    
    
def diff_cont(Nnod, u1_hat, u2_hat, u3_hat, kx, ky, kz, frac_R):
           
    fhat = kx * u1_hat + ky * u2_hat + kz * u3_hat
    
    divhat = - frac_R * fhat

    divhat[0,0,0] = 0.0
    
    return divhat
    
def corrector(Nnod, u1hat_tilde, u2hat_tilde, u3hat_tilde, phat, dt,
              kx, ky, kz):
    
    U1hat_new = u1hat_tilde - kx * phat
    U2hat_new = u2hat_tilde - ky * phat
    U3hat_new = u3hat_tilde - kz * phat
    
    return U1hat_new, U2hat_new, U3hat_new

def dealiasing(cut_off, Nnod):

    Nhalf=int(Nnod/2)
    cf=int(np.round(cut_off*Nnod/2.))
    w = np.ones((Nhalf+1,1))
    cut = np.zeros((Nhalf-cf,1))

    w[cf+1::]=cut
    w_fliped=np.flip(w[1::])

    w=np.append(w,w_fliped)

    cutoff = np.zeros((Nnod,Nnod,Nnod))

    for i3 in range(0,Nnod):
        for i2 in range(0,Nnod):
            for i1 in range(0,Nnod):
                cutoff[i1,i2,i3] = w[i1] * w[i2] * w[i3]

    return cutoff
    
    
def gen_IC_vel(Nnod):

    sz=Nnod**3

    PI=np.pi

    Uhat=np.zeros((sz,3),dtype=complex)

    wave_n=np.array([0.0,0.0,0.0])
    max_wave=int(Nnod/2)
    ndx=0

    for k in range(0,Nnod):
        for j in range(0,Nnod):
            for i in range(0,Nnod):

                wave_n[0]=i
                if i > max_wave:
                    wave_n[0]=i-Nnod
                wave_n[1]=j
                if j > max_wave:
                    wave_n[1]=j-Nnod
                wave_n[2]=k
                if k > max_wave:
                    wave_n[2]=k-Nnod
        
                k_tmp=LA.norm(wave_n, ord=2)
                Esp=np.round(k_tmp)
        
                theta=np.random.uniform(0.0,2*PI,2)
                psi=np.random.uniform(0.0,2*PI)
        
                phs1=np.exp(1j*theta[0])
                phs2=np.exp(1j*theta[1])
        
                Amp=np.sqrt(2.0*np.exp(-2*Esp/3.0)/(3*PI))
                Uhat[ndx,0]=Amp*np.cos(psi)*phs1
                Uhat[ndx,1]=Amp*np.sin(psi)*phs2
        
                ndx +=1

    u1_hat=Uhat[:,0].reshape(Nnod,Nnod,Nnod)
    u2_hat=Uhat[:,1].reshape(Nnod,Nnod,Nnod)
    u3_hat=Uhat[:,2].reshape(Nnod,Nnod,Nnod)

    return u1_hat, u2_hat, u3_hat

def gen_IC_vel1(Nnod, Kf):
    
    sz=Nnod**3

    PI=np.pi
    
    Uhat=np.zeros((sz,3),dtype=complex)
    
    wave_n=np.array([0.0,0.0,0.0])
    max_wave=int(Nnod/2)
    ndx=0
    
    for k in range(0,Nnod):
        for j in range(0,Nnod):
            for i in range(0,Nnod):
            
                wave_n[0]=i
                if i > max_wave:
                    wave_n[0]=i-Nnod
                wave_n[1]=j
                if j > max_wave:
                    wave_n[1]=j-Nnod
                wave_n[2]=k
                if k > max_wave:
                    wave_n[2]=k-Nnod
                
                k_tmp=LA.norm(wave_n, ord=2)
    #            Esp=np.round(k_tmp)
                Esp=k_tmp
                
                theta=np.random.uniform(0.0,2*PI,2)
                psi=np.random.uniform(0.0,2*PI)
                
                phs1=np.exp(1j*theta[0])
                phs2=np.exp(1j*theta[1])
                Amp=1.0/PI
                
                if Esp <= Kf:
                    A1=np.sqrt(Amp*Esp/Kf**3)
                    Uhat[ndx,0]=A1*np.cos(psi)*phs1
                    Uhat[ndx,1]=A1*np.sin(psi)*phs2
                else:
                    A1=np.sqrt(Amp)*Kf/Esp**2
                    Uhat[ndx,0]=A1*np.cos(psi)*phs1
                    Uhat[ndx,1]=A1*np.sin(psi)*phs2
                    
                ndx +=1
    
    u1_hat=Uhat[:,0].reshape(Nnod,Nnod,Nnod)
    u2_hat=Uhat[:,1].reshape(Nnod,Nnod,Nnod)
    u3_hat=Uhat[:,2].reshape(Nnod,Nnod,Nnod)

    return u1_hat, u2_hat, u3_hat

def get_vorticity(Nnod,V1hat,V2hat,kx,ky):
    
    divhat_x = - kx * V2hat
    divhat_y = - ky * V1hat
    Vor = np.real(np.fft.ifft2(divhat_y-divhat_x))
    return Vor

def plot_Vel(X,Y,U,V,n,map_type):


    fig = plt.figure(figsize=(14,11))
    plt.subplot(2,2,1)
    plt.contourf(X,Y,U,100,cmap=map_type)
    plt.title('$u_1(\mathbf{x}),$ $t=$ '+str(n), fontsize=18)
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.contourf(X,Y,V,100,cmap=map_type)
    plt.title('$u_2(\mathbf{x}),$ $t=$ '+str(n), fontsize=18)
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    plt.colorbar()

    plt.show()
    
def plot_Vor(X,Y,Vor,n,icnt,map_type):

    fig = plt.figure(figsize=(6.5,5))
    plt.contourf(X,Y,Vor,100,cmap=map_type)
    plt.title('$\omega_z(\mathbf{x}),$ $t=$ '+str(n), fontsize=18)
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)
    plt.colorbar()

    plt.show()
    
#    plt.savefig('Out_T'+str(icnt)+'.pdf', dpi=None, facecolor='w', edgecolor='w',
#                orientation='portrait', papertype=None, format=None,
#                transparent=False, bbox_inches=None, pad_inches=0.1,
#                metadata=None) 
    
    
    
    
    
    
    
