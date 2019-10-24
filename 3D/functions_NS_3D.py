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
    
    frac_L = (kxx**2+kyy**2+kzz**2)**(alpha)
    
    den = identity+0.5*dt*nu*frac_L
    
    num = identity-0.5*dt*nu*frac_L
    
    operator_diff = num/den
    
    frac_L[0,0] = 1.0
    frac_R = 1.0/frac_L
    
    return operator_diff, den, frac_R
    

def adv_FE(Nnod,vhat,adv_velx_hat,adv_vely_hat,adv_velz_hat,dt,kx,ky,kz,
           operator_diff,den,af,ndx_f,sz_f):
    
    operator_adv = kx*adv_velx_hat+ky*adv_vely_hat+kz*adv_velz_hat
    
    for i in range(0,sz_f):
        operator_adv[ndx_f[i,0],ndx_f[i,1],ndx_f[i,2]] += af[i]

    solution = operator_diff * vhat + dt*operator_adv/den
    
    return solution    
    
def adv_AB(Nnod,vhat,adv_velx_hat,adv_vely_hat,adv_velz_hat,
           adv_velx_hatold,adv_vely_hatold,adv_velz_hatold,dt,kx,ky,kz,
           operator_diff,den,af,af_old,ndx_f,sz_f):

    operator_adv = kx*adv_velx_hat + ky*adv_vely_hat + kz*adv_velz_hat
    operator_adv_old = kx*adv_velx_hatold + ky*adv_vely_hatold + kz*adv_velz_hatold
    
    for i in range(0,sz_f):
        operator_adv[ndx_f[i,0],ndx_f[i,1],ndx_f[i,2]] += af[i]
        operator_adv_old[ndx_f[i,0],ndx_f[i,1],ndx_f[i,2]] += af_old[i]
    
    solution = operator_diff * vhat + dt*(1.5*operator_adv - 0.5*operator_adv_old)/den
    
    return solution 
    
    
def diff_cont(Nnod,uhat,vhat,what,kx,ky,kz,frac_R):
           
    fhat = kx * uhat + ky * vhat + kz * what
    
    divhat = frac_R * fhat

    divhat[0,0] = 0.0
    
    return divhat
    
def corrector(Nnod,Uhat_tilde,Vhat_tilde,What_tilde,phat,dt,kx,ky,kz):
    
    Uhat_new = Uhat_tilde + kx * phat
    Vhat_new = Vhat_tilde + ky * phat
    What_new = What_tilde + kz * phat
    
    return Uhat_new, Vhat_new, What_new

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
    
    
def gen_IC_vel(res):

    sz=res**2

    PI=np.pi

    u_w=np.zeros((sz,2),dtype=complex)

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

            k_tmp=LA.norm(wave_n, ord=2)
            Esp=np.round(k_tmp)

            theta=np.random.uniform(0.0,2*PI,2)
            psi=np.random.uniform(0.0,2*PI)

            phs1=np.exp(1j*theta[0])
            phs2=np.exp(1j*theta[1])

            Amp=np.sqrt(2.0*np.exp(-2*Esp/3.0)/(3*PI))
            u_w[ndx,0]=Amp*np.cos(psi)*phs1
            u_w[ndx,1]=Amp*np.sin(psi)*phs2

            ndx +=1

    u1_w2=u_w[:,0].reshape(res,res)
    u2_w2=u_w[:,1].reshape(res,res)


    return u1_w2, u2_w2

def gen_IC_vel1(res, Kf):
    
    sz=res**2

    PI=np.pi
    
    u_w=np.zeros((sz,2),dtype=complex)
    
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
                u_w[ndx,0]=A1*np.cos(psi)*phs1
                u_w[ndx,1]=A1*np.sin(psi)*phs2
            else:
                A1=np.sqrt(Amp)*Kf/Esp**2
                u_w[ndx,0]=A1*np.cos(psi)*phs1
                u_w[ndx,1]=A1*np.sin(psi)*phs2
                
            ndx +=1
    
    u1_w2=u_w[:,0].reshape(res,res)
    u2_w2=u_w[:,1].reshape(res,res)
    
    
    return u1_w2, u2_w2

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
    
    
    
    
    
    
    
