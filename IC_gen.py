#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 03:31:07 2019

@author: akhavans
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica']})
rc('text', usetex=True)

def deriv_x(Nnod,Vhat):

    kx = np.zeros((Nnod,Nnod),dtype=complex)
    
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           kx[i1,:] = complex(0,i1)
        else:
           kx[i1,:] = complex(0,i1-Nnod)  
           
    divhat = - kx * Vhat
    diver_V = np.real(np.fft.ifftn(divhat))
    diver_V = diver_V.reshape(Nnod,Nnod)
    return diver_V


def deriv_y(Nnod,Vhat):

    ky = np.zeros((Nnod,Nnod),dtype=complex)
    
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           ky[:,i1] = complex(0,i1)
        else: 
           ky[:,i1] = complex(0,i1-Nnod) 
    
    divhat = - ky * Vhat
    diver_V = np.real(np.fft.ifftn(divhat))
    diver_V = diver_V.reshape(Nnod,Nnod)
    return diver_V

def get_vorticity(Nnod,V1hat,V2hat):

    kx = np.zeros((Nnod,Nnod),dtype=complex)
    ky = np.zeros((Nnod,Nnod),dtype=complex)
    
    for i1 in range(Nnod):
        if i1 <= (Nnod-1)/2:
           kx[i1,:] = complex(0,i1)
           ky[:,i1] = complex(0,i1)
        else:
           kx[i1,:] = complex(0,i1-Nnod)  
           ky[:,i1] = complex(0,i1-Nnod) 
    
    divhat_x = - kx * V2hat
    divhat_y = - ky * V1hat
    diverx_V = np.real(np.fft.ifftn(divhat_y-divhat_x))
    diverx_V = diverx_V.reshape(Nnod,Nnod)
    return diverx_V

def Moments(res,sz,u1_w2,u2_w2):
    
    du1dx=deriv_x(res,u1_w2)
    du1dx=du1dx.reshape(sz,1)
    
    du2dy=deriv_y(res,u2_w2)
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


map_type='seismic'
Ks=2
res=512
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
        Amp=1.0/PI
        
        if Esp <= Ks:
            A1=np.sqrt(Amp*Esp/Ks**3)
            u_w[ndx,0]=A1*np.cos(psi)*phs1
            u_w[ndx,1]=A1*np.sin(psi)*phs2
        else:
            A1=np.sqrt(Amp)*Ks/Esp**2
            u_w[ndx,0]=A1*np.cos(psi)*phs1
            u_w[ndx,1]=A1*np.sin(psi)*phs2
            
        ndx +=1

u1_w2=u_w[:,0].reshape(res,res)
u2_w2=u_w[:,1].reshape(res,res)


u=np.fft.ifftn(u1_w2)
v=np.fft.ifftn(u2_w2)

u=u.real
v=v.real
vor=get_vorticity(res,u1_w2,u2_w2)

M1, M2 = Moments(res,sz,u1_w2,u2_w2)

x=np.linspace(0,2*PI,res)
X,Y=np.meshgrid(x,x)

fig = plt.figure(figsize=(18,14))
plt.subplot(2,2,1)
plt.contourf(X,Y,u,100,cmap=map_type)
plt.title('$u_1(\mathbf{x},t_0)$', fontsize=22)
plt.xlabel('$x_1$', fontsize=18)
plt.ylabel('$x_2$', fontsize=18)
plt.colorbar()

plt.subplot(2,2,2)
plt.contourf(X,Y,v,100,cmap=map_type)
plt.title('$u_2(\mathbf{x},t_0)$', fontsize=22)
plt.xlabel('$x_1$', fontsize=18)
plt.ylabel('$x_2$', fontsize=18)
plt.colorbar()

###############################################################
###############################################################

#Taking FFT from the velocity components
u=u.reshape(sz,1)
v=v.reshape(sz,1)

u1hat = np.fft.fftn(u)
u2hat = np.fft.fftn(v)

u1hat=u1hat.reshape(res,res)

u2hat=u2hat.reshape(res,res)

Esp=np.zeros((sz,2))

tmp_U=np.zeros((1,2), dtype=complex)
ndx=0

for j in range(0,res):
    for i in range(0,res):

        tmp_U[0,0]=u1hat[i,j]
        tmp_U[0,1]=u2hat[i,j]

        U_mag=LA.norm(tmp_U, ord=2)
        Esp[ndx,0]=0.5*U_mag**2
        
        wave_n[0]=i
        if i > max_wave:
            wave_n[0]=i-res
        wave_n[1]=j
        if j > max_wave:
            wave_n[1]=j-res


        Esp[ndx,1]=np.round(LA.norm(wave_n, ord=2))
        ndx +=1

Esp_sort = Esp[Esp[:,1].argsort()]

del Esp

Esp_sort[:,1][Esp_sort[:,1]==0.0]=1.0
Energy_Spct = np.zeros(int(Esp_sort[-1,1]))
#Intg_scale = 0.0

for j in range(0,Energy_Spct.size):
    i=j+1
    tmp1=np.nonzero(Esp_sort[:,1]==i)
    tmp2=tmp1[0]
    Energy_Spct[j]=np.mean(Esp_sort[tmp2[0]:tmp2[-1]+1,0])
    
    if i == 1:
        area=(1.5**2)*PI
    else:
        area=((i+0.5)**2-(i-0.5)**2)*PI
        
    Energy_Spct[j] *= area
    
#    Intg_scale +=Energy_Spct[j]/i 
    
sz_Esp = int(np.round(res*2**.5/3.))
Energy_Spct=Energy_Spct[0:sz_Esp]
W_rad=np.linspace(1,sz_Esp,sz_Esp)


plt.subplot(2,2,3)
plt.contourf(X,Y,vor,100,cmap=map_type)
plt.title('$\omega_z(\mathbf{x},t_0)$', fontsize=22)
plt.xlabel('$x_1$', fontsize=18)
plt.ylabel('$x_2$', fontsize=18)
plt.colorbar()

plt.subplot(2,2,4)
plt.loglog(W_rad,Energy_Spct, 'b', linewidth=2, label='$E(\mathbf{\kappa})$')
plt.loglog(W_rad[5:41],W_rad[5:41]**(-3), 'r', linestyle='--',
           linewidth=1.5, label='$\mathbf{\kappa}^{-3}$')
plt.title('\it Energy Spectrum for the I.C.', fontsize=20)
plt.xlabel('$\mathbf{\kappa}$', fontsize=18)
plt.ylabel('$E(\mathbf{\kappa},t_0)$', fontsize=18)
plt.legend(fontsize=15)
plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
#plt.show()

plt.savefig('IC_'+str(res)+'_Kf'+str(Ks)+'.pdf', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None) 

##Integral length scale    
#Intg_scale /= np.sum(Energy_Spct)
#
##Large eddy turnover time
#T_L = Intg_scale/U_rms
#    
##Output statistical characteristics    
#out=np.array([E_tot, U_rms, epsilon, lamb, Re_T, tau_eta, eta, Intg_scale, T_L])
#np.savetxt('Stats_'+time+'.dat', out, delimiter=',')

#Output the modal energy
#np.savetxt('MEng_'+time+'.dat', Energy_Spct, delimiter=',')