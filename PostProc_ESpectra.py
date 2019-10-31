#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:21:49 2019

@author: aliakhavan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica']})
rc('text', usetex=True)

from functions_NS_2D import derivatives, get_vorticity
from functions_stats import get_sphere_waven

def plot_Vor(X,Y,Vor,n,it,map_type):

    fig = plt.figure(figsize=(6.5,5))
    plt.contourf(X,Y,Vor.real,100,cmap=map_type)
    plt.title('$\omega_z(\mathbf{x}),$ $t=$'+format(n, '.1f'), fontsize=22)
    plt.xlabel('$x_1$', fontsize=20)
    plt.ylabel('$x_2$', fontsize=20)
    plt.colorbar()    
#    plt.show()
    
    plt.savefig('Vor_T'+str(it)+'.png', dpi=300, facecolor='w', 
                edgecolor='w', orientation='portrait', papertype=None, 
                format=None, transparent=False, bbox_inches=None, 
                pad_inches=0.1, metadata=None) 

#%%############################################################################
###############################################################################

map_type='seismic'

PI=np.pi
slope=-3.0

#icpath=os.path.join(os.getcwd(),'Out_IC')
icpath='/home/akhavans/Documents/Research/Scalar_Turbulence/tests/test4'
os.chdir(icpath)
inps = np.genfromtxt(icpath+'/'+'inps.txt', delimiter=' ')

#%%############################################################################
###############################################################################

res = int(inps[0])
sz = res**2

meshX = np.linspace(0,2*PI,res+1)
meshX = np.delete(meshX,res,None)
X,Y = np.meshgrid(meshX,meshX)

#Computing derivatives' matrices
kxx,kyy,kx,ky = derivatives(res)
K_sh,K_sh2,K_sh4 = get_sphere_waven(res)


nfiles = int(inps[4]/inps[6])

time = np.linspace(0.0,inps[4],nfiles+1)

Esp=np.zeros((sz,nfiles+2))

Esp[:,nfiles+1] = K_sh.reshape(sz)
Esp[0,1]=1.0

for it in range(0,nfiles+1):
    
    fpath = icpath+'/Out_'+str(it)+'_chk/'
    uhat = np.genfromtxt(fpath+'Uhat.csv', delimiter=',',dtype=complex)
    vhat = np.genfromtxt(fpath+'Vhat.csv', delimiter=',',dtype=complex)
    
    Vor=get_vorticity(1,uhat,vhat,kx,ky)
    plot_Vor(X,Y,Vor,time[it],it,map_type)
    
    E_k = 0.5*((uhat*np.conj(uhat) + vhat*np.conj(vhat)).real)
    Esp[:,it] = E_k.reshape(sz)

#%%############################################################################
###############################################################################

Esp_sort = Esp[Esp[:,nfiles+1].argsort()]

nk = int(Esp_sort[-1,nfiles+1])

Energy_Spct = np.zeros((nk,nfiles+1))

for j in range(0,nk):
    i=j+1
    tmp1=np.nonzero(Esp_sort[:,nfiles+1]==i)
    tmp2=tmp1[0]
    Energy_Spct[j,:]=np.mean(Esp_sort[tmp2[0]:tmp2[-1]+1,0:nfiles+1],axis=0)
    
    if i == 1:
        area=(1.5**2)*PI
    else:
        area=((i+0.5)**2-(i-0.5)**2)*PI
        
    Energy_Spct[j,:] *= area
    
    
sz_Esp = int(res*2**.5/3.)
Energy_Spct=Energy_Spct[0:sz_Esp,:]
W_rad=np.linspace(1,sz_Esp,sz_Esp)

#%%############################################################################
###############################################################################

fig = plt.figure(figsize=(10,7.5))
 
for i in range(0,nfiles+1):
    plt.loglog(W_rad,Energy_Spct[:,i], linewidth=2, label='$t=$'+str(time[i]))
plt.loglog(W_rad[3:40],W_rad[3:40]**slope, 'k', linestyle='--',
           linewidth=1.5, label='$\mathbf{\kappa}^{-3}$')
plt.title('Energy Spectrum', fontsize=30)
plt.xlabel('$\mathbf{\kappa}$', fontsize=28)
plt.ylabel('$E(\mathbf{\kappa},t)$', fontsize=28)
plt.legend(fontsize=18)
plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
#plt.show()

plt.savefig('EngSpecrum.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None) 

#%%############################################################################
###############################################################################

M_vor = np.genfromtxt('Vorticity_moments.txt', delimiter=' ')

fig = plt.figure(figsize=(10,7.5))

plt.plot(M_vor[:,0],M_vor[:,1], linewidth=2, label='$\mu_2$')
plt.plot(M_vor[:,0],M_vor[:,2], linewidth=2, label='$\mu_3/\mu_2^{3/2}$')
plt.plot(M_vor[:,0],M_vor[:,3], linewidth=2, label='$\mu_4/\mu_2^2$')

plt.title('Vorticity Moments', fontsize=30)
plt.xlabel('Time (sec)', fontsize=28)
plt.ylabel('$\mu_n$', fontsize=28)
plt.legend(fontsize=18)
plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
#plt.show()

plt.savefig('Moments.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None) 

#%%############################################################################
###############################################################################

FF = np.genfromtxt('FlowFeatures.txt', delimiter=' ')

fig = plt.figure(figsize=(10,7.5))

plt.plot(FF[:,0],FF[:,1]/FF[0,1], linestyle='--', 
         linewidth=2, label='T.K.E., $K/K_0$')
plt.plot(FF[:,0],FF[:,2]/FF[0,2], linewidth=2, 
         label='Dissipation, $\epsilon/\epsilon_0$')

plt.title('Flow Features', fontsize=30)
plt.xlabel('Time (sec)', fontsize=24)
plt.ylabel('$K(t)/K_0$, $\epsilon(t)/\epsilon_0$', fontsize=24)
plt.legend(fontsize=18)
plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
#plt.show()

plt.savefig('FlowFeatures.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None) 



