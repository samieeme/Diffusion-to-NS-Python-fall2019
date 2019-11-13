#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:47:59 2019

@author: aliakhavan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica']})
rc('text', usetex=True)

plt.rc('xtick', labelsize=18) 
plt.rc('ytick', labelsize=18)

from functions_NS_2D import derivatives, get_vorticity
from functions_NS_2D import plot_Phi, plot_Vor 
from functions_stats import get_sphere_waven, Moments_Vor

#%%############################################################################

def get_PDF(Q_in,Nbins):
    
    Qmax=np.max(Q_in)
    Qmin=np.min(Q_in)
    
    sz=Q_in.size
    
    DQ=(Qmax-Qmin)/(Nbins-1)   
    DQ_half=DQ/2.0
    
    interv=np.array(Qmin)
    interv=np.append(interv,np.linspace(Qmin+DQ_half,Qmax-DQ_half,Nbins-1))
    interv=np.append(interv,Qmax)
    
    bins=np.linspace(Qmin,Qmax,Nbins)
    pdf=np.zeros(Nbins)
    
    for i in range(0,Nbins-1):
        pdf[i]=np.nonzero((interv[i] <= Q_in) & (Q_in < interv[i+1]))[0].size
        
    pdf[Nbins-1]=np.nonzero((interv[Nbins-1] <= Q_in) & 
       (Q_in <= interv[Nbins]))[0].size

    pdf=pdf/sz
    
    return pdf, bins

###############################################################################
    
def read_data(Ks,alpha,Sc):
    
    pth = 'Ks_'+str(Ks)+'_alpha_'+str(alpha)+'_Sc_'+str(Sc)
    dirpath = os.path.join(os.getcwd(), pth)
    os.chdir(dirpath)
    
    inps = np.genfromtxt('inps.txt', delimiter=' ')
    
###############################################################################
    
    res = int(inps[0])
    sz = res**2

    nfiles = int(inps[6]*inps[8])
    
    time = np.linspace(0.0,inps[6],nfiles+1)
    
    Mnt_phi=np.array([])
    
    kxx,kyy,kx,ky = derivatives(res)
    
    for it in range(0,nfiles+1):
        
        tmp_pth = 'Out_'+str(it)+'_chk'
        fpath = os.path.join(dirpath, tmp_pth)
        
#        uhat = np.genfromtxt(os.path.join(fpath,'Uhat.csv'), 
#                             delimiter=',',dtype=complex)
#        vhat = np.genfromtxt(os.path.join(fpath,'Vhat.csv'), 
#                             delimiter=',',dtype=complex)
        phihat = np.genfromtxt(os.path.join(fpath,'Phihat.csv'), 
                               delimiter=',',dtype=complex)
        
        phi = np.fft.ifftn(phihat)*sz
        
        Mnt_phi=np.append(Mnt_phi,Moments_Vor(phi))
        
        pdf,bins=get_PDF(np.real(phi.reshape(sz))/(Mnt_phi[-3])**0.5,300)
        
        plt.figure(figsize=(9,7.5))
        plt.plot(bins,pdf)
        plt.title('$\\alpha=$ '+format(2*alpha,'.1f')+' --- $t=$ '+
                  format(time[it],'.1f'),fontsize=26)
        plt.xlabel('$\phi^{\prime}/\sigma$',fontsize=22)
        plt.ylabel('PDF',fontsize=22)
        plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
#        plt.show()
        
        plt.savefig('PDF_phi_'+str(it)+'.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None) 
        
    
    Mnt_phi=Mnt_phi.reshape(nfiles+1,3)
    
    os.chdir('../')
    
    return Mnt_phi
    
#%%############################################################################
    
def comp_spectra(Ks,alpha,Sc,if_plot,cutoff):
        
    PI=np.pi
    
    pth = 'Ks_'+str(Ks)+'_alpha_'+str(alpha)+'_Sc_'+str(Sc)
    dirpath = os.path.join(os.getcwd(), pth)
    os.chdir(dirpath)
    
    inps = np.genfromtxt('inps.txt', delimiter=' ')
    
###############################################################################
    
    res = int(inps[0])
    sz = res**2
    
    K_sh,K_sh2,K_sh4 = get_sphere_waven(res)
    
    nu = float(inps[1])
    nfiles = int(inps[6]*inps[8])
    
    time = np.linspace(0.0,inps[6],nfiles+1)
    
    Esp=np.zeros((sz,nfiles+2))
    EspD=np.zeros((sz,nfiles+2))
    
    Esp[:,nfiles+1] = K_sh.reshape(sz)
    EspD[:,nfiles+1] = K_sh.reshape(sz)
    
    counter = 0
    
    if if_plot:
        kxx,kyy,kx,ky = derivatives(res)
        meshX = np.linspace(0,2*PI,res+1)
        meshX = np.delete(meshX,res,None)
        X,Y = np.meshgrid(meshX,meshX)
    
    for it in range(0,nfiles+1):
        
        tmp_pth = 'Out_'+str(it)+'_chk'
        fpath = os.path.join(dirpath, tmp_pth)
        
        uhat = np.genfromtxt(os.path.join(fpath,'Uhat.csv'), 
                             delimiter=',',dtype=complex)
        vhat = np.genfromtxt(os.path.join(fpath,'Vhat.csv'), 
                             delimiter=',',dtype=complex)
        phihat = np.genfromtxt(os.path.join(fpath,'Phihat.csv'), 
                               delimiter=',',dtype=complex)
        
        phi = np.fft.ifftn(phihat)*sz
        
        if if_plot:
            Vor=get_vorticity(sz,uhat,vhat,kx,ky)
            plot_Vor(X,Y,Vor,time[it],counter,'rainbow')
            plot_Phi(X,Y,phi,time[it],counter,'coolwarm',2*alpha)
        
        E_k = (phihat*np.conj(phihat)).real
        D_k = (nu/Sc)*K_sh2*E_k
        
        Esp[:,it] = E_k.reshape(sz)
        EspD[:,it] = D_k.reshape(sz)
        
        counter += 1

###############################################################################
    
    Esp_sort = Esp[Esp[:,nfiles+1].argsort()]
    EspD_sort = EspD[EspD[:,nfiles+1].argsort()]
    
    nk = int(Esp_sort[-1,nfiles+1])
    
    E_phi = np.zeros((nk,nfiles+1))
    D_phi = np.zeros((nk,nfiles+1))
    
    for j in range(0,nk):
    
        i=j+1
    
        tmp1=np.nonzero(Esp_sort[:,nfiles+1]==i)
        tmp2=tmp1[0]
    
        E_phi[j,:]=np.mean(Esp_sort[tmp2[0]:tmp2[-1]+1,0:nfiles+1],axis=0)
        D_phi[j,:]=np.mean(EspD_sort[tmp2[0]:tmp2[-1]+1,0:nfiles+1],axis=0)
                
        if i == 1:
            area=(1.5**2)*PI
        else:
            area=((i+0.5)**2-(i-0.5)**2)*PI
            
        E_phi[j,:] *= area
        D_phi[j,:] *= area
    
    E_phi_tot=np.sum(E_phi[:,0:nfiles+1],axis=0)
    D_phi_tot=np.sum(D_phi[:,0:nfiles+1],axis=0)
    
    T_scale_mix = E_phi_tot/D_phi_tot  
        
    sz_Esp = int(res*cutoff)
    
    
    E_phi=E_phi[0:sz_Esp,:]
    D_phi=D_phi[0:sz_Esp,:]
    
    W_rad=np.linspace(1,sz_Esp,sz_Esp)
    
    os.chdir('../')
    
    return nfiles,time,W_rad,E_phi,D_phi,E_phi_tot,D_phi_tot,T_scale_mix

def plot_Spectra(out,E_phi,D_phi,W_rad,time,nfiles,Ks,Sc,alpha):
    
    fig = plt.figure(figsize=(12,7.5))
 
    for i in range(1,nfiles+1,2):
        plt.loglog(W_rad,E_phi[:,i], linewidth=2, label='$t=$'+str(time[i]))
    
    plt.title('$\\textbf{Scalar Variance Spectra}$ --- $Sc=$ '+str(Sc)+
              ', $\kappa_s=$ '+str(Ks)+', $\\alpha=$ '+format(2*alpha,'.1f'), 
              fontsize=24)
    plt.xlabel('$\mathbf{\kappa}$', fontsize=24)
    plt.ylabel('$E_{\phi^{\prime}}(\mathbf{\kappa},t)$', fontsize=24)
    plt.legend(fontsize=16)
    plt.grid(which='both',axis='both',color='grey', linestyle='--',
             linewidth=.3)
    #plt.show()
    
    plt.savefig('VarSpec_'+out+'.png', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            metadata=None) 
    
    ###########################################################################
    
    fig = plt.figure(figsize=(12,7.5))
     
    for i in range(1,nfiles+1,2):
        plt.loglog(W_rad,D_phi[:,i], linewidth=2, label='$t=$'+str(time[i]))
    
    plt.title('$\\textbf{Scalar Dissipation Spectra}$ --- $Sc=$ '+str(Sc)+
              ', $\kappa_s=$ '+str(Ks)+', $\\alpha=$ '+format(2*alpha,'.1f'), 
              fontsize=24)
    plt.xlabel('$\mathbf{\kappa}$', fontsize=24)
    plt.ylabel('$D_{\phi^{\prime}}(\mathbf{\kappa},t)$', fontsize=24)
    plt.legend(fontsize=16)
    plt.grid(which='both',axis='both',color='grey', linestyle='--',
             linewidth=.3)
    #plt.show()
    
    plt.savefig('DissSpec_'+out+'.png', dpi=300, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            metadata=None) 

#%%############################################################################
###############################################################################
    
if_plot=0
cutoff=0.6

Sc = 0.125
Ks = 8

alpha = 1.0
nfiles,time,W_rad,E_1,D_1,Var_1,Eps_1,Tmix_1 = comp_spectra(
        Ks,alpha,Sc,if_plot,cutoff)
out=6
plot_Spectra(str(out),E_1,D_1,W_rad,time,nfiles,Ks,Sc,alpha)


alpha = 0.75
nfiles,time,W_rad,E_2,D_2,Var_2,Eps_2,Tmix_2 = comp_spectra(
        Ks,alpha,Sc,if_plot,cutoff)
out+=1
plot_Spectra(str(out),E_2,D_2,W_rad,time,nfiles,Ks,Sc,alpha)

#Sc = 1.0
#alpha = 1.0
#nfiles,time,W_rad,E_3,D_3,Var_3,Eps_3,Tmix_3 = comp_spectra(
#        Ks,alpha,Sc,if_plot,cutoff)
#out+=1
#plot_Spectra(str(out),E_3,D_3,W_rad,time,nfiles,Ks,Sc,alpha)

##%%############################################################################
################################################################################

fig = plt.figure(figsize=(10,7.5))

plt.semilogy(time,Tmix_1, marker='o', color='b', linestyle='-', linewidth=2, 
             label='$\\alpha=2.0$')
plt.semilogy(time,Tmix_2, marker='v', color='r', linestyle='-', linewidth=2, 
             label='$\\alpha=1.5$')

plt.title('$\\textbf{Scalar Mixing Time-scale}$ --- $\kappa_s=$ '+
          str(Ks)+', $Sc=1/8$', fontsize=24)
plt.xlabel('Time (sec)', fontsize=22)
plt.ylabel('$\\tau_{\phi^{\prime}}$', fontsize=24)
plt.legend(fontsize=15)
plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
#plt.show()

plt.savefig('Ks'+str(Ks)+'_tmix.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None) 

###############################################################################

fig = plt.figure(figsize=(10,7.5))

plt.plot(time,Var_1, marker='o', color='b',linestyle='-', linewidth=2, 
         label='$\\alpha=2.0$')
plt.plot(time,Var_2, marker='v', color='r',linestyle='-', linewidth=2, 
         label='$\\alpha=1.5$')

plt.title('$\\textbf{Scalar Variance}$ --- $\kappa_s=$ '+
          str(Ks)+', $Sc=1/8$', fontsize=24)
plt.xlabel('Time (sec)', fontsize=22)
plt.ylabel('$\left \langle {\phi^{\prime \ }}^2 \\right \\rangle$',
           fontsize=22)
plt.legend(fontsize=15)
plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
#plt.show()

plt.savefig('Ks'+str(Ks)+'_Var.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None) 

###############################################################################

fig = plt.figure(figsize=(10,7.5))

plt.semilogy(time,Eps_1, marker='o', color='b',linestyle='-', linewidth=2, 
             label='$\\alpha=2.0$')
plt.semilogy(time,Eps_2, marker='v', color='r',linestyle='-', linewidth=2, 
             label='$\\alpha=1.5$')

plt.title('$\\textbf{Scalar Dissipation}$ --- $\kappa_s=$ '+
          str(Ks)+', $Sc=1/8$', fontsize=24)
plt.xlabel('Time (sec)', fontsize=22)
plt.ylabel('$\left \langle \\varepsilon_{\phi^{\prime}} \\right \\rangle$',
           fontsize=24)
plt.legend(fontsize=15)
plt.grid(which='both',axis='both',color='grey', linestyle='--', linewidth=.3)
#plt.show()

plt.savefig('Ks'+str(Ks)+'_Diss.png', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        metadata=None) 

#%%############################################################################
###############################################################################