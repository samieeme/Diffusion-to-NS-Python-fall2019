#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:38:37 2019

@author: aliakhavan
"""

import numpy as np
#from numpy import linalg as LA
import sys
import os

from functions_NS_2D import derivatives, get_diffusion_opt
from functions_NS_2D import adv_FE, adv_AB, diff_cont, corrector, dealiasing
from functions_stats import get_sphere_waven, get_stats_eng
from PTM import init_particle, Particle_Tracking

#%%###################### Setting up parameters ###############################

Nnod_in = sys.argv[1]
Nnod = int(Nnod_in)
visc_in = sys.argv[2]
visc = float(visc_in)
schm_in = sys.argv[3]
schm = float(schm_in)
dt_in = sys.argv[4]
dt = float(dt_in)
alpha_in = sys.argv[5]
alpha = float(alpha_in)
Ks_in = sys.argv[6]
Ks = int(Ks_in)

Dc = visc/schm
Diff=(2.0*Dc*dt)**(0.5/alpha)

#Final simulation time and output time
t_end_in = sys.argv[7]
t_end = float(t_end_in)
t_out_freq_in = sys.argv[8]
t_out_freq = float(t_out_freq_in)
chk_freq_in = sys.argv[9]
chk_freq = int(chk_freq_in)

ichk_cnt = int(1.0/(dt*chk_freq))
ichk = ichk_cnt
iout = 0

#Computing the cut-off frequency matrix for dealiasing
cutoff_in = sys.argv[10]
cut_off = float(cutoff_in)#2.0**0.5/3.0
c_off = dealiasing(cut_off,Nnod)

Kf_in = sys.argv[11]
Kf = float(Kf_in)

Npcell_in = sys.argv[12]
Npcell = int(Npcell_in)

#%%############### Computing constant matrices and arrays #####################

meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)
X,Y = np.meshgrid(meshX,meshX)
sz = Nnod**2

#Computing derivatives' matrices
kxx,kyy,kx,ky = derivatives(Nnod)

operator_diff,den,frac_R = get_diffusion_opt(1.0,dt,visc,Nnod,kxx,kyy)                                                         

K_sh,K_sh2,K_sh4 = get_sphere_waven(Nnod)

Ntmax = int(t_end/dt)
out_freq = int(t_out_freq/dt)
iprnt_freq = int(out_freq/5.)

out = np.linspace(0,Ntmax,int(Ntmax/out_freq)+1,dtype=int)

Wmax = Nnod*(2**0.5)/3.0

tmp = np.nonzero(K_sh <= Kf)
ndx_frc = np.array([tmp[0][1::],tmp[1][1::]]).T
sz_frc = ndx_frc.shape[0]

#%%#################### Generating Initial Conditions #########################

icpath = os.path.join(os.getcwd(),'Out_IC')

Uhat = np.genfromtxt(os.path.join(icpath,'Uhat.csv'), 
                     delimiter=',', dtype=complex)
Vhat = np.genfromtxt(os.path.join(icpath,'Vhat.csv'),
                     delimiter=',', dtype=complex)
phihat = np.genfromtxt(os.path.join(icpath,'Phihat_'+str(Ks)+'.csv'), 
                       delimiter=',', dtype=complex)

Uhat *= c_off
Vhat *= c_off

    
pth = 'PTM_'+Npcell_in+'_Ks_'+Ks_in+'_alpha_'+alpha_in+'_Sc_'+schm_in
dirpath = os.path.join(os.getcwd(), pth)
os.mkdir(dirpath)
os.chdir(dirpath)

#Write input variables on a file
f_inp = open('inps.txt', 'w')
print(Nnod_in, visc_in, schm_in, dt_in, alpha_in, Ks_in, t_end_in, 
      t_out_freq_in, chk_freq_in, Kf_in, cutoff_in, Npcell_in, 
      sep=" ", file = f_inp, flush=False)
f_inp.close()


TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,Re,T_L,a_frc = get_stats_eng(
        Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)


U = np.fft.ifftn(Uhat)*sz
V = np.fft.ifftn(Vhat)*sz
phi = np.fft.ifftn(phihat)*sz


dx = meshX[1]-meshX[0]
L = 2.0*np.pi
x2=np.linspace(-dx,L+dx,Nnod+3)

Pt = init_particle(phi.real,Nnod,Npcell,L)

#write output on file
f_time = open('time', 'w')
print(0, sep=" ", end='\n', file = f_time, flush=False)
f_time.close()

MSD=np.zeros(Pt.shape[0])
       
np.savetxt('Pt_'+str(iout)+'.txt', Pt, delimiter=',')  
iout += 1

del phi, phihat

#%%########## Starting the time-stepping w/ Forward-Euler scheme ##############

adv_velxx = U**2
adv_velxy = U*V
adv_velyy = V**2

adv_velxx_hat = np.fft.fftn(adv_velxx)/sz
adv_velxy_hat = np.fft.fftn(adv_velxy)/sz
adv_velyy_hat = np.fft.fftn(adv_velyy)/sz
        
adv_velxx_hatold = adv_velxx_hat
adv_velxy_hatold = adv_velxy_hat
adv_velyy_hatold = adv_velyy_hat

a_frc_old = a_frc

Pt_old = Pt
Pt = Particle_Tracking(Pt,U.real,V.real,x2,Nnod,dx,L,dt,Diff,2.0*alpha,0.0)  
MSD += ((Pt[:,0]-Pt_old[:,0])**2 + (Pt[:,1]-Pt_old[:,1])**2)**0.5

Uhat_tilde = adv_FE(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,dt,
                    kx,ky,operator_diff,den,a_frc[:,0],ndx_frc,sz_frc)
Vhat_tilde = adv_FE(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,dt,
                    kx,ky,operator_diff,den,a_frc[:,1],ndx_frc,sz_frc)

phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde,kx,ky,frac_R)

Uhat,Vhat = corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)

Uhat *= c_off
Vhat *= c_off
    
TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,Re,T_L,a_frc = get_stats_eng(
        Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)

time = dt


U = np.fft.ifftn(Uhat)*sz
V = np.fft.ifftn(Vhat)*sz

adv_velxx = U**2
adv_velxy = U*V
adv_velyy = V**2

icnt = 1
time = 2.0*dt
iprnt = iprnt_freq

#%%############################################################################
############################# Time-Stepping loop ##############################
## 2nd-order Adams-Bashforth scheme for advective & artificial forcing terms ##
################ Crank-Nickelson scheme for diffusion terms ###################
###############################################################################

for nt in range(2,Ntmax+1):

    adv_velxx_hat = np.fft.fftn(adv_velxx)/sz
    adv_velxy_hat = np.fft.fftn(adv_velxy)/sz
    adv_velyy_hat = np.fft.fftn(adv_velyy)/sz

    Pt_old = Pt
    Pt = Particle_Tracking(Pt,U.real,V.real,x2,Nnod,dx,L,dt,Diff,2.0*alpha,0.0)  
    MSD += ((Pt[:,0]-Pt_old[:,0])**2 + (Pt[:,1]-Pt_old[:,1])**2)**0.5

    Uhat_tilde = adv_AB(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,adv_velxx_hatold,
                        adv_velxy_hatold,dt,kx,ky,operator_diff,den,a_frc[:,0],
                        a_frc_old[:,0],ndx_frc,sz_frc)
    Vhat_tilde = adv_AB(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,adv_velxy_hatold,
                        adv_velyy_hatold,dt,kx,ky,operator_diff,den,a_frc[:,1],
                        a_frc_old[:,1],ndx_frc,sz_frc)

    phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde,kx,ky,frac_R)

    Uhat,Vhat = corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)
    
    Uhat *= c_off
    Vhat *= c_off

    a_frc_old = a_frc
    
    TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,Re,T_L,a_frc = get_stats_eng(
            Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)
       
    
    if nt == iprnt:

        f1 = open('FlowFeatures.txt', 'a')
        print(format(time, '.4f'), 
              format(TKE, '.6f'),
              format(Diss, '.6f'),
              format(Wmax/K_eta, '.2f'),
              format(np.mean(MSD), '.8f'), 
              sep=" ", end='\n', file = f1, flush=False)
        f1.close()

        
        iprnt += iprnt_freq
    
    U = np.fft.ifftn(Uhat)*sz
    V = np.fft.ifftn(Vhat)*sz

    #write output on file
    if nt == ichk:        
        
        f_time = open('time', 'a')
        print(time, '.3f', sep=" ", end='\n', file = f_time, flush=False)
        f_time.close()
               
        np.savetxt('Pt_'+str(iout)+'.txt', Pt, delimiter=',')

        iout += 1
        ichk += ichk_cnt        
        
                       
    adv_velxx = U**2
    adv_velxy = U*V
    adv_velyy = V**2
    
    adv_velxx_hatold = adv_velxx_hat
    adv_velxy_hatold = adv_velxy_hat
    adv_velyy_hatold = adv_velyy_hat

    time += dt