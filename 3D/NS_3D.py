#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 18:23:26 2019

@author: aliakhavan
"""

import numpy as np
from numpy import linalg as LA


from functions_NS_2D import derivatives, get_diffusion_opt, adv_FE, adv_AB
from functions_NS_2D import diff_cont, corrector, gen_IC_vel, gen_IC_vel1
from functions_NS_2D import get_vorticity, plot_Vel, plot_Vor, dealiasing
from functions_stats import get_sphere_waven, get_stats_eng, Moments_dVdX

Nnod = 256
visc = 0.0001
dt = 0.0002
alpha = 1.0
Kf = 2.0*2.0**0.5

meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)
X,Y = np.meshgrid(meshX,meshX)

#Computing derivatives' matrices
kxx,kyy,kzz,kx,ky,kz = derivatives(Nnod)

operator_diff,den,frac_R = get_diffusion_opt(alpha,dt,visc,Nnod,kxx,kyy)

K_sh,K_sh2,K_sh4 = get_sphere_waven(Nnod)

#Computing the cut-off frequency matrix for dealiasing
cut_off = 2.0/3.0
c_off = dealiasing(cut_off,Nnod)

#Final simulation time and output time
t_end = 30.0
t_out_freq = 1.0


Ntmax = int(t_end/dt)
out_freq = int(t_out_freq/dt)
iprnt_freq = int(out_freq/2)

out = np.linspace(0,Ntmax,int(Ntmax/out_freq)+1,dtype=int)


Wmax = (Nnod*2**0.5)/3.0
Uhat,Vhat = gen_IC_vel1(Nnod,Kf)
#Uhat,Vhat=gen_IC_vel(Nnod)

M1,M2 = Moments_dVdX(Nnod**2,Uhat,Vhat,kx,ky)
print(M1, sep='   ',
      end='\n****************************************************\n')
print(M2, sep='   ',
      end='\n****************************************************\n')


tmp = np.nonzero(K_sh <= Kf)
ndx_frc = np.array([tmp[0][1::],tmp[1][1::]]).T
sz_frc = ndx_frc.shape[0]

TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,Re,T_L,a_frc = get_stats_eng(
        Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)

Vor = get_vorticity(Nnod,Uhat,Vhat,kx,ky)

U = np.real(np.fft.ifft2(Uhat))
V = np.real(np.fft.ifft2(Vhat))
plot_Vel(X,Y,U,V,0,'seismic')

plot_Vor(X,Y,Vor,0.0,1,'seismic')

print(TKE,Diss,Wmax/K_eta,Re_l,Re,T_L, sep=' ', end='\n')

adv_velxx = U**2
adv_velxy = U*V
adv_velyy = V**2


adv_velxx_hat = (np.fft.fft2(adv_velxx))*c_off
adv_velxy_hat = (np.fft.fft2(adv_velxy))*c_off
adv_velyy_hat = (np.fft.fft2(adv_velyy))*c_off

adv_velxx_hatold = adv_velxx_hat
adv_velxy_hatold = adv_velxy_hat
adv_velyy_hatold = adv_velyy_hat

a_frc_old = a_frc

Uhat_tilde = adv_FE(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,dt,
                    kx,ky,operator_diff,den,a_frc[:,0],ndx_frc,sz_frc)
Vhat_tilde = adv_FE(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,dt,
                    kx,ky,operator_diff,den,a_frc[:,1],ndx_frc,sz_frc)

phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde,kx,ky,frac_R)

Uhat,Vhat = corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)


TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,Re,T_L,a_frc = get_stats_eng(
        Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)


print(TKE,Diss,Wmax/K_eta,Re_l,Re,T_L, sep=' ', end='\n')


U = np.real(np.fft.ifft2(Uhat))
V = np.real(np.fft.ifft2(Vhat))


icnt = 0
icnt += 1

adv_velxx = U**2
adv_velxy = U*V
adv_velyy = V**2

time = 2.0*dt
iprnt = iprnt_freq

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#Time-Stepping loop
###############################################################################

for nt in range(2,Ntmax+1):

    adv_velxx_hat = (np.fft.fft2(adv_velxx))*c_off
    adv_velxy_hat = (np.fft.fft2(adv_velxy))*c_off
    adv_velyy_hat = (np.fft.fft2(adv_velyy))*c_off


    Uhat_tilde = adv_AB(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,adv_velxx_hatold,
                        adv_velxy_hatold,dt,kx,ky,operator_diff,den,a_frc[:,0],
                        a_frc_old[:,0],ndx_frc,sz_frc)
    Vhat_tilde = adv_AB(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,adv_velxy_hatold,
                        adv_velyy_hatold,dt,kx,ky,operator_diff,den,a_frc[:,1],
                        a_frc_old[:,1],ndx_frc,sz_frc)

    phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde,kx,ky,frac_R)

    Uhat,Vhat = corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)

    a_frc_old = a_frc
    
    TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,Re,T_L,a_frc = get_stats_eng(
            Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)
    if nt == iprnt:
        print(TKE,Diss,Wmax/K_eta,Re_l,Re,T_L, sep=' ', end='\n')
        iprnt += iprnt_freq
    
    U = np.real(np.fft.ifft2(Uhat))
    V = np.real(np.fft.ifft2(Vhat))

    if nt == out[icnt]:
               
#        plot_Vel(X,Y,U,V,out_t[icnt],'seismic')        
        Vor = get_vorticity(Nnod,Uhat,Vhat,kx,ky)
        plot_Vor(X,Y,Vor,np.round(time),icnt+1,'seismic')
        
        M1,M2 = Moments_dVdX(Nnod**2,Uhat,Vhat,kx,ky)
        print(M1, sep='   ',
              end='\n****************************************************\n')
        print(M2, sep='   ',
              end='\n****************************************************\n')
       
        icnt += 1
                       
    adv_velxx = U**2
    adv_velxy = U*V
    adv_velyy = V**2
    adv_velxx_hatold = adv_velxx_hat
    adv_velxy_hatold = adv_velxy_hat
    adv_velyy_hatold = adv_velyy_hat
    
    time += dt

plot_Vel(X,Y,U,V,np.round(time),'seismic')