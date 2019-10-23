"""
Created on Fri Oct 11 14:02:26 2019
@author: samieeme
"""

import numpy as np
from numpy import linalg as LA


from functions_NS_2D import derivatives, get_diffusion_opt, adv_FE, adv_AB
from functions_NS_2D import diff_cont, corrector, gen_IC_vel, gen_IC_vel1
from functions_NS_2D import get_vorticity, plot_Vel, plot_Vor, dealiasing
from functions_stats import get_sphere_waven, get_stats_eng

Nnod = 512
visc = 0.01
dt = 0.0001
alpha = 1.0

meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)
X,Y = np.meshgrid(meshX,meshX)

#Computing derivatives' matrices
kxx,kyy,kx,ky = derivatives(Nnod)

operator_diff,den,frac_R = get_diffusion_opt(alpha,dt,visc,Nnod,kxx,kyy)

K_sh,K_sh2,K_sh4 = get_sphere_waven(Nnod)

#Computing the cut-off frequency matrix for dealiasing
cut_off=2.0/3.0
c_off=dealiasing(cut_off,Nnod)



out=np.array([0,5,10,40,100,200,300,500,700,1000,
              2000,3000,4000,5000,6000,7000,8000,9000,10000])
out_t=np.array([0.0,0.001,0.004,0.01,0.02,0.03,0.05,0.07,0.1,
                0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

#out=np.array([0,5000,15000,30000,70000,100000])
#out_t=np.array([0.0,0.5,1.5,3.0,7.0,10.0])

Ntmax=out[-1]
#out=20
#freq_out=out


#Uhat,Vhat=gen_IC_vel(Nnod)
Wmax=(Nnod*2**0.5)/3.0
Kf=2.0*2.0**0.5
Uhat,Vhat=gen_IC_vel1(Nnod,Kf)

tmp = np.nonzero(K_sh <= Kf)
ndx_frc = np.array([tmp[0][1::],tmp[1][1::]]).T
sz_frc = ndx_frc.shape[0]

TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,a_frc=get_stats_eng(
        Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)

Vor=get_vorticity(Nnod,Uhat,Vhat,kx,ky)
plot_Vor(X,Y,Vor,out_t[0],1,'seismic')

U = np.real(np.fft.ifft2(Uhat))
V = np.real(np.fft.ifft2(Vhat))

#plot_Vel(X,Y,U,V,0,'seismic')

adv_velxx = U**2
adv_velxy = U*V
adv_velyy = V**2

#Vhat = np.fft.fft2(V)
#Uhat = np.fft.fft2(U)

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

Uhat,Vhat=corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)

a_frc_old = a_frc

TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,a_frc=get_stats_eng(
        Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)

print(Wmax/K_eta,Re_l,int_l,mic_l,TKE,Enst,eta,Diss, sep=' ', end='\n')


U = np.real(np.fft.ifft2(Uhat))
V = np.real(np.fft.ifft2(Vhat))

#plot_Vel(X,Y,U,V,1,'seismic')
icnt=0
#Vor=get_vorticity(Nnod,Uhat,Vhat)
#plot_Vor(X,Y,Vor,out_t[icnt],icnt+1,'seismic')
icnt +=1

adv_velxx = U**2
adv_velxy = U*V
adv_velyy = V**2

a_frc_old=np.zeros((sz_frc,2))

for nt in range(2,Ntmax+1):

    adv_velxx_hat = (np.fft.fft2(adv_velxx))*c_off
    adv_velxy_hat = (np.fft.fft2(adv_velxy))*c_off
    adv_velyy_hat = (np.fft.fft2(adv_velyy))*c_off


    Uhat_tilde=adv_AB(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,adv_velxx_hatold,
                      adv_velxy_hatold,dt,kx,ky,operator_diff,den,a_frc[:,0],a_frc_old[:,0],ndx_frc,sz_frc)
    Vhat_tilde=adv_AB(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,adv_velxy_hatold,
                      adv_velyy_hatold,dt,kx,ky,operator_diff,den,a_frc[:,1],a_frc_old[:,1],ndx_frc,sz_frc)

    phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde,kx,ky,frac_R)

    Uhat,Vhat=corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)

    a_frc_old = a_frc
    
    TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,a_frc=get_stats_eng(
            Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)

    U = np.real(np.fft.ifft2(Uhat))
    V = np.real(np.fft.ifft2(Vhat))

    if nt == out[icnt]:
               
#        plot_Vel(X,Y,U,V,nt,'seismic')
        
        Vor=get_vorticity(Nnod,Uhat,Vhat,kx,ky)
        plot_Vor(X,Y,Vor,out_t[icnt],icnt+1,'seismic')
        
#        TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l=get_stats_eng(Uhat,Vhat,visc,K_sh,K_sh2,K_sh4)
        print(Wmax/K_eta,Re_l,int_l,mic_l,TKE,Enst,eta,Diss, sep=' ', end='\n')
        
        icnt +=1
#        out += freq_out
                
        
    adv_velxx = U**2
    adv_velxy = U*V
    adv_velyy = V**2
    adv_velxx_hatold = adv_velxx_hat
    adv_velxy_hatold = adv_velxy_hat
    adv_velyy_hatold = adv_velyy_hat
