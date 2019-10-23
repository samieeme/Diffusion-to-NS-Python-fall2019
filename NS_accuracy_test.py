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

Nnod = 256
visc = 0.1
dt = 0.001
alpha = 1.0
Kf = 2.0*2.0**0.5

meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)
X,Y = np.meshgrid(meshX,meshX)

#Computing derivatives' matrices
kxx,kyy,kx,ky = derivatives(Nnod)

operator_diff,den,frac_R = get_diffusion_opt(alpha,dt,visc,Nnod,kxx,kyy)

K_sh,K_sh2,K_sh4 = get_sphere_waven(Nnod)

#Computing the cut-off frequency matrix for dealiasing
cut_off = 2.0/3.0
c_off = dealiasing(cut_off,Nnod)


out_t = np.linspace(0,2,401)
out = np.linspace(0,2000,401,dtype=int)

Ntmax = out[-1]

#Uhat,Vhat=gen_IC_vel(Nnod)
Wmax = (Nnod*2**0.5)/3.0
Uhat,Vhat = gen_IC_vel1(Nnod,Kf)

tmp = np.nonzero(K_sh <= Kf)
ndx_frc = np.array([tmp[0][1::],tmp[1][1::]]).T
sz_frc = ndx_frc.shape[0]

TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,a_frc = get_stats_eng(
        Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)

Vor = get_vorticity(Nnod,Uhat,Vhat,kx,ky)

U = np.real(np.fft.ifft2(Uhat))
V = np.real(np.fft.ifft2(Vhat))
plot_Vel(X,Y,U,V,0,'seismic')

plot_Vor(X,Y,Vor,out_t[0],1,'seismic')

print(TKE,Diss, sep=' ', end='\n')

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

a_frc_old = a_frc

TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,a_frc = get_stats_eng(
        Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)

#print(Wmax/K_eta,TKE,Enst,eta,Diss, sep=' ', end='\n')
print(TKE,Diss, sep=' ', end='\n')


U = np.real(np.fft.ifft2(Uhat))
V = np.real(np.fft.ifft2(Vhat))

#plot_Vel(X,Y,U,V,1,'seismic')
icnt = 0
#Vor=get_vorticity(Nnod,Uhat,Vhat)
#plot_Vor(X,Y,Vor,out_t[icnt],icnt+1,'seismic')
icnt += 1

adv_velxx = U**2
adv_velxy = U*V
adv_velyy = V**2

a_frc_old = np.zeros((sz_frc,2))

for nt in range(2,Ntmax+1):

    adv_velxx_hat = (np.fft.fft2(adv_velxx))*c_off
    adv_velxy_hat = (np.fft.fft2(adv_velxy))*c_off
    adv_velyy_hat = (np.fft.fft2(adv_velyy))*c_off


    Uhat_tilde = adv_AB(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,adv_velxx_hatold,
                        adv_velxy_hatold,dt,kx,ky,operator_diff,den,a_frc[:,0],a_frc_old[:,0],ndx_frc,sz_frc)
    Vhat_tilde = adv_AB(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,adv_velxy_hatold,
                        adv_velyy_hatold,dt,kx,ky,operator_diff,den,a_frc[:,1],a_frc_old[:,1],ndx_frc,sz_frc)

    phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde,kx,ky,frac_R)

    Uhat,Vhat = corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)

    a_frc_old = a_frc
    
    TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,a_frc = get_stats_eng(
            Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)

    print(TKE,Diss, sep=' ', end='\n')
    U = np.real(np.fft.ifft2(Uhat))
    V = np.real(np.fft.ifft2(Vhat))

    if nt == out[icnt]:
               
#        plot_Vel(X,Y,U,V,out_t[icnt],'seismic')        
        Vor = get_vorticity(Nnod,Uhat,Vhat,kx,ky)
        plot_Vor(X,Y,Vor,out_t[icnt],icnt+1,'seismic')
        
#        TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,a_frc = get_stats_eng(
#                Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)
#        print(TKE,Diss, sep=' ', end='\n')        
        icnt += 1
                       
    adv_velxx = U**2
    adv_velxy = U*V
    adv_velyy = V**2
    adv_velxx_hatold = adv_velxx_hat
    adv_velxy_hatold = adv_velxy_hat
    adv_velyy_hatold = adv_velyy_hat