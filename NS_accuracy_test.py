"""
Created on Fri Oct 11 14:02:26 2019
@author: samieeme
"""

import numpy as np
from numpy import linalg as LA
#from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, optimum_prnt
#from Outputs import Output_Corr

from functions_NS_2D import derivatives, adv_FE, adv_AB, diff_cont, corrector, gen_IC_vel, gen_IC_vel1, get_vorticity, plot_Vel, plot_Vor, dealiasing
from functions_stats import get_sphere_waven, get_stats_eng

Nnod = 256
meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)
X,Y = np.meshgrid(meshX,meshX)

#Computing derivatives' matrices

kxx, kyy, kx, ky = derivatives(Nnod)

#Computing the cut-off frequency matrix for dealiasing
cut_off=0.67
c_off=dealiasing(cut_off,Nnod)

visc = 1.0

out=np.array([0,5,10,40,100,200,300,500,700,1000,2000,3000,4000,5000,6000,7000,8000,
              9000,10000])
out_t=np.array([0.0,0.001,0.004,0.01,0.02,0.03,0.05,0.07,0.1,0.2,0.3,0.4,0.5,0.6,0.7,
                0.8,0.9,1.0])

#out=np.array([0,5000,15000,30000,70000,100000])
#out_t=np.array([0.0,0.5,1.5,3.0,7.0,10.0])

dt=0.0001
jmax=out[-1]
#out=20
#freq_out=out


#Uhat,Vhat=gen_IC_vel(Nnod)
Kf=3
Uhat,Vhat=gen_IC_vel1(Nnod,Kf)


Vor=get_vorticity(Nnod,Uhat,Vhat,kx,ky)
plot_Vor(X,Y,Vor,out_t[0],1,'seismic')

U = np.real(np.fft.ifft2(Uhat))
V = np.real(np.fft.ifft2(Vhat))

#plot_Vel(X,Y,U,V,0,'seismic')

adv_velxx = U[:] * U[:]
adv_velxy = U[:] * V[:]
adv_velyy = V[:] * V[:]

#Vhat = np.fft.fft2(V)
#Uhat = np.fft.fft2(U)

adv_velxx_hat = (np.fft.fft2(adv_velxx))*c_off
adv_velxy_hat = (np.fft.fft2(adv_velxy))*c_off
adv_velyy_hat = (np.fft.fft2(adv_velyy))*c_off

adv_velxx_hatold = adv_velxx_hat[:]
adv_velxy_hatold = adv_velxy_hat[:]
adv_velyy_hatold = adv_velyy_hat[:]


Uhat_tilde = adv_FE(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,visc,dt,kxx,kyy,kx,ky)
Vhat_tilde = adv_FE(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,visc,dt,kxx,kyy,kx,ky)

phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde,kxx,kyy,kx,ky)

Uhat,Vhat=corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)

U = np.real(np.fft.ifft2(Uhat))
V = np.real(np.fft.ifft2(Vhat))

#plot_Vel(X,Y,U,V,1,'seismic')
icnt=0
#Vor=get_vorticity(Nnod,Uhat,Vhat)
#plot_Vor(X,Y,Vor,out_t[icnt],icnt+1,'seismic')
icnt +=1


adv_velxx = U[:] * U[:]
adv_velxy = U[:] * V[:]
adv_velyy = V[:] * V[:]

for j in range(2,jmax+1):

    adv_velxx_hat = (np.fft.fft2(adv_velxx))*c_off
    adv_velxy_hat = (np.fft.fft2(adv_velxy))*c_off
    adv_velyy_hat = (np.fft.fft2(adv_velyy))*c_off


    Uhat_tilde=adv_AB(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,adv_velxx_hatold,adv_velxy_hatold,visc,dt,kxx,kyy,kx,ky)
    Vhat_tilde=adv_AB(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,adv_velxy_hatold,adv_velyy_hatold,visc,dt,kxx,kyy,kx,ky)

    phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde,kxx,kyy,kx,ky)

    Uhat,Vhat=corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)


    U = np.real(np.fft.ifft2(Uhat))
    V = np.real(np.fft.ifft2(Vhat))

    if j == out[icnt]:
               
#        plot_Vel(X,Y,U,V,j,'seismic')
        
        Vor=get_vorticity(Nnod,Uhat,Vhat,kx,ky)
        plot_Vor(X,Y,Vor,out_t[icnt],icnt+1,'seismic')
        icnt +=1
#        out += freq_out
                
        
    adv_velxx = U[:] * U[:]
    adv_velxy = U[:] * V[:]
    adv_velyy = V[:] * V[:]
    adv_velxx_hatold = adv_velxx_hat[:]
    adv_velxy_hatold = adv_velxy_hat[:]
    adv_velyy_hatold = adv_velyy_hat[:]
