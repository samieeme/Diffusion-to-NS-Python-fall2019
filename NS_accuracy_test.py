"""
Created on Fri Oct 11 14:02:26 2019
@author: samieeme
"""

import numpy as np
from numpy import linalg as LA
#from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, optimum_prnt
#from Outputs import Output_Corr

from functions_NS_2D import adv_FE, adv_AB, diff_cont, corrector, gen_IC_vel, plot_Vel, dealiasing

Nnod = 128
meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)
X,Y = np.meshgrid(meshX,meshX)

#Computing the cut-off frequency matrix for dealiasing
cut_off=2./3.
c_off=dealiasing(cut_off,Nnod)

visc = 1.0

dt=0.0001
jmax=10000
out=100

#Uhat,Vhat=gen_IC_vel(Nnod)
#
#U = np.real(np.fft.ifft2(Uhat))
#V = np.real(np.fft.ifft2(Vhat))

U = -np.cos(X)*np.sin(Y)
V = np.cos(Y)*np.sin(X)
#plot_Vel(X,Y,U,V,0,'seismic')

adv_velxx = U[:] * U[:]
adv_velxy = U[:] * V[:]
adv_velyy = V[:] * V[:]

Vhat = np.fft.fft2(V)
Uhat = np.fft.fft2(U)

adv_velxx_hat = (np.fft.fft2(adv_velxx))*c_off
adv_velxy_hat = (np.fft.fft2(adv_velxy))*c_off
adv_velyy_hat = (np.fft.fft2(adv_velyy))*c_off

adv_velxx_hatold = adv_velxx_hat[:]
adv_velxy_hatold = adv_velxy_hat[:]
adv_velyy_hatold = adv_velyy_hat[:]


Uhat_tilde = adv_FE(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,visc,dt)
Vhat_tilde = adv_FE(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,visc,dt)

phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde)

Uhat_new,Vhat_new=corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt)


Uhat=Uhat_new[:]
Vhat=Vhat_new[:]


U =  np.real(np.fft.ifft2(Uhat))
V =  np.real(np.fft.ifft2(Vhat))


U_ex = -np.cos(X)*np.sin(Y)*np.exp(-2.0*visc*dt)
V_ex = np.cos(Y)*np.sin(X)*np.exp(-2.0*visc*dt)

plot_Vel(X,Y,U,U_ex,1,'seismic')

        
e1=(LA.norm(U_ex-U, ord=2))/(LA.norm(U_ex, ord=2))
e2=(LA.norm(V_ex-V, ord=2))/(LA.norm(V_ex, ord=2))
print(1,e1,e2, sep=' ', end='\n')

adv_velxx = U[:] * U[:]
adv_velxy = U[:] * V[:]
adv_velyy = V[:] * V[:]


for j in range(2,jmax+1):

    adv_velxx_hat = (np.fft.fft2(adv_velxx))*c_off
    adv_velxy_hat = (np.fft.fft2(adv_velxy))*c_off
    adv_velyy_hat = (np.fft.fft2(adv_velyy))*c_off


    Uhat_tilde=adv_AB(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,adv_velxx_hatold,adv_velxy_hatold,visc,dt)
    Vhat_tilde=adv_AB(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,adv_velxy_hatold,adv_velyy_hatold,visc,dt)

    phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde)

    Uhat,Vhat=corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt)


    U = np.real(np.fft.ifft2(Uhat))
    V = np.real(np.fft.ifft2(Vhat))

    if j == out:
        
        U_ex = -np.cos(X)*np.sin(Y)*np.exp(-2.0*visc*j*dt)
        V_ex = np.cos(Y)*np.sin(X)*np.exp(-2.0*visc*j*dt)
        
        plot_Vel(X,Y,U,U_ex,j,'seismic')
        out += 100
                
        e1=(LA.norm(U_ex-U, ord=2))/(LA.norm(U_ex, ord=2))
        e2=(LA.norm(V_ex-V, ord=2))/(LA.norm(V_ex, ord=2))
        print(j,e1,e2, sep=' ', end='\n')
        
    adv_velxx = U[:] * U[:]
    adv_velxy = U[:] * V[:]
    adv_velyy = V[:] * V[:]
    adv_velxx_hatold = adv_velxx_hat[:]
    adv_velxy_hatold = adv_velxy_hat[:]
    adv_velyy_hatold = adv_velyy_hat[:]
