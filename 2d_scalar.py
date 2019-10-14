# In the name of Gd
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:56:34 2019

@author: samieeme
"""

import numpy as np
#from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, optimum_prnt
#from Outputs import Output_Corr

from functions_diff_2D import solV, forcef,  diff_eq, diff_x, adv_FE, adv_AB, solV_t, forcef_t, diff_FE, diff_AB,Ux_t,Uy_t

Nnod = 10
meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)

V = np.zeros([Nnod,Nnod])
Ux = np.zeros([Nnod,Nnod])
Uy = np.zeros([Nnod,Nnod])
f = np.zeros([Nnod,Nnod])


alpha_t = 0.01
err = np.zeros([5])

for kt in range(5):

    dt = 0.01/2.0**(kt)
    
    for i in range(Nnod):
        for j in range(Nnod):
            V[i,j] = solV_t(meshX[i],meshX[j],(1-1)*dt)
            Ux[i,j] = Ux_t(meshX[i],meshX[j],(1-1)*dt)
            Uy[i,j] = Uy_t(meshX[i],meshX[j],(1-1)*dt)
    #        f[i,j] = forcef(meshX[i],meshX[j])
    
    adv_velx = Ux[:] * V[:]
    adv_vely = Uy[:] * V[:]
    
    
    Vhat = np.fft.fft2(V)
    Vhat_old = Vhat[:]
    jmax = 300*2**kt
    
    for k1 in range(Nnod):
        for k2 in range(Nnod):
            f[k1,k2] = forcef_t(meshX[k1],meshX[k2],(1-1)*dt,alpha_t)    
    fhat = np.fft.fft2(f)
    adv_velx_hat = np.fft.fft2(adv_velx)
    adv_vely_hat = np.fft.fft2(adv_vely)
    adv_velx_hatold = adv_velx_hat[:]
    adv_vely_hatold = adv_vely_hat[:]
    #Vhat_new=diff_FE(Nnod,fhat,Vhat,alpha_t,dt)
    Vhat_new = adv_FE(Nnod,fhat,Vhat,adv_velx_hat,adv_vely_hat,alpha_t,dt)
    Vhat=Vhat_new[:]
    V = np.real(np.fft.ifft2(Vhat))
    adv_velx = Ux[:] * V[:]
    adv_vely = Uy[:] * V[:]
    
    for j in range(2,jmax):     
        fhat_old = fhat[:]
        for k1 in range(Nnod):
            for k2 in range(Nnod):
                f[k1,k2] = forcef_t(meshX[k1],meshX[k2],(j-1)*dt,alpha_t)    
        fhat = np.fft.fft2(f)
        adv_velx_hat = np.fft.fft2(adv_velx)
        adv_vely_hat = np.fft.fft2(adv_vely)
#        adv_velx_hatold = adv_velx_hat[:]
#        adv_vely_hatold = adv_vely_hat[:]
       # Vhat_new=diff_AB(Nnod,fhat,fhat_old,Vhat,Vhat_old,alpha_t,dt)
        #Vhat_new=adv_FE(Nnod,fhat,Vhat,adv_velx_hat,adv_vely_hat,alpha_t,dt)
        Vhat_new=adv_AB(Nnod,fhat,fhat_old,Vhat,adv_velx_hat,adv_vely_hat,adv_velx_hatold,adv_vely_hatold,alpha_t,dt)
        Vhat_old = Vhat[:]
        Vhat=Vhat_new[:]
        
        V = np.real(np.fft.ifft2(Vhat))
        adv_velx = Ux[:] * V[:]
        adv_vely = Uy[:] * V[:]
        adv_velx_hatold = adv_velx_hat[:]
        adv_vely_hatold = adv_vely_hat[:]

        
    V_new = np.real(np.fft.ifft2(Vhat))
    for i in range(Nnod):
        for j in range(Nnod):
            V[i,j] = solV_t(meshX[i],meshX[j],(jmax-1)*dt)
    
    
    #vs = diff_eq(Nnod,fhat)
    #dxV = diff_x(Nnod,Vhat)
    
   # err[kt] = abs(np.max(V_new[:] - V[:])) linalg
    err[kt] = np.linalg.norm(V_new[:] - V[:],2)

#%%
dt1 = 0.001
dt2 = 0.001/2.0**3.
rate_conver = (np.log(err[3])-np.log(err[0]))/(np.log(dt2)-np.log(dt1))

