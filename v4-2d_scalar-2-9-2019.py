# In the name of Gd
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 10:56:34 2019

@author: samieeme
"""

import numpy as np
#from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, optimum_prnt
#from Outputs import Output_Corr

from functions_diff_2D import solV, forcef,  diff_eq, diff_x, adv_FE, adv_AB, solV_t, forcef_t

Nnod = 10
meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)

V = np.zeros([Nnod,Nnod])
f = np.zeros([Nnod,Nnod])


alpha_t = 1
err = np.zeros([5])

for kt in range(5):

    dt = 0.00001/5.0**(kt)
    
    for i in range(Nnod):
        for j in range(Nnod):
            V[i,j] = solV_t(meshX[i],meshX[j],(1-1)*dt)
    #        f[i,j] = forcef(meshX[i],meshX[j])
    
    
    Vhat = np.fft.fft2(V)
    Vhat_old = Vhat[:]
    jmax =10000
    
    for k1 in range(Nnod):
        for k2 in range(Nnod):
            f[k1,k2] = forcef_t(meshX[k1],meshX[k2],(1-1)*dt,alpha_t)    
    fhat = np.fft.fft2(f)
    Vhat_new=adv_FE(Nnod,fhat,Vhat,alpha_t,dt)
    Vhat=Vhat_new
    
    for j in range(2,jmax):
        fhat_old = fhat[:]
        for k1 in range(Nnod):
            for k2 in range(Nnod):
                f[k1,k2] = forcef_t(meshX[k1],meshX[k2],(j-1)*dt,alpha_t)    
        fhat = np.fft.fft2(f)
        #Vhat_new=adv_AB(Nnod,fhat,Vhat,Vhat_old,alpha_t,dt)
        Vhat_new=adv_AB(Nnod,fhat,fhat_old,Vhat,Vhat_old,alpha_t,dt)
        Vhat_old = Vhat
        Vhat=Vhat_new[:]
        #fhat_old = fhat[:]
        
    V_new = np.real(np.fft.ifft2(Vhat_new))
    for i in range(Nnod):
        for j in range(Nnod):
            V[i,j] = solV_t(meshX[i],meshX[j],(jmax-1)*dt)
    
    
    #vs = diff_eq(Nnod,fhat)
    #dxV = diff_x(Nnod,Vhat)
    
    err[kt] = abs(np.max(V_new[:] - V[:]))



dt1 = 0.01
dt2 = 0.01/5.0**5.
rate_conver = (np.log(err[4])-np.log(err[0]))/(np.log(dt2)-np.log(dt1))


#%%


