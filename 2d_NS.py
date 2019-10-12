# In the name of God

"""
Created on Fri Oct 11 14:02:26 2019

@author: samieeme
"""

import numpy as np
#from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, optimum_prnt
#from Outputs import Output_Corr

from functions_NS_2D import solV, forcef,  diff_eq, diff_x, adv_FE, adv_AB, phi_t, forcef_t, diff_FE, diff_AB,Ux_t,Uy_t,  diff_cont, corrector

Nnod = 10
meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)

V = np.zeros([Nnod,Nnod]) 
U = np.zeros([Nnod,Nnod])

phi = np.zeros([Nnod,Nnod]) 
f = np.zeros([Nnod,Nnod]) 

visc = 1 
schm = 0.7 

err = np.zeros([2]) 

for kt in [1]: 
    
    dt = 0.001/5.0**(kt)     
    for i in range(Nnod): 
        for j in range(Nnod): 
            phi[i,j] = phi_t(meshX[i],meshX[j],(1-1)*dt) 
            U[i,j] = Ux_t(meshX[i],meshX[j],(1-1)*dt) 
            V[i,j] = Uy_t(meshX[i],meshX[j],(1-1)*dt) 
    #        f[i,j] = forcef(meshX[i],meshX[j])
    
    adv_velxx = U[:] * U[:] 
    adv_velxy = U[:] * V[:] 
    adv_velyy = V[:] * V[:] 
    
    
    
    Vhat = np.fft.fft2(V) 
    Vhat_old = Vhat[:]    
    Uhat = np.fft.fft2(U) 
    Uhat_old = Uhat[:]    
    phihat = np.fft.fft2(phi) 
    phihat_old = phihat[:]    
    jmax = 5000           
    
#    for k1 in range(Nnod):
#        for k2 in range(Nnod):
#            f[k1,k2] = forcef_t(meshX[k1],meshX[k2],(1-1)*dt,visc)    
    fhat = np.fft.fft2(f)
    adv_velxx_hat = np.fft.fft2(adv_velxx) 
    adv_velxy_hat = np.fft.fft2(adv_velxy) 
    adv_velyy_hat = np.fft.fft2(adv_velyy)  
    adv_velxx_hatold = adv_velxx_hat[:] 
    adv_velxy_hatold = adv_velxy_hat[:] 
    adv_velyy_hatold = adv_velyy_hat[:] 
    #Vhat_new=diff_FE(Nnod,fhat,Vhat,alpha_t,dt)
    Uhat_tilde = adv_FE(Nnod,Uhat,adv_velxy_hat,adv_velyy_hat,visc,dt) 
    Vhat_tilde = adv_FE(Nnod,Vhat,adv_velxx_hat,adv_velxy_hat,visc,dt) 
    
    phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde) 
    
    Uhat_new,Vhat_new=corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt)  
    
    Uhat=Uhat_new[:]
    Vhat=Vhat_new[:]
    
    U =  np.real(np.fft.ifft2(Uhat))
    V =  np.real(np.fft.ifft2(Vhat))

    adv_velxx = U[:] * U[:] 
    adv_velxy = U[:] * V[:] 
    adv_velyy = V[:] * V[:] 
    
    for j in range(2,jmax):
        fhat_old = fhat[:]
#        for k1 in range(Nnod):
#            for k2 in range(Nnod):
#                f[k1,k2] = forcef_t(meshX[k1],meshX[k2],(j-1)*dt,alpha_t)    
#        fhat = np.fft.fft2(f)
        adv_velxx_hat = np.fft.fft2(adv_velxx)
        adv_velxy_hat = np.fft.fft2(adv_velxy)
        adv_velyy_hat = np.fft.fft2(adv_velyy)
       # Vhat_new=diff_AB(Nnod,fhat,fhat_old,Vhat,Vhat_old,alpha_t,dt)
        #Vhat_new=adv_FE(Nnod,fhat,Vhat,adv_velx_hat,adv_vely_hat,alpha_t,dt)
        Uhat_tilde=adv_AB(Nnod,fhat,fhat_old,Uhat,adv_velxx_hat,adv_velxy_hat,adv_velxx_hatold,adv_velxy_hatold,visc,dt)
        Vhat_tilde=adv_AB(Nnod,fhat,fhat_old,Vhat,adv_velxy_hat,adv_velyy_hat,adv_velxy_hatold,adv_velyy_hatold,visc,dt)
        phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde) 
        
        Uhat_new,Vhat_new=corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt)  
        Uhat_old = Uhat[:]
        Vhat_old = Vhat[:]
        Uhat=Uhat_new[:]
        Vhat=Vhat_new[:]
        
        U =  np.real(np.fft.ifft2(Uhat))
        V =  np.real(np.fft.ifft2(Vhat))
    
        adv_velxx = U[:] * U[:] 
        adv_velxy = U[:] * V[:] 
        adv_velyy = V[:] * V[:]     
        adv_velxx_hatold = adv_velxx_hat[:]
        adv_velxy_hatold = adv_velxy_hat[:]
        adv_velyy_hatold = adv_velyy_hat[:]
    
    
    
    
    
    
    