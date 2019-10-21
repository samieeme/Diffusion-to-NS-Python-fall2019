# In the name of God

"""
Created on Fri Oct 11 14:02:26 2019
@author: samieeme
"""

import numpy as np
#from functions import remain, deriv_x, deriv_y, deriv_z, div, Reduce_period, optimum_prnt
#from Outputs import Output_Corr

from functions_NS_2D import solV, forcef,  diff_eq, diff_x, adv_FE, adv_AB, phi_t, forcef_t, diff_FE, diff_AB,Ux_t,Uy_t,  diff_cont, corrector, dealiasing

Nnod = 15
meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)

V = np.zeros([Nnod,Nnod]) 
U = np.zeros([Nnod,Nnod])
U_ex = np.zeros([Nnod,Nnod]) 
V_ex = np.zeros([Nnod,Nnod])
phi_ex = np.zeros([Nnod,Nnod])

phi = np.zeros([Nnod,Nnod]) 
f = np.zeros([Nnod,Nnod]) 
fs = np.zeros([Nnod,Nnod]) 

visc = 1.0 
schm = 1.0
err = np.zeros([5]) 

<<<<<<< HEAD
#for convergence test
#for kt in range(5): 


# InitC
kt=1    
dt = 0.01/3.0**(kt)     
for i in range(Nnod): 
    for j in range(Nnod): 
        phi[i,j] = phi_t(meshX[i],meshX[j],(1-1)*dt) 
        U[i,j] = Ux_t(meshX[i],meshX[j],(1-1)*dt,visc) 
        V[i,j] = Uy_t(meshX[i],meshX[j],(1-1)*dt,visc) 
#        f[i,j] = forcef(meshX[i],meshX[j])
=======
#Computing the cut-off frequency matrix for dealiasing
cut_off = 0.3
c_off=dealiasing(cut_off,Nnod)


for kt in range(5): 
    
    dt = 0.01/3.0**(kt)     
    for i in range(Nnod): 
        for j in range(Nnod): 
            phi[i,j] = phi_t(meshX[i],meshX[j],(1-1)*dt) 
            U[i,j] = Ux_t(meshX[i],meshX[j],(1-1)*dt,visc) 
            V[i,j] = Uy_t(meshX[i],meshX[j],(1-1)*dt,visc) 
    #        f[i,j] = forcef(meshX[i],meshX[j])
>>>>>>> 5f328445eccd1c70481be4c06ece1a1a1a36f314
    
    adv_velxx = U[:] * U[:] 
    adv_velxy = U[:] * V[:] 
    adv_velyy = V[:] * V[:] 
    
    adv_phix = U[:] * phi[:]
    adv_phiy = V[:] * phi[:]    
    
    Vhat = np.fft.fft2(V) 
    Vhat_old = Vhat[:]    
    Uhat = np.fft.fft2(U) 
    Uhat_old = Uhat[:]    
    phihat = np.fft.fft2(phi) 
    phihat_old = phihat[:]    
    jmax = 30*3**kt   #time-steps number        
    
    for k1 in range(Nnod):
        for k2 in range(Nnod):
            fs[k1,k2] = forcef_t(meshX[k1],meshX[k2],(1-1)*dt,schm,visc)    
    fhat = np.fft.fft2(f)
    fshat = np.fft.fft2(fs)
    fshat_old = np.fft.fft2(fs)
    fhat_old = fhat[:]
    adv_velxx_hat = np.fft.fft2(adv_velxx)
    adv_velxy_hat = np.fft.fft2(adv_velxy)
    adv_velyy_hat = np.fft.fft2(adv_velyy)
    adv_velxx_hatold = adv_velxx_hat[:]
    adv_velxy_hatold = adv_velxy_hat[:]
    adv_velyy_hatold = adv_velyy_hat[:]
    
    adv_phix_hat = np.fft.fft2(adv_phix)
    adv_phiy_hat = np.fft.fft2(adv_phiy)
    adv_phix_hatold = adv_phix_hat[:]
    adv_phiy_hatold = adv_phiy_hat[:]
    
#    #Vhat_new=diff_FE(Nnod,fhat,Vhat,alpha_t,dt)
    Uhat_tilde = adv_FE(Nnod,fhat,Uhat,adv_velxx_hat,adv_velxy_hat,visc,dt)
    Vhat_tilde = adv_FE(Nnod,fhat,Vhat,adv_velxy_hat,adv_velyy_hat,visc,dt)
    phihat_new = adv_FE(Nnod,fshat,phihat,adv_phix_hat,adv_phiy_hat,schm,dt)
    
    phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde) 
    
    Uhat_new,Vhat_new=corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt)  
    
    Uhat=Uhat_new[:]
    Vhat=Vhat_new[:]
    phihat=phihat_new[:]
    
    U =  np.real(np.fft.ifft2(Uhat))
    V =  np.real(np.fft.ifft2(Vhat))
#    for i in range(Nnod): 
#        for j in range(Nnod): 
#           U[i,j] = Ux_t(meshX[i],meshX[j],(2-1)*dt,visc) 
#           V[i,j] = Uy_t(meshX[i],meshX[j],(2-1)*dt,visc)
    phi =  np.real(np.fft.ifft2(phihat))
    
    adv_velxx = U[:] * U[:] 
    adv_velxy = U[:] * V[:] 
    adv_velyy = V[:] * V[:] 
  
    adv_phix = U[:] * phi[:]
    adv_phiy = V[:] * phi[:]  
    
    for j in range(2,jmax):
        fshat_old = fshat[:]
        for k1 in range(Nnod):
            for k2 in range(Nnod):
                fs[k1,k2] = forcef_t(meshX[k1],meshX[k2],(j-1)*dt,schm,visc)    
        fshat = np.fft.fft2(fs)
        adv_velxx_hat = (np.fft.fft2(adv_velxx)) * c_off
        adv_velxy_hat = (np.fft.fft2(adv_velxy)) * c_off
        adv_velyy_hat = (np.fft.fft2(adv_velyy)) * c_off
        
        adv_phix_hat = (np.fft.fft2(adv_phix)) * c_off
        adv_phiy_hat = (np.fft.fft2(adv_phiy)) * c_off        
       # Vhat_new=diff_AB(Nnod,fhat,fhat_old,Vhat,Vhat_old,alpha_t,dt)
        #Vhat_new=adv_FE(Nnod,fhat,Vhat,adv_velx_hat,adv_vely_hat,alpha_t,dt)
        Uhat_tilde=adv_AB(Nnod,fhat,fhat_old,Uhat,adv_velxx_hat,adv_velxy_hat,adv_velxx_hatold,adv_velxy_hatold,visc,dt)
        Vhat_tilde=adv_AB(Nnod,fhat,fhat_old,Vhat,adv_velxy_hat,adv_velyy_hat,adv_velxy_hatold,adv_velyy_hatold,visc,dt)
        phihat_new=adv_AB(Nnod,fshat,fshat_old,phihat,adv_phix_hat,adv_phiy_hat,adv_phix_hatold,adv_phiy_hatold,schm,dt)
        
        phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde) 
        
        Uhat_new,Vhat_new=corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt)  
        Uhat_old = Uhat[:]
        Vhat_old = Vhat[:]
        Uhat=Uhat_new[:]
        Vhat=Vhat_new[:]
        
        phihat = phihat_new[:]
        
        U =  np.real(np.fft.ifft2(Uhat))
        V =  np.real(np.fft.ifft2(Vhat))
#        for i in range(Nnod): 
#            for jj in range(Nnod):  
#                U[i,jj] = Ux_t(meshX[i],meshX[jj],(j)*dt,visc) 
#                V[i,jj] = Uy_t(meshX[i],meshX[jj],(j)*dt,visc)
        phi =  np.real(np.fft.ifft2(phihat))
        
        adv_velxx = U[:] * U[:] 
        adv_velxy = U[:] * V[:] 
        adv_velyy = V[:] * V[:]     
        adv_velxx_hatold = adv_velxx_hat[:]
        adv_velxy_hatold = adv_velxy_hat[:]
        adv_velyy_hatold = adv_velyy_hat[:]
        
        adv_phix = U[:] * phi[:]
        adv_phiy = V[:] * phi[:]  
        adv_phix_hatold = adv_phix_hat[:]
        adv_phiy_hatold = adv_phiy_hat[:]
        
        
    for i in range(Nnod):
        for jj in range(Nnod):
            U_ex[i,jj] = Ux_t(meshX[i],meshX[jj],(jmax-1)*dt,visc)
            V_ex[i,jj] = Uy_t(meshX[i],meshX[jj],(jmax-1)*dt,visc)    
            phi_ex[i,jj] = phi_t(meshX[i],meshX[jj],(jmax-1)*dt)
            
 #   err[kt] = abs(np.max(U_ex[:] - U[:])) + abs(np.max(V_ex[:] - V[:]))
    err[kt] = np.linalg.norm(U_ex[:] - U[:],2)+np.linalg.norm(V_ex[:] - V[:],2)+np.linalg.norm(phi_ex[:] - phi[:],2)     
   # err[kt] = np.linalg.norm(phi_ex[:] - phi[:],2)   
#%%    
dt1 = 0.001
dt2 = 0.001/3.0**3.
rate_conver = (np.log(err[3])-np.log(err[0]))/(np.log(dt2)-np.log(dt1))
print(rate_conver)