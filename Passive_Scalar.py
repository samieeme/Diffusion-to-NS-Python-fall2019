"""
Created on Fri Oct 11 14:02:26 2019
@author: samieeme
"""

import numpy as np
from numpy import linalg as LA
import sys
import os

from functions_NS_2D import derivatives, get_diffusion_opt
from functions_NS_2D import gen_IC_vel, gen_IC_vel1, gen_IC_vel2, gen_IC_scalar
from functions_NS_2D import adv_FE, adv_AB, diff_cont, corrector, dealiasing
from functions_NS_2D import adv_FE_phi, adv_AB_phi, plot_Phi_Vor
from functions_NS_2D import get_vorticity, plot_Vel, plot_Vor, plot_Phi
from functions_stats import get_sphere_waven, get_stats_eng, Moments_Vor

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

Kf = 2.0**0.5
Dc = visc/schm

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
cut_off = 0.6#2.0**0.5/3.0
c_off = dealiasing(cut_off,Nnod)

#%%############### Computing constant matrices and arrays #####################

meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)
X,Y = np.meshgrid(meshX,meshX)
sz = Nnod**2

#Computing derivatives' matrices
kxx,kyy,kx,ky = derivatives(Nnod)

operator_diff,den,frac_R = get_diffusion_opt(1.0,dt,visc,Nnod,kxx,kyy)

operator_diff_phi,den_phi,tmp = get_diffusion_opt(alpha,dt,Dc,Nnod,kxx,kyy)
del tmp                                                         

K_sh,K_sh2,K_sh4 = get_sphere_waven(Nnod)

Ntmax = int(t_end/dt)
out_freq = int(t_out_freq/dt)
iprnt_freq = int(out_freq)

out = np.linspace(0,Ntmax,int(Ntmax/out_freq)+1,dtype=int)

Wmax = Nnod*cut_off

tmp = np.nonzero(K_sh <= Kf)
ndx_frc = np.array([tmp[0][1::],tmp[1][1::]]).T
sz_frc = ndx_frc.shape[0]

#%%#################### Generating Initial Conditions #########################

icpath = os.path.join(os.getcwd(),'Out_IC/N'+Nnod_in+'_nu'+visc_in)
Uhat = np.genfromtxt(icpath+'/'+'Uhat.csv', delimiter=',', dtype=complex)
Vhat = np.genfromtxt(icpath+'/'+'Vhat.csv', delimiter=',', dtype=complex)
phihat = np.genfromtxt(icpath+'/'+'Phihat_'+str(Ks)+'.csv', delimiter=',', 
                       dtype=complex)

Uhat *= c_off
Vhat *= c_off
#phihat *= c_off
    
pth = 'Ks_'+Ks_in+'_alpha_'+alpha_in+'_Sc_'+schm_in+'_nu_'+visc_in+'_N_'+Nnod_in
dirpath = os.path.join(os.getcwd(), pth)
os.mkdir(dirpath)
os.chdir(dirpath)

#Write input variables on a file
f_inp = open('inps.txt', 'w')
print(Nnod_in, visc_in, schm_in, dt_in, alpha_in, Ks_in, t_end_in, 
      t_out_freq_in, chk_freq_in, sep=" ", file = f_inp, flush=False)
f_inp.close()

#write output on file
np.savetxt('Uhat.csv', Uhat, delimiter=',')
np.savetxt('Vhat.csv', Vhat, delimiter=',')
np.savetxt('Phihat.csv', phihat, delimiter=',')

f_time = open('time', 'w')
print(0, sep=" ", file = f_time, flush=False)
f_time.close()
         
os.system('mkdir Out_0_chk')
os.system('mv time *.csv Out_0_chk')


iout += 1

TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,Re,T_L,a_frc = get_stats_eng(
        Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)

U = np.fft.ifftn(Uhat)*sz
V = np.fft.ifftn(Vhat)*sz
phi = np.fft.ifftn(phihat)*sz

#%%########## Starting the time-stepping w/ Forward-Euler scheme ##############

adv_velxx = U[:]*U[:]
adv_velxy = U[:]*V[:]
adv_velyy = V[:]*V[:]

adv_phix = U[:]*phi[:]
adv_phiy = V[:]*phi[:] 

adv_velxx_hat = np.fft.fftn(adv_velxx)/sz
adv_velxy_hat = np.fft.fftn(adv_velxy)/sz
adv_velyy_hat = np.fft.fftn(adv_velyy)/sz

adv_phix_hat = np.fft.fftn(adv_phix)/sz
adv_phiy_hat = np.fft.fftn(adv_phiy)/sz
        
adv_velxx_hatold = adv_velxx_hat[:]
adv_velxy_hatold = adv_velxy_hat[:]
adv_velyy_hatold = adv_velyy_hat[:]
adv_phix_hatold = adv_phix_hat[:]
adv_phiy_hatold = adv_phiy_hat[:]

a_frc_old = a_frc

phihat = adv_FE_phi(Nnod,phihat,adv_phix_hat,adv_phiy_hat,dt,kx,ky,
                        operator_diff_phi,den_phi)

Uhat_tilde = adv_FE(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,dt,
                    kx,ky,operator_diff,den,a_frc[:,0],ndx_frc,sz_frc)
Vhat_tilde = adv_FE(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,dt,
                    kx,ky,operator_diff,den,a_frc[:,1],ndx_frc,sz_frc)

phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde,kx,ky,frac_R)

Uhat,Vhat = corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)

Uhat *= c_off
Vhat *= c_off
phihat *= c_off
    
TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,Re,T_L,a_frc = get_stats_eng(
        Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)

time = dt

f1 = open('FlowFeatures.txt', 'w')
print(format(time, '.2f'), 
      format(TKE, '.6f'),
      format(Diss, '.6f'),
      format(Wmax/K_eta, '.2f'),
      format(Re, '.1f'), 
      format(T_L, '.3f'), sep=" ", end='\n', file = f1, flush=False)
f1.close()

phi_flc=phi.real-np.mean(phi.real)
M = Moments_Vor(phi_flc)

f2 = open('Scalar_moments.txt', 'w')
print(format(time, '.2f'), 
      format(M[0], '.5f'), 
      format(M[1], '.5f'), 
      format(M[2], '.5f'), sep=" ", end='\n', file = f2, flush=False)
f2.close()

U = np.fft.ifftn(Uhat)*sz
V = np.fft.ifftn(Vhat)*sz
phi = np.fft.ifftn(phihat)*sz

adv_velxx = U[:]**2
adv_velxy = U[:]*V[:]
adv_velyy = V[:]**2

adv_phix = U[:]*phi[:]
adv_phiy = V[:]*phi[:]

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

    adv_phix_hat = np.fft.fftn(adv_phix)/sz
    adv_phiy_hat = np.fft.fftn(adv_phiy)/sz

    phihat = adv_AB_phi(Nnod,phihat,adv_phix_hat,adv_phiy_hat,
                            adv_phix_hatold,adv_phiy_hatold,dt,kx,ky,
                            operator_diff_phi,den_phi)    
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
    phihat *= c_off

    a_frc_old = a_frc[:]
    
    TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,Re,T_L,a_frc = get_stats_eng(
            Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)
    
    
    
    if nt == iprnt:

        f1 = open('FlowFeatures.txt', 'a')
        print(format(time, '.2f'), 
              format(TKE, '.6f'),
              format(Diss, '.6f'),
              format(Wmax/K_eta, '.2f'),
              format(Re, '.1f'), 
              format(T_L, '.3f'), sep=" ", end='\n', file = f1, flush=False)
        f1.close()

        
        iprnt += iprnt_freq
    
    U = np.fft.ifftn(Uhat)*sz
    V = np.fft.ifftn(Vhat)*sz
    phi = np.fft.ifftn(phihat)*sz

    if nt == out[icnt]:               
        
        phi_flc=phi.real-np.mean(phi.real)
        M = Moments_Vor(phi_flc)

        f2 = open('Scalar_moments.txt', 'a')
        print(format(time, '.2f'), 
              format(M[0], '.5f'), 
              format(M[1], '.5f'), 
              format(M[2], '.5f'), sep=" ", end='\n', file = f2, flush=False)
        f2.close()      
       
        icnt += 1


    if nt == ichk:
        
        #write output on file
        np.savetxt('Uhat.csv', Uhat, delimiter=',')
        np.savetxt('Vhat.csv', Vhat, delimiter=',')
        np.savetxt('Phihat.csv', phihat, delimiter=',')

        f_time = open('time', 'w')
        print(format(time, '.3f'), sep=" ", file = f_time, flush=False)
        f_time.close()        

        os.system('mkdir Out_'+str(iout)+'_chk')
        os.system('mv time *.csv Out_'+str(iout)+'_chk')

        iout += 1
        ichk += ichk_cnt        
        
                       
    adv_velxx = U[:]**2
    adv_velxy = U[:]*V[:]
    adv_velyy = V[:]**2
    adv_phix = U[:]*phi[:]
    adv_phiy = V[:]*phi[:]
    
    adv_velxx_hatold = adv_velxx_hat[:]
    adv_velxy_hatold = adv_velxy_hat[:]
    adv_velyy_hatold = adv_velyy_hat[:]
    adv_phix_hatold = adv_phix_hat[:]
    adv_phiy_hatold = adv_phiy_hat[:]
    
    time += dt
