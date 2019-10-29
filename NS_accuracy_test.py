"""
Created on Fri Oct 11 14:02:26 2019
@author: samieeme
"""

import numpy as np
from numpy import linalg as LA
import sys
import os

from functions_NS_2D import derivatives, get_diffusion_opt, check_div_free
from functions_NS_2D import gen_IC_vel, gen_IC_vel1, gen_IC_vel2
from functions_NS_2D import adv_FE, adv_AB, diff_cont, corrector, dealiasing
from functions_NS_2D import get_vorticity, plot_Vel, plot_Vor
from functions_stats import get_sphere_waven, get_stats_eng, Moments_Vor

#%%###################### Setting up parameters ###############################

Nnod_in = sys.argv[1]
Nnod = int(Nnod_in)
visc_in = sys.argv[2]
visc = float(visc_in)
dt_in = sys.argv[3]
dt = float(dt_in)
alpha_in = sys.argv[4]
alpha = float(alpha_in)
Kf = 2.0*2.0**0.5

#Final simulation time and output time
t_end_in = sys.argv[5]
t_end = float(t_end_in)
t_out_freq_in = sys.argv[6]
t_out_freq = float(t_out_freq_in)
chk_freq_in = sys.argv[7]
chk_freq = int(chk_freq_in)

ichk_cnt = int(1.0/(dt*chk_freq))
ichk = ichk_cnt
iout = 0

#Selecting a method for velocity I.C.
ic_mthd_in = sys.argv[8]
ic_mthd = int(ic_mthd_in)

#Computing the cut-off frequency matrix for dealiasing
cut_off = 2.0/3.0
c_off = dealiasing(cut_off,Nnod)

#Write input variables on a file
f_inp = open('inps.txt', 'w')
print(Nnod_in, visc_in, dt_in, alpha_in, t_end_in, t_out_freq_in, chk_freq_in,
      sep=" ", file = f_inp, flush=False)
f_inp.close()

#%%############### Computing constant matrices and arrays #####################

meshX = np.linspace(0,2*np.pi,Nnod+1)
meshX = np.delete(meshX,Nnod,None)
X,Y = np.meshgrid(meshX,meshX)
sz = Nnod**2

#Computing derivatives' matrices
kxx,kyy,kx,ky = derivatives(Nnod)

operator_diff,den,frac_R = get_diffusion_opt(alpha,dt,visc,Nnod,kxx,kyy)

K_sh,K_sh2,K_sh4 = get_sphere_waven(Nnod)

Ntmax = int(t_end/dt)
out_freq = int(t_out_freq/dt)
iprnt_freq = int(out_freq/5.)

out = np.linspace(0,Ntmax,int(Ntmax/out_freq)+1,dtype=int)

Wmax = Nnod*(2**0.5)/3.0

tmp = np.nonzero(K_sh <= Kf)
ndx_frc = np.array([tmp[0][1::],tmp[1][1::]]).T
sz_frc = ndx_frc.shape[0]

#%%#################### Generating Initial Conditions #########################

if ic_mthd == 0:
#    Uhat,Vhat = gen_IC_vel2(Nnod)
    Uhat,Vhat = gen_IC_vel1(Nnod,Kf)
#    Uhat,Vhat=gen_IC_vel(Nnod)

    np.savetxt('Uhat.csv', Uhat, delimiter=',')
    np.savetxt('Vhat.csv', Vhat, delimiter=',')
    
    os.system('mkdir Out_'+str(iout)+'_chk')
    os.system('mv *.csv Out_'+str(iout)+'_chk')

    
else:
    icpath=os.path.join(os.getcwd(),'Out_IC')
    Uhat = np.genfromtxt(icpath+'/'+'Uhat.csv', delimiter=',',dtype=complex)
    Vhat = np.genfromtxt(icpath+'/'+'Vhat.csv', delimiter=',',dtype=complex)

iout += 1

div, Uhat, Vhat = check_div_free(sz,Uhat,Vhat,kx,ky)


TKE,Enst,eta,Diss,K_eta,int_l,mic_l,Re_l,Re,T_L,a_frc = get_stats_eng(
        Uhat,Vhat,visc,K_sh,K_sh2,K_sh4,ndx_frc,sz_frc)

Vor = get_vorticity(sz,Uhat,Vhat,kx,ky)

U = np.fft.ifftn(Uhat)*sz
V = np.fft.ifftn(Vhat)*sz

#plot_Vel(X,Y,U,V,0,1,'seismic')

#plot_Vor(X,Y,Vor,0.0,1,'seismic')

#%%########## Starting the time-stepping w/ Forward-Euler scheme ##############

adv_velxx = U[:]*U[:]
adv_velxy = U[:]*V[:]
adv_velyy = V[:]*V[:]

adv_velxx_hat = np.fft.fftn(adv_velxx)/Nnod**2*c_off
adv_velxy_hat = np.fft.fftn(adv_velxy)/Nnod**2*c_off
adv_velyy_hat = np.fft.fftn(adv_velyy)/Nnod**2*c_off

adv_velxx_hatold = adv_velxx_hat[:]
adv_velxy_hatold = adv_velxy_hat[:]
adv_velyy_hatold = adv_velyy_hat[:]

a_frc_old = a_frc

Uhat_tilde,adv_velx_hat = adv_FE(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,dt,
                    kx,ky,operator_diff,den,a_frc[:,0],ndx_frc,sz_frc)
Vhat_tilde, adv_vely_hat = adv_FE(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,dt,
                    kx,ky,operator_diff,den,a_frc[:,1],ndx_frc,sz_frc)

phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde,kx,ky,frac_R)

Uhat,Vhat = corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)

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

Vor = get_vorticity(sz,Uhat,Vhat,kx,ky)

M = Moments_Vor(Vor)

f2 = open('Vorticity_moments.txt', 'w')
print(format(time, '.2f'), 
      format(M[0], '.5f'), 
      format(M[1], '.5f'), 
      format(M[2], '.3f'), sep=" ", end='\n', file = f2, flush=False)
f2.close()

U = np.fft.ifftn(Uhat)*sz
V = np.fft.ifftn(Vhat)*sz

adv_velxx = U[:]**2
adv_velxy = U[:]*V[:]
adv_velyy = V[:]**2

icnt = 1
time = 2.0*dt
iprnt = iprnt_freq

#%%############################################################################
############################# Time-Stepping loop ##############################
## 2nd-order Adams-Bashforth scheme for advective & artificial forcing terms ##
################ Crank-Nickelson scheme for diffusion terms ###################
###############################################################################

for nt in range(2,Ntmax+1):

    adv_velxx_hat = (np.fft.fftn(adv_velxx)/sz)*c_off
    adv_velxy_hat = (np.fft.fftn(adv_velxy)/sz)*c_off
    adv_velyy_hat = (np.fft.fftn(adv_velyy)/sz)*c_off

    Uhat_tilde = adv_AB(Nnod,Uhat,adv_velxx_hat,adv_velxy_hat,adv_velxx_hatold,
                        adv_velxy_hatold,dt,kx,ky,operator_diff,den,a_frc[:,0],
                        a_frc_old[:,0],ndx_frc,sz_frc)
    Vhat_tilde = adv_AB(Nnod,Vhat,adv_velxy_hat,adv_velyy_hat,adv_velxy_hatold,
                        adv_velyy_hatold,dt,kx,ky,operator_diff,den,a_frc[:,1],
                        a_frc_old[:,1],ndx_frc,sz_frc)

    phat = diff_cont(Nnod,Uhat_tilde,Vhat_tilde,kx,ky,frac_R)

    Uhat,Vhat = corrector(Nnod,Uhat_tilde,Vhat_tilde,phat,dt,kx,ky)

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

    if nt == out[icnt]:               
        
        Vor = get_vorticity(sz,Uhat,Vhat,kx,ky)

#        plot_Vor(X,Y,Vor,time,icnt+1,'seismic')
               
        M = Moments_Vor(Vor)
        
        f2 = open('Vorticity_moments.txt', 'a')
        print(format(time, '.2f'), 
              format(M[0], '.5f'), 
              format(M[1], '.5f'), 
              format(M[2], '.3f'), sep=" ", end='\n', file = f2, flush=False)
        f2.close()      
       
        icnt += 1


    if nt == ichk:

        np.savetxt('Uhat.csv', Uhat, delimiter=',')
        np.savetxt('Vhat.csv', Vhat, delimiter=',')        
        np.savetxt('Velhat_xx_old.csv', adv_velxx_hat, delimiter=',')
        np.savetxt('Velhat_xy_old.csv', adv_velxy_hat, delimiter=',')
        np.savetxt('Velhat_yy_old.csv', adv_velyy_hat, delimiter=',')

        os.system('mkdir Out_'+str(iout)+'_chk')
        os.system('mv *.csv Out_'+str(iout)+'_chk')

        iout += 1
        ichk += ichk_cnt        
        
                       
    adv_velxx = U[:]**2
    adv_velxy = U[:]*V[:]
    adv_velyy = V[:]**2
    adv_velxx_hatold = adv_velxx_hat[:]
    adv_velxy_hatold = adv_velxy_hat[:]
    adv_velyy_hatold = adv_velyy_hat[:]
    
    time += dt

#plot_Vel(X,Y,U,V,time,icnt+1,'seismic')
    
os.system('mkdir sim_N'+Nnod_in+'_nu'+visc_in+'_dt'+dt_in+'_alpha'+alpha_in)
os.system('mv *.txt Out_* sim_N*')
os.system('mv sim_N* ../')