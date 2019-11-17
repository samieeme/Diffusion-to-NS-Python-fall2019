#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:33:35 2019

@author: akhavans
"""

import numpy as np
from scipy import interpolate
from scipy.stats import levy_stable
import time


def pos_adjust_v(x,L):
    
    #adjusting particles w/ "x > L"
    dec = np.modf(x[np.nonzero(x > L)]/L)
    x[np.nonzero(x > L)] = dec[0]*L
    
    #adjusting particles w/ "x < 0.0"
    dec = np.modf(np.abs(x[np.nonzero(x < 0.0)]/L))
    x[np.nonzero(x < 0.0)] = (1.0-dec[0])*L

    return x

def pos_adjust(x,L):
    if x > L:
        dec=np.modf(x/L)
        x = dec[0]*L
    elif x < 0.0:
        dec=np.modf(np.abs(x/L))
        x = (1.0-dec[0])*L
    return x

def init_particle(Phi_p,res,Npcell,L):
    
    Phi=Phi_p.real
    
    pos=np.nonzero(Phi>0.0)
    
    npos=Phi[pos[0],pos[1]].size
    
    ################################################
    
    Pt_1=np.array([])
    Pt_2=np.array([])
    dx=L/res
    
    for ndx in range(0,npos):
        lhd = np.random.rand(Npcell,2)
        
        for i in range(0,2):
            lhd[:,i]=(pos[i][ndx]-0.5+lhd[:,i])*dx
            if pos[i][ndx] == 0:
                lhd[i,np.nonzero(lhd[i,:]<0)] = lhd[i,np.nonzero(lhd[i,:]<0)]+L
        
        for i in range(0,Npcell):
            lhd[i,0]=pos_adjust(lhd[i,0],L)
            lhd[i,1]=pos_adjust(lhd[i,1],L)
            
        Pt_1=np.append(Pt_1,lhd[:,0])
        Pt_2=np.append(Pt_2,lhd[:,1])
    
    print('Number of particles: '+str(Pt_1.size))
    
    out=np.zeros((Pt_1.size,2))
    out[:,0]=Pt_1
    out[:,1]=Pt_2
    
    return out

def U_periodic(U,Uper,res):
    Uper[0:res,0:res]=U
    Uper[:,res]=Uper[:,0]
    Uper[res,:]=Uper[0,:]
    
    return Uper


def U_intp(U,U_interp,res2):

    U_interp[1:res2+1,1:res2+1]=U
    
    U_interp[0,:]=U_interp[res2-1,:]
    U_interp[res2+1,:]=U_interp[1,:]
    
    U_interp[:,0]=U_interp[:,res2-1]
    U_interp[:,res2+1]=U_interp[:,1]
    
    U_interp[0,0]=U_interp[res2,res2]
    U_interp[0,res2+1]=U_interp[res2,1]
    U_interp[res2+1,0]=U_interp[1,res2]
    U_interp[res2+1,res2+1]=U_interp[1,1]
    
    return U_interp

def Particle_Tracking(Pt,U,V,x,res,dx,L,dt,Diff,alpha,beta):
    
    res2 = res+1

    U_interp=np.zeros((res2+2,res2+2))
    V_interp=np.zeros((res2+2,res2+2))

    Uper=np.zeros((res2,res2))
    Vper=np.zeros((res2,res2))
    
    #velocity interpolatin
    starttime = time.process_time()    
    Uper=U_periodic(U,Uper,res)
    U_interp=U_intp(Uper,U_interp,res2)
    f_u=interpolate.RectBivariateSpline(x, x, U_interp, kx=5, ky=5)
    
    Vper=U_periodic(V,Vper,res)
    V_interp=U_intp(Vper,V_interp,res2)
    f_v=interpolate.RectBivariateSpline(x, x, V_interp, kx=5, ky=5)

    endtime = time.process_time()
    print(endtime - starttime)
    
    Np=Pt.shape[0]
    #noise generation for paricles
    r=levy_stable.rvs(alpha, beta, size=Np)
    ang=np.random.uniform(0,L,Np)
    
    Z1=Diff*r*np.cos(ang)
    Z2=Diff*r*np.sin(ang)
    
    starttime = time.process_time()
    #solving for the new particle location        
    u1=f_u.ev(Pt[:,0],Pt[:,1])
    u2=f_v.ev(Pt[:,0],Pt[:,1])
    
    x1_pr=Pt[:,0]+dt*u1+Z1
    x2_pr=Pt[:,1]+dt*u2+Z2
    
    x1_pr=pos_adjust_v(x1_pr,L)
    x2_pr=pos_adjust_v(x2_pr,L)
       
    u1_pr=f_u.ev(x1_pr,x2_pr)
    u2_pr=f_v.ev(x1_pr,x2_pr)
    
    x1_pr=Pt[:,0]+0.5*dt*(u1+u1_pr)+Z1
    x2_pr=Pt[:,1]+0.5*dt*(u2+u2_pr)+Z2
    
    Pt[:,0]=pos_adjust_v(x1_pr,L)
    Pt[:,1]=pos_adjust_v(x2_pr,L)
        
    endtime = time.process_time()
    print(endtime - starttime)
    
    return Pt 