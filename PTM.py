#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:33:35 2019

@author: akhavans
"""

import numpy as np
from scipy import interpolate
from scipy.stats import levy_stable


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
    Uper=U_periodic(U,Uper,res)
    U_interp=U_intp(Uper,U_interp,res2)
    f_u=interpolate.interp2d(x, x, U_interp, kind='cubic')
    
    Vper=U_periodic(V,Vper,res)
    V_interp=U_intp(Vper,V_interp,res2)
    f_v=interpolate.interp2d(x, x, V_interp, kind='cubic')
    
    Np=Pt.shape[0]
    #noise generation for paricles
    r=levy_stable.rvs(alpha, beta, size=Np)
    ang=np.random.uniform(0,L,Np)
    
    Z1=Diff*r*np.cos(ang)
    Z2=Diff*r*np.sin(ang)
    
    #solving for the new particle location
    for ip in range(0,Np):
        
        u1=f_u(Pt[ip,0],Pt[ip,1])
        u2=f_v(Pt[ip,0],Pt[ip,1])
        
        x1_pr=Pt[ip,0]+dt*u1+Z1[ip]
        x2_pr=Pt[ip,1]+dt*u2+Z2[ip]
        
        x1_pr=pos_adjust(x1_pr,L)
        x2_pr=pos_adjust(x2_pr,L)
           
        u1_pr=f_u(x1_pr,x2_pr)
        u2_pr=f_v(x1_pr,x2_pr)
        
        x1_pr=Pt[ip,0]+0.5*dt*(u1+u1_pr)+Z1[ip]
        x2_pr=Pt[ip,1]+0.5*dt*(u2+u2_pr)+Z2[ip]
        
        Pt[ip,0]=pos_adjust(x1_pr,L)
        Pt[ip,1]=pos_adjust(x2_pr,L)
        
        
    return Pt 