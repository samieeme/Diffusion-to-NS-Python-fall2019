#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:47:59 2019

@author: aliakhavan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Helvetica']})
rc('text', usetex=True)

from functions_NS_2D import derivatives, get_vorticity
from functions_NS_2D import plot_Phi, plot_Vor 
from functions_stats import get_sphere_waven


#%%############################################################################
###############################################################################
PI=np.pi

Ks = 1
alpha = 1.0
Sc = 1.0

pth = 'Ks_'+str(Ks)+'_alpha_'+str(alpha)+'_Sc_'+str(Sc)
dirpath = os.path.join(os.getcwd(), pth)
os.chdir(dirpath)

inps = np.genfromtxt('inps.txt', delimiter=' ')

#%%############################################################################
###############################################################################

res = int(inps[0])
sz = res**2

meshX = np.linspace(0,2*PI,res+1)
meshX = np.delete(meshX,res,None)
X,Y = np.meshgrid(meshX,meshX)

#Computing derivatives' matrices
kxx,kyy,kx,ky = derivatives(res)

nu = float(inps[1])
nfiles = int(inps[6]*inps[8])

time = np.linspace(0.0,inps[6],nfiles+1)

counter = 0

for it in range(0,nfiles+1,5):
    
    tmp_pth = 'Out_'+str(it)+'_chk'
    fpath = os.path.join(dirpath, tmp_pth)
    
    uhat = np.genfromtxt(os.path.join(fpath,'Uhat.csv'), 
                         delimiter=',',dtype=complex)
    vhat = np.genfromtxt(os.path.join(fpath,'Vhat.csv'), 
                         delimiter=',',dtype=complex)
    phihat = np.genfromtxt(os.path.join(fpath,'Phihat.csv'), 
                           delimiter=',',dtype=complex)
    
    phi = np.fft.ifftn(phihat)*sz
    
    Vor=get_vorticity(sz,uhat,vhat,kx,ky)
    plot_Vor(X,Y,Vor,time[it],counter,'coolwarm')
    
    plot_Phi(X,Y,phi,time[it],counter,'coolwarm')
    
    counter += 1



