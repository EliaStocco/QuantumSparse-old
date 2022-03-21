# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:40:40 2022

@author: Elia Stocco
"""
#%%
import numpy as np

from .physical_constants import muB,g

#%%
# https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/
def Rx(phi):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, np.cos(phi),-np.sin(phi)],
                   [ 0, np.sin(phi), np.cos(phi)]])
  
def Ry(theta):
  return np.matrix([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(psi):
  return np.matrix([[ np.cos(psi), -np.sin(psi), 0 ],
                   [ np.sin(psi), np.cos(psi) , 0 ],
                   [ 0           , 0            , 1 ]])

    

#%%
def rotate(EulerAngles,Sx,Sy,Sz):
    N = len(Sx)
    SxR = Sx.copy()
    SyR = Sy.copy()
    SzR = Sz.copy()
    
    for n in range(N):
        
        phi   = EulerAngles[n,0]
        theta = EulerAngles[n,1]
        psi   = EulerAngles[n,2]
        
        R = Rz(psi) @ Ry(theta) @ Rx(phi)
    
        temp = R @ np.asarray([Sx[n],Sy[n],Sz[n]])
        SxR[n] = temp[0,0]    
        SyR[n] = temp[0,1]   
        SzR[n] = temp[0,2]   
        
    return SxR,SyR,SzR

#%%
def magnetic_moment_operator(Sx,Sy,Sz):
    #global g
    #global muB
    Mx = 0#np.sum(g*muB*Sx)
    My = 0#np.sum(g*muB*Sy)
    Mz = 0#np.sum(g*muB*Sz)
    for S in Sx:
        Mx += S
    for S in Sy:
        My += S
    for S in Sz:
        Mz += S
        
    Mx *= g*muB
    My *= g*muB
    Mz *= g*muB
    
    return Mx,My,Mz
