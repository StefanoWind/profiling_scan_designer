# -*- coding: utf-8 -*-
"""
Utilities for scan optimizazions
"""
import numpy as np

def vec2str(vec,separator=' ',format='%f'):
    s=''
    for v in vec:
        s=s+format % v+separator
    return s[:-len(separator)]

def tan(x):
    return np.tan(np.radians(x))

def arctan(x):
    return np.degrees(np.arctan(x))

def cos(x):
    return np.cos(np.radians(x))

def sin(x):
    return np.sin(np.radians(x))

def linearize_angle(x):
    '''
    Remove 360-deg discontinuity in angle series
    '''
    dang=(x[1:] - x[:-1] + 180) % 360 - 180
        
    return x[0]+np.cumsum(np.append(0,dang))

def projection_matrix(azimuth,elevation):
    '''
    Cartesian->LOS matrix
    '''
    Nb=len(azimuth)
    A=np.zeros((Nb,3))
    for i in range(Nb):
        A[i,:]=np.array([cos(elevation[i])*cos(90-azimuth[i]),
                         cos(elevation[i])*sin(90-azimuth[i]),
                         sin(elevation[i])])
    return A

def rs_matrix(azimuth,elevation):
    '''
    Reynolds stress reconstruction matrix
    '''
    Nb=len(azimuth)
    B=np.zeros((Nb,6))
    B[:,0]=cos(elevation)**2*cos(90-azimuth)**2
    B[:,1]=cos(elevation)**2*sin(90-azimuth)**2
    B[:,2]=sin(elevation)**2
    B[:,3]=2*cos(elevation)**2*cos(90-azimuth)*sin(90-azimuth)
    B[:,4]=2*cos(elevation)*sin(elevation)*cos(90-azimuth)
    B[:,5]=2*cos(elevation)*sin(elevation)*sin(90-azimuth)
    
    return B

def error_ws_wd(azimuth,elevation,wd=None):
    '''
    Variance of wind speed
    '''
    
    Nb=len(azimuth)
    
    A=projection_matrix(azimuth, elevation)
    #LOS->cartesian matrix
    try:
        A_plus=np.linalg.pinv(A)
    except:
        return np.inf 
    
    #expanded solution matrix
    M=np.zeros((3,Nb*3))
    for i in range(3):
        for j in range(Nb):
            for k in range(3):
                M[i,j*3+k]=A_plus[i,j]*A[j,k]
    
    #error covariance
    S=M@M.T
    
    #error propagation
    if wd==None:
        J_Jt_avg=np.array([[0.5,0,0],
                           [0,0.5,0],
                           [0, 0, 0]])
        var_ws=np.trace(S@J_Jt_avg)
        var_wd=np.trace(S@J_Jt_avg)*180**2/np.pi**2
    else:
        J=np.array([cos(270-wd),sin(270-wd),0])
        var_ws=J@S@J.T
        
        J=np.array([sin(270-wd),-cos(270-wd),0])
        var_wd=J@S@J.T*180**2/np.pi**2
        
    return var_ws,var_wd


def error_uu(azimuth,elevation,wd=None):
    
    '''
    Error on wind speed std
    
    '''
    Nb=len(azimuth)
    
    #RS to LOS variance matrix
    B=rs_matrix(azimuth, elevation)
    
    try:
        B_inv=np.linalg.inv(B)
    except:
        return np.inf          
    
    #expanded solution matrix
    M=np.zeros((6,Nb*6))
    for i in range(6):
        for j in range(Nb):
            for k in range(6):
                M[i,j*6+k]=B_inv[i,j]*B[j,k]
                    
    #error covariance
    S=M@M.T
    
    #error propagation
    if wd==None:
        J_Jt_avg=np.array([[3/8, 1/8, 0,   0, 0, 0],
                           [1/8, 3/8, 0,   0 ,0, 0],
                           [0,   0,   0,   0, 0, 0],
                           [0,   0,   0, 1/2, 0, 0],
                           [0,   0,   0,   0, 0, 0],
                           [0,   0,   0,   0, 0, 0]])
        var_uu=np.trace(S@J_Jt_avg)
    else:
        J=np.array([cos(270-wd)**2,sin(270-wd)**2,0,2*cos(270-wd)*sin(270-wd),0,0])
        var_uu=J@S@J.T
    return var_uu