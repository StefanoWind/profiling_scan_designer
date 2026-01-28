# -*- coding: utf-8 -*-
"""
Monte Carlo valitation of scan design framework
"""
import os
cd=os.path.dirname(__file__)
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import utils as utl

plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['savefig.dpi'] = 18

#%% Inputs
azimuth=np.array([0,72,144,216,288,288])#[deg] azimuth angles
elevation=np.array([45,45,45,45,45,90])#[deg] elevation angles

RS_rot=np.array([1,0.8,0.5,0,-0.1,0])#[IEC 61400-1]
err=0.1#relative error on flow field

wds=np.arange(360)#[deg] wind direction loop
L=1000#MC draws

#%% Initialization
assert len(azimuth)==len(elevation), "Azimuth vs. elevation length mismatch"
Nb=len(azimuth)

#zeroing
u_var=np.zeros(len(wds))
u_var2=np.zeros(len(wds))
wd_var=np.zeros(len(wds))
wd_var2=np.zeros(len(wds))
uu_var=np.zeros(len(wds))
uu_var2=np.zeros(len(wds))

 #%% Main
 
#velocity matrix
A=utl.projection_matrix(azimuth, elevation)
A_plus=np.linalg.pinv(A)

# RS matrix
B=utl.rs_matrix(azimuth,elevation)
B_inv=np.linalg.inv(B)

#MC over wind directions
i_wd=0
for wd in wds:
    
    #nominal wind vector
    u0=utl.cos(270-wd)
    v0=utl.sin(270-wd)
    w0=0
    
    #nominal Reynolds stresses [Sathe et al., 2015]
    M_rot=np.array([[utl.sin(wd)**2,utl.cos(wd)**2,0, utl.sin(2*wd),0,0],
                    [utl.cos(wd)**2,utl.sin(wd)**2,0,-utl.sin(2*wd),0,0],
                    [0,0,1,0,0,0],
                    [-0.5*utl.sin(2*wd),0.5*utl.sin(2*wd),0,-utl.cos(2*wd),0,0],
                    [0,0,0,0,-utl.sin(wd),-utl.cos(wd)],
                    [0,0,0,0, utl.cos(wd),-utl.sin(wd)]])
    
    RS0=np.linalg.inv(M_rot)@RS_rot

    #perturbed wind vectors
    vel_los=np.zeros((Nb,L))
    for i in range(L):
        vel_vector=np.zeros((3,Nb))
        vel_vector[0,:]=np.random.normal(0,err,Nb)+u0
        vel_vector[1,:]=np.random.normal(0,err,Nb)+v0
        vel_vector[2,:]=np.random.normal(0,err,Nb)+w0
        for j in range(Nb):
            vel_los[j,i]=A[j,:]@vel_vector[:,j]
            
    #perturbed Reynolds stresses
    var_los=np.zeros((Nb,L))
    for i in range(L):
        RS=np.tile(RS0,(Nb,1)).T+np.random.normal(0,err,(6,Nb))
        for j in range(Nb):
            var_los[j,i]=B[j,:]@RS[:,j]
     
    #reconstruct ws and wd
    vel_rec=A_plus@vel_los
    u_rec=(vel_rec[0,:]**2+vel_rec[1,:]**2)**0.5
    wd_rec=utl.linearize_angle((270-np.degrees(np.arctan2(vel_rec[1,:],vel_rec[0,:])))%360)
    
    #reconstruct uu
    RS_rec=B_inv@var_los
    uu_rec=RS_rec[0,:]*utl.cos(270-wd)**2+RS_rec[1,:]*utl.sin(270-wd)**2+2*RS_rec[3,:]*utl.cos(270-wd)*utl.sin(270-wd)
    
    #store directional error on wind speed
    u_var[i_wd]=np.mean((u_rec-1)**2)
    wd_var[i_wd]=np.mean(((wd_rec - wd + 180) % 360 - 180)**2)
    u_var2[i_wd]=utl.error_ws_wd(azimuth,elevation,wd)[0]*err**2
    wd_var2[i_wd]=utl.error_ws_wd(azimuth,elevation,wd)[1]*err**2
    
    #store directional error on uu
    uu_var[i_wd]=np.mean((uu_rec-RS_rot[0])**2)
    uu_var2[i_wd]=utl.error_uu(azimuth,elevation,wd)*err**2
    
    i_wd+=1
    print(wd)
 
#omnidirectional error on wind speed
u_var_avg=np.mean(u_var)
u_var_avg2=utl.error_ws_wd(azimuth,elevation)[0]*err**2

wd_var_avg=np.mean(wd_var)
wd_var_avg2=utl.error_ws_wd(azimuth,elevation)[1]*err**2

uu_var_avg=np.mean(uu_var)
uu_var2_avg=utl.error_uu(azimuth,elevation)*err**2

#%% Plots
plt.close('all')
plt.figure(figsize=(20,6))

plt.subplot(1,3,1)
plt.plot(wds,u_var,'k',label=f'MC (mean={str(np.round(u_var_avg,4))})')
plt.plot(wds,u_var2,'r',label=f'Theory (mean={str(np.round(u_var_avg2,4))})')
plt.plot(wds,wds**0*u_var_avg,'--k',linewidth=2)
plt.plot(wds,wds**0*u_var_avg2,'--r',linewidth=2)
plt.xlabel(r'$\theta_w$ [$^\circ$]')
plt.ylabel(r'$\sigma^2(\hat{\overline{u}})$')
plt.grid()
plt.ylim([0,np.max(u_var)])
plt.legend()

plt.subplot(1,3,2)
plt.plot(wds,wd_var,'k',label=f'MC (mean={str(np.round(wd_var_avg,4))})')
plt.plot(wds,wd_var2,'r',label=f'Theory (mean={str(np.round(wd_var_avg2,4))})')
plt.plot(wds,wds**0*wd_var_avg,'--k',linewidth=2)
plt.plot(wds,wds**0*wd_var_avg2,'--r',linewidth=2)
plt.xlabel(r'$\theta_w$ [$^\circ$]')
plt.ylabel(r'$\sigma^2(\hat{\overline{\theta}}_w)$')
plt.grid()
plt.ylim([0,np.max(wd_var)])
plt.legend()

plt.subplot(1,3,3)
plt.plot(wds,uu_var,'k',label=f'MC (mean={str(np.round(uu_var_avg,4))})')
plt.plot(wds,uu_var2,'r',label=f'Theory (mean={str(np.round(uu_var2_avg,4))})')
plt.xlabel(r'$\theta_w$ [$^\circ$]')
plt.ylabel(r'$\sigma^2(\hat{\overline{u^\prime}})$')
plt.grid()
plt.legend()
plt.ylim([0,np.max(uu_var)])
plt.legend()
plt.tight_layout()



