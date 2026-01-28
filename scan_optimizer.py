# -*- coding: utf-8 -*-
"""
Optimization of scan
"""
import os
cd=os.path.dirname(__file__)
import os
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import minimize
import utils as utl

plt.close('all')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['savefig.dpi'] = 500

#%% Inputs
N_opt=20#number of optimizations
ele_min=70#[deg] minimum elevation
x_con=0#[m] x of contraint
z_con=1000#[m] z of constraint
Nb=6#number of beams
weight=0.5#weight of wind speed vs. wind speed variance

#%% Functions
def error(azi_ele,weight):
    
    #extract scan geometry
    Nb=int(len(azi_ele)/2)
    azimuth=azi_ele[:Nb]
    elevation=azi_ele[Nb:]
    
    assert len(azimuth)==len(elevation), "Azimuth vs. elevation length mismatch"

    #calculate errors
    var_u=utl.error_ws_wd(azimuth, elevation)[0]
    var_uu=utl.error_uu(azimuth, elevation)
    
    return var_u*weight+var_uu*(1-weight)

#%% Initialization
f_opt=[]
azi_opt=[]
ele_opt=[]
f_best=[]
azi_best=[]
ele_best=[]

#%% Main
ctr=0
while len(f_opt)<N_opt:
    
    azi0=np.random.rand(Nb)*360
    ele0=np.random.rand(Nb)*(90-ele_min)+ele_min
    
    #constraints
    cons=({'type': 'ineq', 'fun': lambda x: min(1/utl.tan(x[6:])*utl.cos(90-x[:6]))-x_con/z_con})

    if x_con<=0:
        bous=((0,360),(0,360),(0,360),(0,360),(0,360),(0,360),
              (ele_min,90),(ele_min,90),(ele_min,90),(ele_min,90),(ele_min,90),(ele_min,90))
    else:
        ele_max=utl.arctan(z_con/x_con)    
        bous=((-90,90),(-90,90),(-90,90),(-90,90),(-90,90),(-90,90),
              (ele_min,ele_max),(ele_min,ele_max),(ele_min,ele_max),(ele_min,ele_max),(ele_min,ele_max),(ele_min,ele_max))
  
    #optimization
    res = minimize(error, np.append(azi0,ele0),
                   method='SLSQP', tol=1e-7,
                   bounds=bous,constraints=cons,
                   options={'maxiter':1000},  args=(weight))
    
    if res.success==True: 

        azi_opt.append(np.around(res.x[:Nb],2))
        ele_opt.append(np.around(res.x[Nb:],2))
        f_opt.append(np.around(res.fun,5))
    
        f_best.append(min(f_opt))
        azi_best.append(azi_opt[np.where(f_opt==f_best[-1])[0][0]])
        ele_best.append(ele_opt[np.where(f_opt==f_best[-1])[0][0]])
        
        ctr+=1
        print(f'{ctr}/{N_opt} successfull optimizations completed')
    else:
        print(res.message)
    
    
   
#%% Plots
fig=plt.figure(figsize=(20,15))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=60, azim=280)

for b,a in zip(ele_best[-1],90-azi_best[-1]):       
    
    r=1/utl.sin(b)
    x=np.append(0,utl.cos(b)*utl.cos(a)*r)
    y=np.append(0,utl.cos(b)*utl.sin(a)*r)
    z=np.append(0,utl.sin(b)*r)
    
    ax.plot3D(x,y,z,'g',linewidth=2)
 
x = [x_con/z_con,x_con/z_con,x_con/z_con,x_con/z_con]
y = [-1,-1,1,1]
z = [1,0,0,1]
verts = [list(zip(x,y,z))]
ax.add_collection3d(Poly3DCollection(verts, facecolors='k', linewidths=1, alpha=0.2))
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_aspect('equal')
plt.title(r'$'+str(weight)+r'\sigma^2(\hat{\overline{u}}) + '\
             +str(1-weight)+'\sigma^2(\hat{\overline{u^{\prime 2}}})='+str(f_best[-1])+r'$'+'\n'+
          r'$\theta=['+str(utl.vec2str(np.round(azi_best[-1],1),', ','%06.2f'))+']^\circ$'+'\n'+
          r'$\beta=['+str(utl.vec2str(np.round(ele_best[-1],1),', ','%06.2f'))+']^\circ$')
ax.set_xlabel(r'$x/z_\text{cos}$')
ax.set_ylabel(r'$y/z_\text{cos}$')
ax.set_zlabel(r'$z/z_\text{cos}$')

plt.grid()

        
        

