# -*- coding: utf-8 -*-
"""
Psuedo-test unit for upgrades to heap's ability to evolve in time.'

INSTRUCTIONS:
    
    1) Init Heap( d, N )
    2) Stack Heap => x, y, t, mesh
    3) Set up IC's
    4) Call diffs() => dx, dy, dx
    5) Call heap.init_ops( accuracy )
    6) Define PDE diff. operator
    7) Call heap.energy_exchange(source, bc, operator)

"""

##### IMPORTS #################################################################

import findiff as fd
import matplotlib.pyplot as plt
import numpy as np
# import os
# cwd = os.getcwd()
# d_main = cwd + '\\main'
# os.chdir(d_main)
import heap

##### BASE UNITS ##############################################################

meter = heap.meter
second = heap.second
kelvin = heap.kelvin
kPa = heap.kPa
kg = heap.kg
mol = heap.mol
kJ = heap.kJ

##### CONTROL #################################################################

# Time
Delt = 10 # Full duration of simulation: choose time-step units below
# Set time step units:
t_step_option = 0
t_steps = [ 'second', 'day', 'week', 'month', 'year'] 
t_step = t_steps[t_step_option]

# Finite difference accuracy 
acc = 4

##### INIT HEAP ###############################################################

# Dimensions D = (W, H, Delt):
W = 10 # Width -> x
H = 3 # Height -> y
D = (W, H, Delt) 

# Mesh and differentials:
scale_xy = 5
scale_t = 2
N = (W * scale_xy + 1, H*scale_xy + 1, Delt * scale_t + 1) # Mesh shape
hp = heap.Heap( D, N ) # Init heap
x, y, t, mesh = hp.stack() # Get mesh
Nxy = (hp.ms[0], hp.ms[1]) # 2d spatial slice
dx, dy, dt = hp.diffs(x, y, t) # Get differentials

##### STATE VARIABLES  ########################################################

T = None

##### INITIAL CONDITIONS ######################################################

# Atmospheric temp. throughout heap
T_i = np.full(Nxy, hp.params['T_atmos'][0].magnitude ) * kelvin 
bc = fd.BoundaryConditions(hp.ms)
bc[:, :, 0] = T_i


##### SOURCE FUNCTION #########################################################

# Respiration rate, Vm( T ):
T1 = np.linspace(273, 323, 300) * kelvin # temperature
Vm = hp.Vm(T1) # Respiration rate by temperature

# Optimal temperature => maximum respiration rate:
i = list(Vm).index(max(Vm)) # index of max(Vm)
T_opt = T1[i] # optimal temperature

# Maximum concentration of gasesous oxygen
coxg_fac = hp.coxg_fac 
coxg = np.full(N, coxg_fac.magnitude) * hp.params['rho_air'][0].units
coxl = hp.dissolve(coxg, T_opt)

# Rate of reaction:
alpha_dot = hp.ccu_dot(T_opt, coxl) 
alpha_dot = alpha_dot.to(f'1/meter**3/{t_step}')

# Source function
Esource_fac  = - hp.DeltaH_R * hp.source_fac
Esource =   Esource_fac *  alpha_dot


##### ENERGY EXCHANGE #########################################################

# Initiate differential operators
hp.init_ops(acc)

# Define Energy Exchange operator (Ex):
Ec = hp.Ec # Conduction 
EL = hp.EL # Downward liquid flow
# Eg = hp.Eg # Convection 
Te = hp.Te # Time-evolution
fudge_fac = 1 # 10**(-6) # to test behaviour
Ex = Ec - EL - fudge_fac * Te 
Ex = Ex.to(f'kilojoule/kelvin/meter**3/{t_step}') 

# Boundary conditions:
## Dirichlet BC
bc[0,:, :] = hp.params['T_atmos'][0] # left 
bc[-1, :, :] = hp.params['T_atmos'][0] # right
# bc[0, :, :] = hp.params['T_atmos'][0] # bottom
bc[:,-1, :] = hp.params['T_L'][0] # top
## Neumann BC
for i in range(0, Delt - 1):
    bc[:, 0, i] = fd.FinDiff(1, dy, 1), 0 # bottom
    

# Solve
T = hp.energy_exchange(Esource, bc, Ex)


#### PLOTS ####################################################################


# T Plot Initial
fig, ax = plt.subplots()
cs = plt.contourf(x,y, T[:, :, 1].T)
plt.title(label = f'Temp. distribution after 1 {t_step} \n of heat accumulation.', loc = 'left')
plt.ylabel(f'{meter}')
plt.xlabel(f'{meter}' )
cbar = fig.colorbar(cs,  orientation='vertical')
cbar.set_label(f"{hp.params['T_atmos'][0].units}", rotation=270, loc = 'center')
cbar.ax.get_yaxis().labelpad = 15
fig.show()

# T Plot Final
fig, ax = plt.subplots()
cs = plt.contourf(x,y, T[:, :, -1].T)
plt.title(label = f'Temp. distribution after {Delt} {t_step}s \n of heat accumulation.', loc = 'left')
plt.ylabel(f'{meter}')
plt.xlabel(f'{meter}' )
cbar = fig.colorbar(cs,  orientation='vertical')
cbar.set_label(f"{hp.params['T_atmos'][0].units}", rotation=270, loc = 'center')
cbar.ax.get_yaxis().labelpad = 15
fig.show()


# T[:, :, 0] == T[:, :, -1]








