# -*- coding: utf-8 -*-
"""

Pseudo-Test unit to incorporate dimensionless PDE to increase stability.

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
Delt = 3 # Full duration of simulation: choose time-step units below
# Set time step units:
t_step_option = 1
t_steps = [ 'second', 'day', 'week', 'month', 'year'] # month and year do not seem to work well...
t_step = t_steps[t_step_option]

# Finite difference accuracy 
acc = 4

##### INIT HEAP ###############################################################

# Dimensions D = (W, H, Delt):
W = 10 # Width -> x
H = 3 # Height -> y
D = (W, H, Delt) 

# Mesh and differentials:
scale_xy = 10
scale_t = 5
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
hp.init_ops(acc) # dimensionless by default 

# Define energy exchange operator
Ec = hp.Ec_fac_dim * hp.d2T_dXY2 
El = hp.El_fac_dim * hp.dT_dY
Te = hp.Te_fac_dim * hp.dT_dtau
Ex = Ec - El - Te
# Ex = Ex.to(f'kilojoule/kelvin/meter**3/{t_step}') 

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


print(T[:, :, 0] == T[:, :, -1])













