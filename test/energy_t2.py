# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:38:45 2021

@author: Theodore B
"""

# -*- coding: utf-8 -*-
"""
# Estimation of upper bound on heat generation


## Assumptions:
    
    ### Each volume element experiences maximum aeration => concentration liquid oxygen (coxl) is at max.
    ### e.g. Valid close to heap boundary
    ### Heat loss due to conduction, downward liquid flow (not molecular diffusion)
    
"""

######### IMPORTS ############################################################

import heap as h
import numpy as np
import findiff as fd
import matplotlib.pyplot as plt

######### INITIATE HEAP #######################################################

# N = (100, 101)
# N = (101, 31)
# scale = 15

# Set dimensions and number of mesh nodes
d = (10, 3)
scale = 21
N = (d[0]*scale + 1, d[1]*scale + 1)

heap = h.Heap(d, N) # init heap 
x, y, mesh = heap.stack() # setup mesh
dx, dy = heap.space_grain(x,y) # define spatial differentials 

######### REACTION SETUP ######################################################

# Set pyrite factor
heap.params['FPY'][0] = 0


# Change bacterial density
# heap.params['X'][0] = 5*10**(13) / h.cube

# Set time step
t_step_option = 2
t_steps = [ 'second', 'day', 'week', 'month', 'year'] # month and year seem to be unstable?
t_step = t_steps[t_step_option]

# Respiration rate, Vm
T1 = np.linspace(273, 323, 300)*h.kelvin # temperature
Vm = h.Vm(T1) # Respiration rate

# Optimal temperature => maximum respiration rate
i = list(Vm).index(max(Vm)) # index of max Vm
T_opt = T1[i] # optimal temperature

# Maximum concentration of gasesous oxygen
coxg_fac = heap.params['Ox_in_air'][0] * heap.params['rho_air'][0] * h.kg/h.cube
coxg = np.full(N, coxg_fac) * heap.params['rho_air'][0].units


# Maximum concentration of liquid oxygen 
coxl = heap.dissolve(coxg, T_opt) 

# Maximum rate of Copper-Sulfide dissolution => max rate of heat generation
alpha_dot = heap.ccu_dot(T_opt, coxl) 
alpha_dot = alpha_dot.to(f'1/meter**3/{t_step}') 
# alpha_dot = np.zeros(heap.nodes)
# midx = round(heap.nodes[0]/2)
# midy = round(heap.nodes[1]/2)
# alpha_dot[midx:midx + 1, midy: midy +1 ] = 0


######### ENERGY EXCHANGE SETUP ###############################################

# Initiate differential operators
acc = 4 # findiff accuracy
heap.init_ops(acc) 

Ec = heap.Ec # Conduction
EL = heap.EL # Downward liquid flow
Eg = heap.Eg # no convective gas transport <= max aeration

Ex =  Ec + EL
Ex = Ex.to(f'kilojoule/kelvin/meter**3/{t_step}') # per day 

# Source function ~ reaction velocity (non-homogoneous)
Esource_fac  = - heap.DeltaH_R * (heap.params['rho_B'][0]* heap.params['G^0'][0]) / (heap.params['sigma_1'][0] *  heap.params['X'][0] )
Esource =   Esource_fac *  alpha_dot 

# Boundary conditions
bc = fd.BoundaryConditions(heap.nodes)
## Dirichlet BC
bc[0,:] = h.params['T_atmos'][0] # left 
bc[-1,:] = h.params['T_atmos'][0] # right
bc[:,-1] = h.params['T_L'][0] # top
## Neumann BC
bc[:, 0] = fd.FinDiff(1, dy, 1), 0 # bottom
# mid = round( heap.nodes[0]/2) 
# bc[mid, 1:-1] = fd.FinDiff(0, dx, 1), 0



# Solve
T = heap.energy_exchange(Esource, bc, Ex)

fig, ax = plt.subplots()
cs = plt.contourf(x,y, T.T)
plt.title(label = f'Temp. distribution after 1 {t_step} \n of heat accumulation.', loc = 'left')
plt.ylabel(f'{h.meter}')
plt.xlabel(f'{h.meter}' )
cbar = fig.colorbar(cs,  orientation='vertical')
cbar.set_label(f"{heap.params['T_atmos'][0].units}", rotation=270, loc = 'center')
cbar.ax.get_yaxis().labelpad = 15
fig.show()



# ######### SYSTEM EVOLUTION ###################################################


# # temperature container
# T = [] 

# # End time (seconds, days, months)
# end = 31
# Esource = [] 
# Esource.append(Esource_fac * alpha_dot.to(f'1/meter**3/{t_step}'))
# # Esource_init = Esource

# for t in range(0, end):
#     # print('Esource - day {}'.format(day), Esource[0][0:5])
#     # print('Esource 1', Esource[5:8][0])
#     T.append( heap.energy_exchange(Esource[t], bc, Ex) )
#     alpha_dot =  heap.ccu_dot( T_opt , coxl)             #heap.ccu_dot( T[t] * h.kelvin , coxl)  
#     # print('Esource 2', Esource[5][5]) 
#     Esource.append(Esource_fac * alpha_dot.to(f'1/meter**3/{t_step}')) 
#     # alpha_dot # Restate Esource becuase pde.solve sets it to 298 / 0 .....
#     # Update Interior Conditions
#     # for i in range(1, heap.nodes[0]-1):
#     #     for j in range(1, heap.nodes[1]-1):
#     #         bc[i,j] = T[day][i,j]
            
# print('T difference Test', T[0][0:5][:] == T[-1][0:5][:])
# # print('T difference', T[0][3][0] - T[-1][3][0])


# # ######### PLOTS ##############################################################

# # # # Plot 
# # # fig, ax = plt.subplots()
# # # cs = plt.contourf(x,y, Esource_init)
# # # fig.colorbar(cs,  orientation='vertical')
# # # plt.title('Energy Source Initial')
# # # fig.show()

# # # # Plot 
# # # fig, ax = plt.subplots()
# # # cs = plt.contourf(x,y, Esource)
# # # fig.colorbar(cs,  orientation='vertical')
# # # plt.title(label = 'Energy Source Final', loc = 'left')
# # # fig.show()

# # Plot T for day = end_day
# fig, ax = plt.subplots()
# cs = plt.contourf(x,y, T[0].T)
# plt.title(label = f'Temp. distribution after 1 {t_step} \n of heat accumulation.', loc = 'left')
# plt.ylabel(f'{h.meter}')
# plt.xlabel(f'{h.meter}' )
# cbar = fig.colorbar(cs,  orientation='vertical')
# cbar.set_label(f"{heap.params['T_atmos'][0].units}", rotation=270, loc = 'center')
# cbar.ax.get_yaxis().labelpad = 15
# fig.show()


# # Plot T for day = end_day
# fig, ax = plt.subplots()
# cs = plt.contourf(x,y, T[-1].T)
# plt.title(label = f'Temp. distribution after {end} {t_step}s \n of heat accumulation.', loc = 'left')
# plt.ylabel(f'{h.meter}')
# plt.xlabel(f'{h.meter}' )
# cbar = fig.colorbar(cs,  orientation='vertical')
# cbar.set_label(f"{heap.params['T_atmos'][0].units}", rotation=270, loc = 'center')
# cbar.ax.get_yaxis().labelpad = 15
# fig.show()



# # Plot T for day = end_day
# fig, ax = plt.subplots()
# cs = plt.contourf(x,y, Esource[0].T)
# plt.title(label = f'Energy source after 1 {t_step} \n of heat accumulation.', loc = 'left')
# plt.ylabel(f'{h.meter}')
# plt.xlabel(f'{h.meter}' )
# cbar = fig.colorbar(cs,  orientation='vertical')
# cbar.set_label(f"{heap.params['T_atmos'][0].units}", rotation=270, loc = 'center')
# cbar.ax.get_yaxis().labelpad = 15
# fig.show()


# # Plot T for day = end_day
# fig, ax = plt.subplots()
# cs = plt.contourf(x,y, Esource[-1].T)
# plt.title(label = f'Energy source after {end} {t_step}s \n of heat accumulation.', loc = 'left')
# plt.ylabel(f'{h.meter}')
# plt.xlabel(f'{h.meter}' )
# cbar = fig.colorbar(cs,  orientation='vertical')
# cbar.set_label(f"{heap.params['T_atmos'][0].units}", rotation=270, loc = 'center')
# cbar.ax.get_yaxis().labelpad = 15
# fig.show()


