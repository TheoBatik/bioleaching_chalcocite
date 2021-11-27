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

N = (101, 101) # number of lattice points on mesh
d = (4, 3) # dimensions of heap
heap = h.Heap(d, N) # init heap 
x, y, mesh = heap.stack() # setup mesh
dx, dy = heap.space_grain(x,y) # define spatial differentials 

######### REACTION SETUP ######################################################

# Set Pyrite factor
heap.params['FPY'][0] = 0

# Set time step
t_steps = [ 'second', 'day', 'month']
t_step = t_steps[1]

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
alpha_dot = alpha_dot.to('1/meter**3/{}'.format(t_step))

######### ENERGY EXCHANGE SETUP ###############################################

# Set finite difference accuracy
acc = 4

# Initiate differential operators
heap.init_ops(acc) # input = finite difference accuracy
Ex = heap.Ec - heap.EL  #- heap.Eg  # no molecular diffusions <= max aeration
Ex = Ex.to('kilojoule/kelvin/meter**3/{}'.format(t_step)) # per day 

# Source function ~ reaction velocity (non-homogoneous)
Esource_fac  = - heap.DeltaH_R * (heap.params['rho_B'][0]* heap.params['G^0'][0]) / (heap.params['sigma_1'][0] * heap.params['X'][0] )
Esource =   Esource_fac *  alpha_dot  

# Boundary conditions:
bc = fd.BoundaryConditions(heap.nodes)
## Dirichlet BC
bc[0,:] = h.params['T_atmos'][0] # left 
bc[-1,:] = h.params['T_atmos'][0] # right
bc[:,-1] = h.params['T_atmos'][0] # top
## Neumann BC
bc[:, 0] = fd.FinDiff(1, dy, 1), 0 # bottom
# mid = round( heap.nodes[0]/2) 
# bc[mid, 1:-1] = fd.FinDiff(0, dx, 1), 0


######### SYSTEM EVOLUTION ###################################################

# temperature container
T = [] 

# End time (seconds, days, months)
end = 5

for t in range(0, end):
    # print('Esource - day {}'.format(day), Esource[0][0:5])
    T.append( heap.energy_exchange(Esource, bc, Ex) )
    alpha_dot = heap.ccu_dot( T[t] * h.kelvin, coxl) 
    Esource =   Esource_fac * alpha_dot.to('1/meter**3/{}'.format(t_step))  # alpha_dot # Restate Esource becuase pde.solve sets it to 298 / 0 .....
    # Update Interior Conditions
    # for i in range(1, heap.nodes[0]-1):
    #     for j in range(1, heap.nodes[1]-1):
    #         bc[i,j] = T[day][i,j]
            
print('T difference Test', T[0][0:5][:] == T[-1][0:5][:])
print('T difference', T[0][3][0] - T[-1][3][0])

# Plot T for day = 0
fig, ax = plt.subplots()
cs = plt.contourf(x,y, T[0].T)
fig.colorbar(cs,  orientation='vertical')
fig.show()

# Plot T for day = end_day
fig, ax = plt.subplots()
cs = plt.contourf(x,y, T[-1].T)
fig.colorbar(cs,  orientation='vertical')
fig.show()






