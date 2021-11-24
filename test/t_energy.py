# -*- coding: utf-8 -*-
"""
Checking the bahviour of the energy exchange step
"""
import heap as h
import numpy as np
import findiff as fd
import matplotlib.pyplot as plt



##################### Initiate Heap ##########################################

N = (21, 41)
d = (10, 20)
x, y, mesh  =  h.stack( d , N )


DX, DY = h.space_grain(mesh)
Y = mesh[1][:][:]
X = mesh[0][:][:]

## Oxygen balance
C_L = np.full(N, 0.006) * h.kg/h.cube # N = size of heap
CL_mol = C_L/h.c['M_Ox'][0]/h.mol # dividing by mol because I still need to figure out what the units of alpha should be....
alpha = np.zeros(N) / h.cube
T = np.full(N, 250) * h.kelvin
dt = 5/100*h.second

# Reaction rate
a_dot = h.alpha_dot(T, C_L) # outside of the leach method because it will be needed elsewhere
alpha_formed = a_dot*dt
alpha += alpha_formed
ox_to_alpha = 2.5/2 # Oxygen to Cu2SO4 based on the stoiciometric coefficients
# oxygen_lost = ox_to_alpha*alpha_formed
CL_mol -= ox_to_alpha*alpha_formed

############# COUNDUCTION ####################################################

shape = N

# Conduction Operator
E_c = h.E_c
f = a_dot.magnitude
E_source = h.E_source

## Boundary Conditions
bc = fd.BoundaryConditions(N)
# Dirichlet BC
bc[0,:] = h.params['T_atmos'][0] # left 
bc[-1,:] = h.params['T_atmos'][0] # right
bc[:,-1] = h.params['T_atmos'][0] # top
# Neumann BC
bc[:, 0] = fd.FinDiff(1, DY, 1), 0 # bottom
# mid = round(N[0]/2) 
# bc[mid, 1:-1] = fd.FinDiff(0, DX, 1), 0

pde = fd.PDE(E_c, f, bc)
T = pde.solve()

fig, ax = plt.subplots()
cs = plt.contourf(x,y, T.T)
fig.colorbar(cs,  orientation='vertical')
fig.show()


############# Donward Liquid Flow ############################################

# Liquid Flow Operator
E_L = h.E_L
f = a_dot.magnitude
E_source = h.E_source

## Boundary Conditions
bc = fd.BoundaryConditions(N)
# Dirichlet BC
bc[0,:] = h.params['T_atmos'][0] # left 
bc[-1,:] = h.params['T_atmos'][0] # right
bc[:,-1] = h.params['T_atmos'][0] # top
# Neumann BC
bc[:, 0] = fd.FinDiff(1, DY, 1), 0 # bottom
# mid = round(N[0]/2) 
# bc[mid, 1:-1] = fd.FinDiff(0, DX, 1), 0

pde = fd.PDE(E_L, f, bc)
T = pde.solve()

fig, ax = plt.subplots()
cs = plt.contourf(x,y, T.T)
fig.colorbar(cs,  orientation='vertical')
fig.show()

# for t in range(0, 10):
#     bc[:, 0] = T[:, 0]
#     pde = fd.PDE(E_L, f, bc)
#     T = pde.solve()
    

############# Gas Flow #######################################################

# Gas Flow Operator
E_g = -1 * h.E_g
f = a_dot.magnitude
E_source = h.E_source

## Boundary Conditions
bc = fd.BoundaryConditions(N)

# Dirichlet BC
bc[0,:] = h.params['T_atmos'][0] # left 
bc[-1,:] = h.params['T_atmos'][0] # right
bc[:,-1] = h.params['T_atmos'][0] # top
# Neumann BC
bc[:, 0] = fd.FinDiff(1, DY, 1), 0 # bottom
mid = round(N[0]/2) 
bc[mid, 1:-1] = fd.FinDiff(0, DX, 1), 0

pde = fd.PDE(E_g, E_source.magnitude, bc)
T = pde.solve()

fig, ax = plt.subplots()
cs = plt.contourf(x,y, T.T)
fig.colorbar(cs,  orientation='vertical')
fig.show()