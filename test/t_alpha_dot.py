# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 15:09:15 2021

@author: Theodore B
"""


import heap as h
import numpy as np
import findiff as fd
import matplotlib.pyplot as plt


### Examining the behviour of Vm and alpha_dot ###


# Plot of Vm(T)

T = np.linspace(273, 323, 300)*h.kelvin
Vm = h.Vm(T)
plt.figure()
plt.plot(T, Vm)
plt.title('Respiration Rate: Vm(T)')
plt.xlabel('{}'.format(T.units) )
plt.ylabel('{}'.format( Vm[0].units) )

# Comment: qualitatively the same as depicted in paper by Casas et al.

# Plot of alpha_dot(T, C_L) 

T = 300*h.kelvin # constant T
C_L = np.linspace(0, 0.006, 30)* h.kg/h.cube # linearly increasing C_L
a_dot = h.alpha_dot(T, C_L)
plt.figure()
plt.plot( C_L, a_dot ) 
plt.title('alpha_dot(C_L) with T = const.')
plt.xlabel('{}'.format(C_L.units) + ' of liquid Oxygen' )
plt.ylabel('{}'.format( a_dot.units) )

# Comments: 
# standard behaviour of Michaelis-Menten equation with oxygen as limiting substrate;
# same as population growth with carrying capacity set by K_m


#### Solutions to the reaction rate differential equation, alpha_dot = f(T, C_L) ####

shape = (100,)
t_end = 10
t = np.linspace(0, t_end, shape[0])
dt = t[1]-t[0] 


L = fd.FinDiff( 0, dt, 1, acc = 2) 

## Test 1: Constant T and C_L ##

T = 300*h.kelvin
C_L = 0.006* h.kg/h.cube
f = np.full( len(t),  h.alpha_dot(T, C_L).magnitude ) # need .magnitude?


# BC's
bc = fd.BoundaryConditions(shape)
bc[0] = 0 

# Solve
pde = fd.PDE(L, f, bc)
alpha = pde.solve()

# Plot
plt.figure()
plt.plot(t, alpha)
plt.title('Concentration of product as a function of time: alpha(T)')
plt.xlabel('{}'.format(h.second) )
alpha_u = h.alpha_dot(T, C_L) * h.second
plt.ylabel('{}'.format( alpha_u.units) )

# Comment: Positive linear increase, as expected.


## Test 2: Constant C_L with linearly increasing / decreasing T ##

T1 = np.linspace(350, 50, shape[0]) * h.kelvin
T2 = np.linspace(50, 350, shape[0]) * h.kelvin

f1 = np.full( len(t),  h.alpha_dot(T1, C_L).magnitude)
f2 = np.full( len(t),  h.alpha_dot(T2, C_L).magnitude)

# Solve
pde1 = fd.PDE(L, f1, bc)
pde2 = fd.PDE(L, f2, bc)
alpha1 = pde1.solve()
alpha2 = pde2.solve()

# Plot
plt.figure()
plt.plot(t, alpha1, t, alpha2)
plt.title('Concentration of product over time for linearly increasing / decreasing T.')
plt.xlabel('{}'.format(h.second) )
alpha_u = h.alpha_dot(T, C_L) * h.second
plt.ylabel('{}'.format( alpha_u.units) )

# Comments: 
# alpha_dot increases with increasing T before reaching platau
# alpha_dot decreasing with decreasing T before reaching platau

