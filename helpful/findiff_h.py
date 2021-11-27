# -*- coding: utf-8 -*-
"""
Check out findiff python package:
https://github.com/maroba/findiff
"""


import numpy as np
import findiff as fd
import matplotlib.pyplot as plt

# Example 1: Population Growth
    
shape = (300,)

t_end = 10
t = np.linspace(0, t_end, shape[0])
dt = t[1]-t[0]

L = fd.FinDiff( 0, dt, 1, acc = 6) - 1.05*fd.Identity() # Population Growth, dN/dt = 5 * N

bc1 = fd.BoundaryConditions(shape)
bc1[0] = 5

f = np.zeros(shape)

pde = fd.PDE(L, f, bc1)
N = pde.solve()

plt.plot(t, N)


# Example 2: 1D forced harmonic oscillator with friction and Dirichlet BC's
    
shape = (300, )
t_end = 10
t = np.linspace(0, t_end, shape[0])
dt = t[1]-t[0]

L = fd.FinDiff(0, dt, 2) - fd.FinDiff(0, dt, 1) + 5*fd.Identity()
f = np.cos(2*t)

bc = fd.BoundaryConditions(shape)
bc[0] = 0
bc[-1] = 1

print('f', f[1:5])
pde = fd.PDE(L, f, bc)
u = pde.solve()
print('f', f[1:5])


plt.figure()
plt.plot(t, u)

# Example 3: 2D Heat Conduction with Dirichlet BC's

shape = (100, 100)
x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
dx, dy = x[1]-x[0], y[1]-y[0]
X, Y = np.meshgrid(x, y, indexing='ij')

L = fd.FinDiff(0, dx, 2) + fd.FinDiff(1, dy, 2)
f = np.zeros(shape)

T_amb = 500
bc = fd.BoundaryConditions(shape)
#bc[:,50] = fd.FinDiff(0, dx, 1), 0 # Neumann BC
bc[-1, :] = T_amb  # top is at ambiant temp
bc[0, :] = 0  # base is at absolute zero (for all x)
#bc[:, 0:5] = T_amb - 200*Y
bc[:, -1] = -200*Y

pde = fd.PDE(L, f, bc)
u = pde.solve()

plt.figure()
plt.imshow(u, interpolation='spline16')
plt.show()

# Example 4: 2D Heat Conduction with Neumann BC's

shape = (100, 100)
x, y = np.linspace(0, 1, shape[0]), np.linspace(0, 1, shape[1])
dx, dy = x[1]-x[0], y[1]-y[0]
X, Y = np.meshgrid(x, y, indexing='ij')

L = fd.FinDiff(0, dx, 2) + fd.FinDiff(1, dy, 2)
f = np.zeros(shape)

bc = fd.BoundaryConditions(shape)
bc[1,:] = fd.FinDiff(0, dx, 1), 0 # Neumann BC
bc[-1,:] = 300. - 200*Y  # Dirichlet BC
bc[:, 0] = 300.   # Dirichlet BC
bc[1:-1, -2] = fd.FinDiff(1, dy, 1), 0  # Neumann BC

print(f[0][:])
pde = fd.PDE(L, f, bc)
u = pde.solve()
print(f[0][:]) 

plt.figure()
plt.imshow(u, interpolation='spline16')
plt.show()


