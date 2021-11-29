# -*- coding: utf-8 -*-
"""
Psuedo-test unit for upgrades to heap's ability to evolve in time.'

"""

import heap
import numpy as np
# import findiff as fd
# import matplotlib.pyplot as plt

##### INIT HEAP #####

# Dimensions (Lx, Ly, Delt):
Lx = 10
Ly = 3
Delt = 5
D =(Lx, Ly, Delt)

# Mesh and differentials
scale_xy = 10
scale_t = 10
N = (Lx * scale_xy + 1, Ly*scale_xy + 1, Delt * scale_t + 1) 
hp = heap.Heap( D, N )
x, y, t, mesh = hp.stack()
dx, dy, dt = hp.diffs(x, y, t)

# Respiration rate, Vm
T1 = np.linspace(273, 323, 300) * heap.kelvin # temperature
Vm = hp.Vm(T1) # Respiration rate

# Optimal temperature => maximum respiration rate
i = list(Vm).index(max(Vm)) # index of max Vm
T_opt = T1[i] # optimal temperature

# Maximum concentration of gasesous oxygen
coxg_fac = hp.coxg_fac 
Nxy = (hp.ms[0], hp.ms[1]) ## add as attribute?
coxg = np.full(Nxy, coxg_fac) * hp.params['rho_air'][0].units

