# -*- coding: utf-8 -*-
"""
Estimation of upper bound on heat generation

@author: Theodore B
"""

import heap as h
import numpy as np
# import findiff as fd
import matplotlib.pyplot as plt


# Respiration rate

T = np.linspace(273, 323, 300)*h.kelvin
Vm = h.Vm(T)
plt.figure()
plt.plot(T, Vm)
plt.title('Respiration Rate: Vm(T)')
plt.xlabel('{}'.format(T.units) )
plt.ylabel('{}'.format( Vm[0].units) )

# Optimal temperature for maximum Respiration rate
i = list(Vm).index(max(Vm))
T_opt = T[i]

# Rate of Copper-Sulfide dissolution
CL_max = 0.006 * h.kg/h.cube
a_max = h.alpha_dot(T_opt, CL_max)



