# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 23:04:58 2021

@author: Theodore B
"""
import heap as h
import numpy as np


N = (6, 8)
d = (5, 7)
stk =  h.stack( d , N )

C_L = np.full(N, 0.006) * h.kg/h.cube
CL_mol = C_L/h.c['M_Ox'][0]/h.mol # dividing by mol because I still need to figure out what the units of alpha should be....
alpha = np.zeros(N) / h.cube
T = np.full(N, 250) * h.kelvin
dt = 5/100*h.second

# Reaction rate
a_dot = h.alpha_dot(T, C_L) # outside of the leach method because it will be needed elsewhere
alpha_formed = a_dot*dt
alpha += alpha_formed

# Oxygen balance --> this step actually comes in later, including convection and diffusions
ox_to_alpha = 2.5/2 # Oxygen to Cu2SO4 based on the stoiciometric coefficients 
# oxygen_lost = ox_to_alpha*alpha_formed
CL_mol -= ox_to_alpha*alpha_formed


for i in range(0,10):
    alpha += alpha_formed
    CL_mol -= ox_to_alpha*alpha_formed
    print('alpha =', alpha[0][0])
    print('Ox =', CL_mol[0][0])
    
    
    