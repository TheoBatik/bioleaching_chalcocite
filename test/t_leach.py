# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 23:04:58 2021

@author: Theodore B
"""
import heap 
import numpy as np

C_L = np.full(N, 0.006) * kg/cube
CL_mol = C_L/c['M_Ox'][0]/mol # dividing by mol because I still need to figure out what the units of alpha should be....
alpha = np.zeros(N) / cube
T = np.full(N, 250) * kelvin
dt = 5/100*second

# Reaction rate
a_dot = alpha_dot(T, C_L) # outside of the leach method because it will be needed elsewhere
alpha_formed = a_dot*dt
alpha += alpha_formed
ox_to_alpha = 2.5/2 # Oxygen to Cu2SO4 based on the stoiciometric coefficients
# oxygen_lost = ox_to_alpha*alpha_formed
CL_mol -= ox_to_alpha*alpha_formed


for i in range(0,10):
    alpha += alpha_formed
    CL_mol -= ox_to_alpha*alpha_formed
    print('alpha =', alpha[0][0])
    print('Ox =', CL_mol[0][0])