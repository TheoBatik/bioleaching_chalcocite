# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:46:53 2021

@author: Theodore B
"""
import heap as h
import numpy as np
import findiff as fd
import matplotlib.pyplot as plt


################### INITIAL CONDITIONS ########################################

N = (21, 41) # N = size of heap
d = (10, 20)

# Concentrations 

# # henry = lambda T: params['Henry_1'][0]  + params['Henry_2'][0] * T -  params['Henry_3'][0] * T **2 
# # # print(henry(params['T_atmos'][0]))
# # a = 0.006/henry(20)

# # Gasesous oxygen
# coxg_fac = params['Ox_in_air'][0] * params['rho_air'][0] * kg/cube
# coxg = np.full(N, coxg_fac)

# ## Copper-sulphide
# CuS_fac = params['rho_B'][0] * params['G^0'][0]
# CuS = np.full(N, CuS_fac ) * kg/cube

# # Copper-sulphate
# cu = np.zeros(N)

# # Thermodynamic #

# ## temperature
# T = np.full(N, params['T_atmos'][0] )
# ## gas density
# rho_non_ox = np.full( N, 0.24 * params['rho_B'][0].units )
# gd = Coxg + rho_non_ox


################### INITIATE HEAP ############################################


heap = h.Heap(d, N)
T = 1 * heap.kelvin
Vmax = heap.Vm(T)
 
# a_dot = heap.cu_dot(T, )

x, y, mesh = heap.stack()
dx, dy = heap.space_grain(x, y)

p = heap.params

heap.plot_mesh(mesh)


