# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:46:53 2021

@author: Theodore B
"""
import heap as h
import numpy as np
import findiff as fd
import matplotlib.pyplot as plt



################### INITIATE HEAP ############################################

heap = h.Heap(d, N)
T = 1 * h.kelvin
Vmax = heap.Vm(T)
 
# a_dot = heap.cu_dot(T, )
print('mesh', heap.mesh, '\n')
x, y, mesh = heap.stack()


dx, dy = heap.space_grain(x, y)

p = heap.params

print(heap.dx, heap.x)  

h.plot_mesh(mesh)


