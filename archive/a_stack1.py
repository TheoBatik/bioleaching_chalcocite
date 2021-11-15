# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 16:06:55 2021

@author: Theodore B
"""
import numpy as np
import matplotlib.pyplot as plt


###### Stack method 1.0 #######

def stack( d , N, symmetric = True ): # spacetime / spacetime_boundary
    ''' Returns a numpy.meshgrid with number of lattice nodes set by N 
    and spacetime bounds set by d = (x, y t).'''
    if symmetric:
        x = np.linspace(-d[0]/2, d[0]/2, N[0])
        y = np.linspace(-d[1]/2, d[1]/2, N[1])
        t = np.linspace(0, d[2], N[2])
        print( 'Heap succesfully stacked!')
        print( '\t Shape: \n \t\t (3, {}, {}, {})'.format(N[0], N[1], N[2]) )
        print( '\t Dimensions (relative to origin):' )
        print( '\t'*2,  'd = (x, y, t) = {}'.format(d))
        print( '\t => spacetime bounds (symmetric):')
        print( '\t'*2,  'x_a -> x_b = {} -> {}'.format( -d[0]/2, d[0]/2))
        print( '\t'*2,  'y_a -> y_b = {} -> {}'.format( -d[1]/2, d[1]/2))
        print( '\t'*2,  't_a -> t_b = {} -> {}'.format( -d[2]/2, d[2]/2)) 
        print( '\t Number of nodes (per dimension):'  )
        print( '\t'*2,  'N = (Nx, Ny , Nt) = {}'.format(N)) 
        return np.array(np.meshgrid( x, y, t , indexing = 'ij'))
    else:
        print('complete non-symmetric version later')
    
def get_grain(stack):
       DX = abs(stack[0][1][0][0] - stack[0][0][0][0])
       DY = abs(stack[1][0][1][0] - stack[1][0][0][0])
       DT = abs(stack[2][0][0][1] - stack[2][0][0][0])
       return (DX, DY, DT)


stack = stack( (10, 20, 100), (11, 21, 51) ) # xx, yy, tt



print(get_grain(stack))

def plt_stack_3d(stack):
    XX = stack[0][:][:][:]
    YY = stack[1][:][:][:]
    tt = stack[2][:][:][:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(XX, YY, tt, marker = '.', s = 1)
    plt.show() 
    
def plt_stack_xy(stack):
    axis_a = stack[0][:][:][0]
    axis_b = stack[1][:][:][0]
    print(axis_a.shape)
    print(axis_b.shape)
    plt.figure()
    plt.plot(axis_a, axis_b, marker='.', color='k', linestyle='none')


axis_a = stack[0][0][0][:]
axis_b = stack[1][:][:][0]  
axis_c = stack[2][:][:][0]  






# print(stack.shape)

plt_stack_3d(stack)
plt_stack_xy(stack)