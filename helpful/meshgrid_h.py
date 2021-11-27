# -*- coding: utf-8 -*-
"""
The Power of Meshgrids! 

Check out:
    https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    https://likegeeks.com/numpy-meshgrid/
    https://imada.sdu.dk/~marco/DM545/Resources/Ipython/Broadcast.html
"""

import numpy as np
import matplotlib.pyplot as plt


# np.meshgrid generates a (not necessarily Cartesian) grid based on two 1-D arrays

xvalues = np.array([0, 1, 2, 3, 4])
yvalues = np.array([0, 1, 2, 3, 4])*5
xx, yy = np.meshgrid(xvalues, yvalues, indexing = 'ij')
# plt.figure()
plt.plot(xx, yy, marker='.', color='k', linestyle='none')
print('\n xx =', xx, '\n')
print('yy =', yy)



# Given the redundancy, the grids can be made "sparse" (thanks to array broadcasting)

xs, ys = np.meshgrid(xvalues, yvalues,  sparse=True)
print('xs =', xs)
print('ys =', ys)
plt.figure()
plt.plot(xs, ys.T, marker='.', color='k', linestyle='none')

# Example using the contour plot
x = np.arange(-5, 5, 1)
y = np.arange(-5, 5, 1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
plt.figure()
plt.contourf(x,y,z) # contour can handle 1-D co-ordinate arrays

# Example: contour plot with legend
fig, ax = plt.subplots()
x, y = np.meshgrid(np.arange(10),np.arange(10))
z = np.sqrt(x**2 + y**2)
cs = plt.contourf(x,y,z)
fig.colorbar(cs,  orientation='vertical')
fig.show()

# A polar co-ordinate meshgrid

R = np.linspace(1,5,10)
THETA = np.linspace(0, np.pi, 45)
radii, thetas = np.meshgrid(R,THETA)
print("R:{}".format(R.shape))
print("THETA:{}".format(THETA.shape))
print("meshgrid radii:{}".format(radii.shape))
print("mehgrid thetas:{}".format(thetas.shape))

ax = plt.subplot(111, polar=True)
ax.plot(thetas, radii, marker='.', ls='none')
plt.show()

# Changing the indexing:

# Cartesian 'xy' => (column, row) = default
# Matrix 'ij' => (row, column)

# 'ij' is just the transpose of 'xy'
i = np.array([1,2,3,4,5]) # rows
j = np.array([11, 12, 13, 14, 15]) # columns
ii, jj = np.meshgrid(i,j, indexing='ij')
print("row indices:\n{}\n".format(ii))
print("column indices:\n{}".format(jj))
print("ii equal to xx transposed ? ==>",np.all(ii == xx.T)) 
print("jj equal to yy transposed ? ==>",np.all(jj == yy.T))


# # 3-D meshgrid

X = np.linspace(1,4,4)
Y = np.linspace(6,8, 3)
Z = np.linspace(12,15,4)
xx, yy, zz = np.meshgrid(X,Y,Z, indexing = 'ij')
print(xx.shape, yy.shape, zz.shape)

# 3-D meshgrid stored in single array

X = np.linspace(1,4,4)
Y = np.linspace(6,8, 3)
Z = np.linspace(12,15,4)
XYZ = np.array(np.meshgrid(X,Y,Z, indexing = 'ij'))
print(XYZ.shape)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz)
ax.set_zlim(12, 15)
plt.show()


# # Plotting 3-D surfaces using meshgrid

X = np.linspace(-20,20,100)
Y = np.linspace(-20,20,100)
X, Y = np.meshgrid(X,Y)
Z = 4*X**2 + Y**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap="plasma", linewidth=0, antialiased=False, alpha=0.4)
plt.show()


# Retrieve the 1-D arrays from the meshgrid

x_ = np.linspace(0., 1., 10)
y_ = np.linspace(1., 2., 20)
z_ = np.linspace(3., 4., 30)

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

assert np.all(x[:,0,0] == x_)
assert np.all(y[0,:,0] == y_)
assert np.all(z[0,0,:] == z_)
