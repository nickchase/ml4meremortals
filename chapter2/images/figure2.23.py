'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
#ax = fig.gca(projection='3d')

# Make data.
m = np.arange(-10, 10, 0.5)
b = np.arange(-5, 5, 0.5)
x = np.arange(-10, 10, 0.5)
y = np.arange(-10, 10, 0.5)

#m, b = np.meshgrid(m, b)
#ypred = m*x# + b
#cost = (y - ypred)**2
cost = m*m


# Plot the surface.
#surf = ax.plot_surface(m, b, cost, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=True)

plt.plot(m, cost)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
