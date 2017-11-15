import numpy as np
import matplotlib.pyplot as plt
import random, time
from matplotlib.animation import FuncAnimation


colors = ['b','g','r', 'c', 'm', 'y']

t = np.arange(-10, 10, 0.2)


# Plot the data
plt.plot(t, t**2, color='g')

# Label the axes
plt.ylabel('Cost')
plt.xlabel('Slope')

plt.show()


