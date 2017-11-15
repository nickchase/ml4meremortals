import numpy as np
import matplotlib.pyplot as plt
import random, time
from matplotlib.animation import FuncAnimation


colors = ['b','g','r', 'c', 'm', 'y']

plt.axis([-.1, 105, -1, 105])

numguesses = [0., 50., 75, 82., 85.]
lossleft = [82., 32., 7., 0., 3]


# Plot the data
plt.scatter(numguesses, lossleft, color='r', s=15.0)

plt.plot(numguesses, lossleft, color='g')
plt.plot([82,100], [0,18], color='g') 

# Label the axes
plt.ylabel('Guesses')
plt.xlabel('Cost (difference between the prediction and the real data)')

plt.show()


