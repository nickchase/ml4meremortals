import matplotlib.pyplot as plt
import numpy as np

points = [[1, 3, 7], [8, 14, 26]]
points2 = [[0, 3, 10], [5, 14, 35]]

# Set the size of the data points in the graph
scale = 5.0

# Create a new figure
fig = plt.figure(1)
# Plot the data
plt.scatter(points[0], points[1], color='r', s=30.0)
#plt.plot(points2[0], points2[1], color='g')
# Label the axes
plt.xlabel('Boxes of paper clips')
plt.ylabel('Price')
plt.ylim(0, 30)
plt.xlim(0, 10)
plt.text(1, 6.75, "(1, 8)")
plt.text(3, 12.75, "(3, 14)")
plt.text(7, 24.75, "(7, 26)")

# Displa all the figures
plt.show()


