import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Set up our parameters
points_n = 500
clusters_n = 2
iteration_n = 100
field_size = 10
not_finished = True
this_iteration = 0

# Set up the image and create the variables we'll use for animating the data
fig = plt.figure()
ax = plt.axes(xlim=(-1*field_size, field_size), ylim=(-1*field_size, field_size))
line, = ax.plot([], [], 'kx', lw=2)
centroidsimg, = ax.plot([], [], 'kx', markersize=15)
pointsimg = {}
pointsimg[0], = ax.plot([], [], 'r.')
pointsimg[1], = ax.plot([], [], 'b.')


# Create the points we're going to use

#points_build = np.random.poisson(field_size/2, 1000)
#points_build = points_build.reshape([500, 2])
#print(points_build)

# Start with an array of coordinates randomly positioned around the field
# We'll use the laplace distribution to kind of "bunch them up"
points_build = np.random.laplace(field_size/4, field_size/4, [points_n,2])

# Now create the actual points objects in Tensorflow using those coordinates
points = tf.constant(points_build, dtype = tf.float64)

# Start the centroids in random places, scaled to the overall size
centroids = np.random.random_sample([clusters_n, 2])
centroids = field_size * centroids

# Assign the points to the nearest centroid
points_expanded = tf.expand_dims(points, 0)
centroids_expanded = tf.expand_dims(centroids, 1)
distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
assignments = tf.argmin(distances, 0)

# Initialize everything
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

#Create the points constant so Tensorflow can act on it
points_values = session.run(points)
  
def updatefigure(this_iteration):

  # We'll need to get access to the global variables so they hold onto their
  # values between updates
  global points, centroids, centroids_expanded, distances, assignments, points_values, not_finished, iteration_n

  # If we're already finished, stop
  if (not_finished and this_iteration < iteration_n):
  
    # Assign the points to the nearest centroid
    distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
    assignments = tf.argmin(distances, 0)
    assignment_values = session.run(assignments)

    # Create groups based on the assignments, then find the centers of
    # those new groups
    partitions = tf.dynamic_partition(points, tf.to_int32(assignment_values), clusters_n)
    newcentroids = np.zeros((clusters_n, 2))
    for i in range(clusters_n):
      newcentroids[i] = session.run(tf.reduce_mean(partitions[i], 0))

    # If the centroids haven't moved, we're done
    if (np.array_equal(centroids, newcentroids)):
      not_finished = False
    else:
      # If the centroids have moved, move everything to the
      # new values and update the image
      centroids = newcentroids
      centroids_expanded = tf.expand_dims(centroids, 1)

      # Clear the existing points in the image so we only get the new ones
      pointsimg[0].set_data([], [])
      pointsimg[1].set_data([], [])

      # Assign each point to a color group
      for i in range(points_n-1):
          # Get the existing data and append the current point to it, then
          # set the data to be the new set
          this_set, = pointsimg[assignment_values[i]],
          xdata = this_set.get_data()[0]
          xdata = np.append(xdata, points_values[i][0])
          ydata = this_set.get_data()[1]
          ydata = np.append(ydata, points_values[i][1])
          this_set.set_data(xdata, ydata)

      # Set the data for the centroids image
      centroidsimg.set_data(centroids[:, 0], centroids[:, 1])

  return centroidsimg, pointsimg[0], pointsimg[1], 

# Set the initial data to be nothing; the updatefigure function sets the real data
def init():
    COLUMNS = ["url", "title_length", "article_length", "keywords", "shares"]
    data = np.genfromtxt("../OnlineNewsPopularityPicSample.csv", delimiter=',', names=COLUMNS)

    # Plot the data
    plt.scatter(data['article_length'], data['shares'], color='r', s=5.0)
    plt.plot(data['article_length'], data['article_length']*5, color='g')
    plt.plot(data['article_length'], data['article_length']*5.25, color='b')
    plt.plot(data['article_length'], data['article_length']*4.75, color='y')
    # Label the axes
    plt.xlabel('Article Length')
    plt.ylabel('Shares')


    centroidsimg.set_data([], [])
    # pointsimg[0].set_data([], [])
    # pointsimg[1].set_data([], [])
    return centroidsimg, #pointsimg[0], pointsimg[1],

# Do the calculations and update the image.
anim = animation.FuncAnimation(fig, updatefigure, init_func=init,
                               frames=200, interval=500, blit=True)

plt.show()

# Print the final results
print(centroids)
