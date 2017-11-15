# Optional; supresses warnings about GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Read the data
COLUMNS = ["url", "title_length", "article_length", "keywords", "shares"]
data = np.genfromtxt("OnlineNewsPopularitySample.csv", delimiter=',', names=COLUMNS)

# We're looking for shares based on article_length
article_length = tf.placeholder("float")
shares = tf.placeholder("float")

# Started at 1.0, 1.0; after 1,000,000 iterations, it was [1.908, -157.272]
# Another 100,000 and it was [1.928, -170.681]
# Another 100,000 and it was [1.949, -184.156]
# Another 100,000: Predicted model: 1.985x + -207.630

# Set up the variables we're going to use
initial_m = 1.0
initial_b = 1.0

# We're looking for y = mx + b, so we'll use "weights"
# for m and b
w = tf.Variable([initial_b, initial_m], name="w")

# Now we can define the actual prediction. Remember that
# m = w[1]
# x = article_length
# b = w[0]
#predicted_shares = tf.add(tf.multiply(w[1], article_length), w[0])
predicted_shares = w[1]*article_length + w[0]

# The "loss" is the difference between what the algorithm thinks it will
# be, and what it actually is.  (But we square it so that it's always positive.)
error = tf.multiply(tf.square(predicted_shares - shares), .00001)

# Now we set up the optimizer; we want to take very small steps, and create
# an optimizer that's trying to minimize the error term.
step_size = .0015
optimizer = tf.train.GradientDescentOptimizer(step_size).minimize(error)

# We'll define the model as this collection of variables
model = tf.global_variables_initializer()

# Finally let's create a session to actually run the model
with tf.Session() as session:
   # First initialize all the variables
   session.run(model)

   # Now we're going to run the optimizer
   for i in range(100000):
      # We are running the optimizer, feeding it the data to use for article_length and shares. 
      data_to_use = { article_length: data['article_length'],
                      shares: data['shares'] }
      session.run(optimizer, feed_dict=data_to_use)
      
      #Every 100 iterations, we'll display the current values of w[0] and w[1]
      if (i % 100 == 0):
         print (session.run(w))

   # Once it's done, we need to get the value of w so we can display it.	
   w_value = session.run(w)
   print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[1], b=w_value[0]))

# Now display the data and the link we're predicting
fig = plt.figure()

# This is the data
plt.scatter(data['article_length'], data['shares'], color='r', s=5.0)

# This is the line; we use article_length for x, then compute the shares based on the model
plt.plot(data['article_length'], data['article_length']*w_value[1] + w_value[0])

# Show the figure
plt.xlabel('Article (words)')
plt.ylabel('Shares')
plt.show()


