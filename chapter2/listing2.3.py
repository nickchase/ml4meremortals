# Optional; supresses warnings about GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

# Read the data
COLUMNS = ["url", "title_length", "article_length", "keywords", "shares"]
data = np.genfromtxt("OnlineNewsPopularityNonLinear.csv", delimiter=',', names=COLUMNS)

# Set up the article-related data
article_data = np.zeros([3, 100])
article_data[0] = data["title_length"]
article_data[1] = data["article_length"]
article_data[2] = data["keywords"]

# Set up targets for training
share_data = np.zeros([1, 100])
share_data[0] = data["shares"]

# Set up weights and bias
initial_m = np.zeros([1, 3])
initial_b = np.zeros([1])

w = tf.Variable(initial_m, dtype="float32", name="w")
b = tf.Variable(initial_b, dtype="float32", name="b")

# Placeholders for data and targets
x = tf.placeholder("float32", shape=[3, 100])
actual_target = tf.placeholder("float32", shape=[1, 100])

# Predicted target values
predicted_shares = tf.add(tf.matmul(w, x), b)

# Divide error by 1000 to avoid exploding gradient
error = tf.multiply(tf.reduce_mean(tf.squared_difference(predicted_shares, actual_target)), .001)

# Create optimizer
step_size = .001
optimizer = tf.train.GradientDescentOptimizer(step_size).minimize(error)

# Define the model as all of the variables
model = tf.global_variables_initializer()

# Create the session to run the algorithm
with tf.Session() as session:
    # First initialize all the variables
    session.run(model)

    # Run the algorithm
    for i in range(1000000):
        
        # Run the optimizer and return the loss and weight
        _, loss, w_value = session.run([optimizer, error, w], feed_dict={x: article_data, actual_target: share_data})

        # Display every 1000 values
        if (i % 1000 == 0):
           print (loss, w_value)

