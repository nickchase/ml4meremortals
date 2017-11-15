import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

COLUMNS = ["url", "title_length", "article_length", "keywords", "shares"]
data = np.genfromtxt("OnlineNewsPopularitySample.csv", delimiter=',', names=COLUMNS)

initial_b = 1.0
initial_m = 1.0
w = tf.Variable([initial_b, initial_m], name="w")

print (w[0])
print (w[1])

article_length = tf.placeholder("float")
shares = tf.placeholder("float")

#predicted_shares = w[1]*article_length + w[0]
predicted_shares = tf.add(tf.multiply(w[1], article_length), w[0])

error = tf.multiply(tf.square(predicted_shares - shares), .00001)

step_size = .0015
optimizer = tf.train.GradientDescentOptimizer(step_size).minimize(error)

model = tf.global_variables_initializer()

with tf.Session() as session:
   session.run(model)
   data_to_use = { article_length: data['article_length'],
                   shares: data['shares'] }

   for i in range(100000):
      session.run(optimizer, feed_dict=data_to_use)
      if (i % 100 == 0):
         print (session.run(w))


   w_value = session.run(w)
   print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[1], b=w_value[0]))

fig = plt.figure()
plt.scatter(data['article_length'], data['shares'], color='r', s=5.0)

plt.plot(data['article_length'], data['article_length']*w_value[1] + w_value[0])

plt.xlabel('Article (words)')
plt.ylabel('Shares')
plt.show()

