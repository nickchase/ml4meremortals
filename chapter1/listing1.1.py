# Tensorflow runs better on a GPU; since we haven't
# installed it that way, we'll turn off messages telling
# us about it

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
 
# Tell Python3 that we want to use Tensorflow

import tensorflow as tf
 
# Tensorflow has its own variables, constants, and so on.
# We'll create one here just to make sure everything is working.

message = tf.constant("Hooray, Tensorflow is working!!!")
print ( message )
 
# In order to make Tensorflow do anything, we need to create a 
# session in which the work will be done.

session = tf.Session()
 
# Right now message is just an empty object; let's "run" it to 
# make it meaningful, then assign its value to a variable, output.

output = session.run(message)
 
# Now we can pretty up the message (by default, it's bytecode) 
# and print it out:

print ( output.decode() )

