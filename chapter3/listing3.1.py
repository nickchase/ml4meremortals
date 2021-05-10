# Optional; supresses warnings about GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np

%tensorflow_version 1.14
import tensorflow as tf 

"""
In this example, we're going to create a classifier that uses all of the feature columns
we give it. When we train the classifier, we will use an input function, input_fn. To
make things simple, each function will tell the get_inputs() function where to find the
data.  The get_inputs() function will then load the data set and return the data and the
target values.

Once we've trained the classifier, we'll test it, then we'll give it "unknown" values and
see what it predicts the values to be.
"""

# Return the data for the input function
def get_inputs(data_file):

    data_set = tf.contrib.learn.datasets.base.load_csv_with_header(
         filename=data_file,
         target_dtype=np.int64,
         features_dtype=np.float32)

    feature_sets = tf.constant(data_set.data)
    targets = tf.constant(data_set.target)

    return feature_sets, targets

def get_training_inputs():
    return get_inputs("OnlineNewsPopularityClassification.csv")

def get_test_inputs():
    return get_inputs("OnlineNewsPopularityClassification_test.csv")

def get_new_inputs():
    return get_inputs("OnlineNewsPopularityClassification_newsamples.csv")

# All features have real-value data, so we won't specify any in particular
feature_columns = [tf.contrib.layers.real_valued_column("")]
classifier = tf.contrib.learn.LinearClassifier(feature_columns=feature_columns)

# Now we'll train the classifier, or "fit" it to the data.  In this case we're 
# going to do 2000 iterations
classifier.fit(input_fn=get_training_inputs, steps=2000)

# Next we'll check to see how good a job it did by feeding it the test data
accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
print("Accuracy against the test data: {a:f}".format(a=accuracy_score))

# Finally, we'll ask it to predict whether new articles will be successful or not
predictions = list(classifier.predict_classes(input_fn=get_new_inputs))
print("Predictions: {a}".format(a=predictions))



