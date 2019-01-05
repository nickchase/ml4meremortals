from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import sklearn
import os
from bs4 import BeautifulSoup  

# First we'll read in the file we created previously
emails = pd.read_csv('./data/emails.csv', header=0, delimiter=",")

# Split off 1/4 of the data to test the classifier
train, test = sklearn.model_selection.train_test_split(emails, test_size=.25)

print("There are {a} training emails and {b} test emails.".format(a=len(train), b=len(test)))

# First we create a CountVectorizer, scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",  
                             tokenizer = None,  
                             preprocessor = None,
                             stop_words = "english", 
                             max_features = 5000)

# Now we'll use the CountVectorizer to take list of strings that makes up our data
# and turn it into an actual model. To do that, fit_transform() learns all the words
# it finds (the vocabularly) and turns it into a series of vectors that show how many
# times each word is used in each document. These are called "feature vectors".
train_data_features = vectorizer.fit_transform(train["email"])

# Create a Random Forest classifier
forest = RandomForestClassifier(n_estimators = 100)

# Next we can train the classifier; our target label is the "priority"
forest = forest.fit( train_data_features, train["priority"] )

# Now let's see how good the classifier is by predicting our test data's priorities 
test_data_features = vectorizer.transform(test["email"])
result = forest.predict(test_data_features)
score = forest.score(test_data_features, test["priority"])

# Output the result
print("The classifier predicted the correct priority {a:.2f} percent of the time.".format(a=score*100))

# Now we'll classify our unknown emails
print("Classifying incoming emails...")
new_emails = pd.read_csv('./data/new_emails.csv', header=0, delimiter=",")
new_data_features = vectorizer.transform(new_emails["email"])
new_result = forest.predict(new_data_features)

# Make a DataFrame with the data so we can output it to a final file
final_results = pd.DataFrame( data={"id":new_emails["id"], "email":new_emails["email"], "priority":new_result} )

# Write the final output file.
final_results.to_csv("./data/results.csv", index=False)
print ("Done.  You can find the reuslts in ./data/results.csv")


