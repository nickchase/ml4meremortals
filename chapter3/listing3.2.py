from bs4 import BeautifulSoup  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import os
import re
import sklearn
import numpy as np
import pandas as pd

SPAMDIR = "./data/spam/"
HAMDIR = "./data/ham_emails/"
STEAKDIR = "./data/priority_emails/"

TESTDIR = "./test_emails/"

def unicode_escape(unistr):
    """
    Tidys up unicode entities into HTML friendly entities

    Takes a unicode string as an argument

    Returns a unicode string
    """
    
    escaped = ""

    for char in unistr:
        print (char)
        if ord(char) < 127:
            escaped = escaped + char

    return escaped

def email_to_words( raw_email ):
    
    # First, pull out all the actual text and get rid of any HTML tags
    email_text = BeautifulSoup(raw_email, 'lxml').get_text() 
    
    # Now get rid of any punctuation, etc, and replace it with spaces
    letters_only = re.sub("[^a-zA-Z]", " ", email_text)
    
    # Change everything to lowercase
    lowercase = letters_only.lower()
    
    # Split everything into words so that we can make it one long string
    all_the_words = lowercase.split(" ") 
                                
    # Make the email into one long string
    all_the_words_together = " ".join(all_the_words)

    # Get rid of any funky characters
    all_the_words_together_clean = all_the_words_together.encode('ascii')

    return all_the_words_together_clean

# Create an array of {id, email, priority}
# Priority: -1 = spam, 0 = whenever, 1 = priority

id = 1
all_emails = []

# We're going to load each group of emails with its own priority level, incrementing
# the ID as we go.

print("load spam")

# Loop through the files in the directory
for file in os.listdir(SPAMDIR):

    # For each file, get the actual name of the file
    filename = os.fsdecode(file)

    # Open the file so we can read it
    thisfile = open(SPAMDIR+filename, 'r', encoding='ISO-8859-1')

    # Get all the text from the file
    raw_email = thisfile.read()
 
    # Create an empty object so we can add attributes to it
    this_email = {}

    # Set the ID, then increment it by one
    this_email["id"] = id
    id = id + 1

    # Set the actual email attribute to the words of the email
    this_email["email"] = email_to_words(raw_email)

    # This email is spam, so we'll set the priority to -1
    this_email["priority"] = -1

    # Add this email object to the array of emails
    all_emails.append(this_email)

    # Close the file
    thisfile.close()


# Now do the same for the Ham and the Steak, setting the directories and 
# priorities appropriately.
print("load ham")

for file in os.listdir(HAMDIR):
    filename = os.fsdecode(file)
    thisfile = open(HAMDIR+filename, 'r', encoding='ISO-8859-1')
    raw_email = thisfile.read()
    this_email = {}
    this_email["id"] = id
    id = id + 1
    this_email["email"] = email_to_words(raw_email)
    this_email["priority"] = 0
    all_emails.append(this_email)
    thisfile.close()

print("load steak")

for file in os.listdir(STEAKDIR):
    filename = os.fsdecode(file)
    thisfile = open(STEAKDIR+filename, 'r', encoding='ISO-8859-1')
    raw_email = thisfile.read()
    this_email = {}
    this_email["id"] = id
    id = id + 1
    this_email["email"] = email_to_words(raw_email)
    this_email["priority"] = 1
    all_emails.append(this_email)
    thisfile.close()

# Just to make sure we know what we have, output the number of emails, and a sample.
# print(len(all_emails))
# print(all_emails[10])

# Now we want to write the data to a file, so we'll first turn it into a DataFrame...
emails_to_save = pd.DataFrame( data=all_emails )
# ... and then write the DataFrame to disk.
emails_to_save.to_csv("./data/emails.csv", index=False, quoting=3, encoding="ascii")

# Next we'll prep the emails that we'll ultimately want to classify
print("load unclassified emails")
id = 1
test_emails = []
for file in os.listdir(TESTDIR):
    filename = os.fsdecode(file)
    thisfile = open(TESTDIR+filename, 'r', encoding='ISO-8859-1')
    raw_email = thisfile.read()
    this_email = {}
    this_email["id"] = id
    id = id + 1
    this_email["email"] = email_to_words(raw_email)
    test_emails.append(this_email)
    thisfile.close()

emails_to_save = pd.DataFrame( data=test_emails )
emails_to_save.to_csv("./data/new_emails.csv", index=False, quoting=3, encoding="ascii")

print ("Done.")


