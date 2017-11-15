import matplotlib.pyplot as plt
import numpy as np

# Read in the data

# Set the names for the columns
COLUMNS = ["url", "title_length", "article_length", "keywords", "shares"]

#Actually read the data
data = np.genfromtxt("OnlineNewsPopularitySample.csv", delimiter=',', names=COLUMNS)

# Print the number of rows
print ("There are " + str(len(data)) + " items in this array.")

# Print the first row
print (data[0])

# Print out one column
print (data['article_length'])

# Set the size of the data points in the graph
scale = 5.0

# Create a new figure
fig = plt.figure(1)
# Plot the data
plt.scatter(data['title_length'], data['shares'], color='r', s=5.0)
# Label the axes
plt.xlabel('Title (words)')
plt.ylabel('Shares')

# Create a second figure
fig = plt.figure(2)
plt.scatter(data['article_length'], data['shares'], color='r', s=5.0)
plt.xlabel('Article (words)')
plt.ylabel('Shares')

# And a third
fig = plt.figure(3)
plt.scatter(data['keywords'], data['shares'], color='r', s=5.0)
plt.xlabel('Keywords')
plt.ylabel('Shares')

# Displa all the figures
plt.show()


