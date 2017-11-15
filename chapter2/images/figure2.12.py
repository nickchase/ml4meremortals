import matplotlib.pyplot as plt
import numpy as np

# Read the data
COLUMNS = ["url", "title_length", "article_length", "keywords", "shares"]
data = np.genfromtxt("../OnlineNewsPopularityPicSample.csv", delimiter=',', names=COLUMNS)

# Set the size of the data points in the graph
scale = 5.0

# Create a new figure
fig = plt.figure(1)
# Plot the data
plt.scatter(data['article_length'], data['shares'], color='r', s=5.0)
#plt.plot(data['article_length'], data['article_length']*5, color='g')
#plt.plot(data['article_length'], data['article_length']*5.25, color='b')
#plt.plot(data['article_length'], data['article_length']*4.75, color='y')
# Label the axes
plt.xlabel('Article Length')
plt.ylabel('Shares')

# Displa all the figures
plt.show()


