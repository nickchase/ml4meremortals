import matplotlib.pyplot as plt
import numpy as np

COLUMNS = ["url", "title_length", "article_length", "keywords", "shares"]

data = np.genfromtxt("OnlineNewsPopularitySample.csv", delimiter=',', names=COLUMNS)

fig = plt.figure(1)

plt.xlabel('Title (words)')
plt.ylabel('Shares')

plt.scatter(data['title_length'], data['shares'], color='r', s=5.0)

fig = plt.figure(2)
plt.xlabel('Article (words)')
plt.ylabel('Shares')
plt.scatter(data['article_length'], data['shares'], color='r', s=5)

plt.plot(data['article_length'], 1*data['article_length'], color='g')

fig = plt.figure(3)
plt.xlabel('Keywords')
plt.ylabel('Shares')
plt.scatter(data['keywords'], data['shares'], color='r', s=5)

plt.show()
