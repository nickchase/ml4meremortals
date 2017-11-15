import numpy as np
import matplotlib.pyplot as plt
import random, time
from matplotlib.animation import FuncAnimation
# plt.style.use('ggplot')

COLUMNS = ["url", "title_length", "article_length", "keywords", "shares"]
data = np.genfromtxt("../OnlineNewsPopularityPicSample.csv", delimiter=',', names=COLUMNS)
#losslines = np.zeros([len(data), 2, 2])

losslines = [[[175, 175], [945, (175*5)]], [[201, 201], [1060, (201*5)]], [[194, 194], [960, (194*5)]], [[219, 219], [1081, (219*5)]], [[237, 237], [1197, (237*5)]]]

print(losslines)


#for x in range(len(data)):
   # Set reality
#   losslines[x][0][0] = data[x]['article_length']
#   losslines[x][0][1] = data[x]['shares']

   # set Prediction
#   losslines[x][1][0] = data[x]['article_length']
#   losslines[x][1][1] = data[x]['article_length']*5

#print(losslines[0])
#print(losslines[0][1])

colors = ['b','g','r', 'c', 'm', 'y']

fig, ax = plt.subplots(figsize=(10, 5))
#ax.set(xlim=(-3, 3), ylim=(-1, 1))

#x = np.linspace(-0, 1200, 500)
#t = np.linspace(0, 6000, 3000)
#X2, T2 = np.meshgrid(x, t)
 
#line = ax.plot(x, F[0, :], color='k', lw=2)[0]

# Plot the data
plt.scatter(data['article_length'], data['shares'], color='r', s=5.0)
plt.plot(data['article_length'], data['article_length']*5)

for x in range(len(data)):
    plt.plot(losslines[x][0], losslines[x][1], color='g')

# Label the axes
plt.xlabel('Article Length')
plt.ylabel('Shares')


 
plt.draw()
plt.show()


