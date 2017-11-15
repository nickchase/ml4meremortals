import numpy as np
import matplotlib.pyplot as plt
import random, time
from matplotlib.animation import FuncAnimation
# plt.style.use('ggplot')

COLUMNS = ["url", "title_length", "article_length", "keywords", "shares"]
data = np.genfromtxt("../OnlineNewsPopularityPicSample.csv", delimiter=',', names=COLUMNS)
plotdata = np.genfromtxt("../OnlineNewsPopularityLineSample.csv", delimiter=',', names=COLUMNS)

colors = ['b','g','r', 'c', 'm', 'y']

fig, ax = plt.subplots(figsize=(10, 5))
#ax.set(xlim=(-3, 3), ylim=(-1, 1))

x = np.linspace(-0, 1200, 500)
t = np.linspace(0, 6000, 3000)
X2, T2 = np.meshgrid(x, t)
 
sinT2 = np.sin(2*np.pi*T2/T2.max())
F = 0.9*sinT2*np.sinc(X2*(1 + sinT2))

#line = ax.plot(x, F[0, :], color='k', lw=2)[0]

# Plot the data
plt.scatter(data['article_length'], data['shares'], color='r', s=5.0)
# Label the axes
plt.xlabel('Article Length')
plt.ylabel('Shares')

def animate(i):
    ax.plot(plotdata['article_length'], plotdata['article_length']*(random.randrange(300, 700, 5)/100), color=colors[random.randrange(0, 5, 1)])
    plt.savefig("linestep"+str(i)+".png")

anim = FuncAnimation(
    fig, animate, interval=1, frames=1000)


 
plt.draw()
plt.show()


