import numpy as np
import matplotlib.pyplot as plt

x=[i for i in range(1,21)]
y = [1.0*i/15 for i in range(1,13)]

y.append(517*1.0/600)
y.append(517*1.0/600)
y.append(557*1.0/600)
y.append(557*1.0/600)
y.append(598*1.0/600)
y.append(598*1.0/600)
y.append(598*1.0/600)
y.append(597*1.0/600)

plt.plot(x,y,'g-')
plt.xlabel('k')
plt.ylabel('Cluster purity')
plt.xticks(x)
plt.show()
