import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

x = np.arange(0,10)
y = x ^ 2
z = x ^ 3
t = x ^ 4
print(x)
print(y)
print(z)
print(t)

plt.title("Graph Drawing")
plt.xlabel("Time")
plt.ylabel("Distance")
plt.plot(x,y)
plt.plot(x,t)
plt.plot(x,z)

#plt.plot(x,y,'>')
#plt.savefig('exout.pdf',format='pdf')

plt.annotate(xy=[0,2], s='Second Entry')
plt.annotate(xy=[8,12], s='Third Entry')
plt.annotate(xy=[4,6], s='first column')

plt.legend(['REC1','REC2','REC3'],loc=4)
#Style the background
plt.style.use('fast')
plt.show()