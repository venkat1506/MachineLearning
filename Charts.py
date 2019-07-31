import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,10)
y = x ^ 2
#Simple Plot
plt.xlabel("range")
plt.ylabel("Distance")
plt.title("Chat diagram")
plt.plot(x,y)
plt.show()