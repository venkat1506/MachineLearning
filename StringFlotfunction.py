import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
A=[10,11,12,13,14,15]
B=[34,52,47,22,45,28]
plt.xlabel("year")
plt.ylabel("Temperature")
plt.title("weather")
#plt.plot(A,B, 'Dr--',markersize=15,)
plt.plot(A,B,color='green',alpha=0.2)  #alpha is use in transference purpose
plt.show()