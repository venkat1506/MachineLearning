import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

A=[1,2,3,4,5,6,7]
max_temp=[50,51,48,47,49,48,46]
min_temp=[43,42,40,44,33,35,37]
avg_temp=[45,48,44,42,47,30,46]
plt.xlabel("days")
plt.ylabel("Temperature")
plt.title("Weather condition ")

plt.plot(A,max_temp,label="MAX")
plt.plot(A,min_temp,label="MIN")
plt.plot(A,avg_temp,label="AVG")
plt.legend(loc='lower left',shadow=True,fontsize='large')
plt.annotate(xy=[2,50],s='maximum weather')
plt.annotate(xy=[5,33],s='min weather')
plt.annotate(xy=[5,39],s='avg weather')
plt.grid()
plt.show()