import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(12345678)
x = np.random.random(10)
y = 1.6*x + np.random.random(10)

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print("slope: %f  \n intercept: %f" % (slope, intercept))
#To get coefficient of determination (r_squared):
print("r-squared: %f" % r_value**2)

plt.plot(x, y, 'o', label='original data')
plt.plot(x, intercept + slope*x, 'r', label='fitted line')
plt.legend()
plt.show()