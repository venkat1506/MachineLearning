import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

new=np.random.rand(10,3)
df = pd.DataFrame(new,columns=['first','second','Three'])
df.plot.box(color='red')
plt.grid()
plt.show()