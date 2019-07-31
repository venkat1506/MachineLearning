import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Blood_Suger=[139,140,85,98,75,110,120,128,93]
Women_BS=[125,85,65,77,110,135,125,86,]
plt.xlabel("suger Range",color='brown')
plt.ylabel("patient list")
plt.title("BP Status")
plt.hist([Blood_Suger,Women_BS],bins=[80,105,125,140],rwidth=0.80,color=['green','red'],label=['men','women'])
         #,orientation='horizontal'
plt.legend()
plt.show()
