import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

expen=[250,100,50,300,175,125]
ex_label=["rend","food","travel","cloths","medicin","loan"]
plt.pie(expen,labels=ex_label,autopct='%0.2f%%',radius=1,shadow=True,explode=[0,0,0,0.1,0,0],startangle=180)
plt.show()