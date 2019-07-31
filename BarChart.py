import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

company=["Google","microsoft","oracle","gresstech"]
revinue=[75,80,84,60]
prof=[44,56,22,18]
com=np.arange(len(company))
#this is for we create two bar charts side by side so its useful other wise you get error
plt.xticks(com,company)
plt.ylabel("Revinue(bin)",color="red")
plt.title("Us Tech stoke",color="red")
plt.bar(com-0.2,revinue,width=0.4,label="REv")
plt.bar(com+0.2,prof,width=0.4,label="PROFF")
plt.legend()
plt.show()
