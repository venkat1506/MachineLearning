import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('College.csv',index_col=0)
print(data.head())

print(data.info())
print(data.describe())

# dara visulazition
#sns.lmplot(x='Room.Board',y="Grad.Rate",data=data,hue='Private')
#plt.show()

#sns.lmplot(x='Room.Board',y="Grad.Rate",data=data,hue='Private',fit_reg=False,palette='coolwarm',height=6,aspect=1)
#plt.show()


sns.lmplot(x='Outstate',y="F.Undergrad",data=data,hue='Private',fit_reg=False,height=6,aspect=1)
plt.show()


g=sns.FacetGrid(data,hue='Private',palette='coolwarm',height=6,aspect=2)
g=g.map(plt.hist,'Outstate',bins=15,alpha=0.6)
plt.show()

a=data[data['Grad.Rate']>100]
print(a)

# k mean cluster creation

from sklearn.cluster import KMeans

kmean=KMeans(n_clusters=3)

fit=kmean.fit(data.drop('Private',axis=1))
print(fit)

kmeancluster=kmean.cluster_centers_
print(kmeancluster)

# predicated label values
print(kmean.labels_)
print(kmean.inertia_)
print(kmean.n_iter_)

# Eveluation

def converter(cluster) :
    if cluster == 'Yes':
        return 1
    else:
        return 0


data['cluster']=data['Private'].apply(converter)

print(data['cluster'])

# confusion matrix

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(data['cluster'],kmean.labels_))
print(classification_report(data['cluster'],kmean.labels_))
