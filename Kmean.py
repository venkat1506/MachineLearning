import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

# generation some artificial data
from sklearn.datasets import make_blobs

data=make_blobs(n_samples=200,
                n_features=2,
                centers=4,
                cluster_std=1.8,
                random_state=10)
# printing no of samples and features
print(data[0].shape)

plt.scatter(data[0][:,0],data[0][:,1])
plt.show()



# correct label values
print(data[1])


#plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
#plt.show()




from sklearn.cluster import KMeans

kmean =KMeans(n_clusters=8)    #also u change n_clusters=n nos

# data features
fitdata=kmean.fit(data[0])
print(fitdata)
# Actuval centers
clustercenter=kmean.cluster_centers_
print(clustercenter)

# predicated label values
print(kmean.labels_)


fig,(ax1,ax2)=plt.subplots(1,2,sharey=True,figsize=(10,6))

ax1.set_title('K Mean')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmean.labels_,cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
plt.show()


