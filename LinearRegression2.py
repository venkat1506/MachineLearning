import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("USA_Housing.csv")
print(data.head())
print(data.info())
print(data.describe())
print(data.columns)

#sns.pairplot(data)
#plt.show()
sns.distplot(data['Price'])
plt.show()
sns.heatmap(data.corr(),annot=True)
plt.show()

X=data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

Y=data['Price']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,random_state=101)

from sklearn.linear_model import LinearRegression

lm=LinearRegression()
lm.fit(x_train,y_train)

print(lm.intercept_)
print("coefficent ",lm.coef_)

print(x_train.columns)

from sklearn.datasets import load_boston

boston=load_boston()
boston.keys()
print(boston['feature_names'])
print(boston['target'])

#prediction finding
prediction=lm.predict(x_test)
print(prediction)
print(y_test)
#actuvall price

plt.scatter(y_test,prediction)
plt.show()

sns.distplot(y_test-prediction)
plt.show()

print("lenear regression Evaluation methods ")


from sklearn import metrics
MAE=metrics.mean_absolute_error(y_test,prediction)
MSE=metrics.mean_squared_error(y_test,prediction)
MSLE=metrics.mean_squared_log_error(y_test,prediction)

print("MEAN ABSOULET ERROR"  ,MAE,"\n"
      "mean squared error"   ,MSE,"\n"
      "mean squred log error",MSLE)
MSESQRT=np.sqrt(metrics.mean_squared_error(y_test,prediction))
print("sqrt of mean sqred error",MSESQRT)

expvariancescore=metrics.explained_variance_score(y_test,prediction)
print("variance score ",expvariancescore)


cdf=pd.DataFrame(lm.coef_,X.columns,columns=['coeff'])
print(cdf.head())
