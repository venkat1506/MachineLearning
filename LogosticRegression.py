import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns

titanic=pd.read_csv("tatanic_train.csv")
print(titanic.head())
print(titanic.info())

print(titanic.isnull())
#see the missing values in heatmap
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

#sns.set_style('whitegrid')
#sns.countplot(x='Survived',data=titanic,hue='Sex')
#plt.show()

#sns.distplot(titanic['Age'].dropna(),kde=False,bins=30,color='green')
#plt.show()

#titanic['Age'].plot.hist(bins=20)
#plt.show()


#sns.countplot(x='SibSp',data=titanic)
#plt.show()

#titanic['Fare'].hist(bins=20,figsize=(10,4))
#plt.show()

sns.boxplot(x='Pclass',y='Age',data=titanic)
plt.show()

#handle missing values in Age column

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]

    if pd.isnull(Age):
        if Pclass ==1:
            return 37
        elif Pclass ==2:
            return 29
        else:
            return 24

    else:
        return Age

titanic['Age']=titanic[['Age','Pclass']].apply(impute_age,axis=1,)
plt.figure(figsize=(10,7))

sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()

#handle missing values in Cabin column

titanic.drop('Cabin',axis=1,inplace=True)
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


titanic.dropna(inplace=True)
print(titanic.head())


sex =pd.get_dummies(titanic['Sex'],drop_first=True)


embark=pd.get_dummies(titanic['Embarked'],drop_first=True)
print(embark.head())

titanic1=pd.concat([titanic,sex,embark],axis=1)

print(titanic1.head())

#onely geting numeric values

titanic.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
print(titanic.head(4))

titanic.drop(['PassengerId'],axis=1,inplace=True)
print(titanic.head())

print("test and train case ")

X=titanic.drop('Survived',axis=1)
Y=titanic['Survived']

print("\n")
print("check")

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=102)

from sklearn.linear_model import LogisticRegression

lm=LogisticRegression()
lm.fit(x_train,y_train)

print("prediction values")
prediction=lm.predict(x_test)


from sklearn.metrics import classification_report
a=classification_report(y_test, prediction)
print(a)

print("confusion matrix")
from  sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,prediction))
