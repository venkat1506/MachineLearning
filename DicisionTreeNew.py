import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Kyphosis.csv')
print(df.head())

df.info()

sns.pairplot(df,'Kyphosis')
plt.show()


from sklearn.model_selection import train_test_split
X=df.drop('Kyphosis',axis=1)
Y=df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.45, random_state=42)




from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)
prediction=dtree.predict(X_test)



from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction))
print("\n")
print(classification_report(y_test,prediction))





from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rfc_prediction=rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_prediction))
print("\n")
print(classification_report(y_test,rfc_prediction))
print(df['Kyphosis'].value_counts())





from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

features=list(df.columns[1:])
print(features)

dot_data=StringIO()
export_graphviz(dtree,
                out_file=dot_data,
                feature_names=features,
                filled=True,
                rounded=True)

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())

#graph.write_png('kyphos.png')
Image(graph.create_png())
