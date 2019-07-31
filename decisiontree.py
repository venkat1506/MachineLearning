import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.preprocessing import LabelEncoder
import pydotplus
import os
import graphviz


os.environ["path"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin;'

col_names = ['day', 'weather', 'temperature', 'humidity','wind', 'play']
data = pd.read_csv("weather.csv", header=0, names=col_names)
print(data.head())

print("splitting data into features and target variables")
feature_cols = ['day', 'weather', 'temperature', 'humidity', 'wind']
x = data[feature_cols]
y = data.play
u = data.iloc[:, [0,1,2,3,4]].values # Here first : means fetch all rows :-1 means except last column

v = data.iloc[:, [5]].values # : is fetch all rows 5 means 5th column
print(u)
print(v)

labelencoder_X = LabelEncoder()
u[:,1] = labelencoder_X.fit_transform(u[:,1])# All rows and first column i.e weather column
u[:,2] = labelencoder_X.fit_transform(u[:,2])
u[:,3] = labelencoder_X.fit_transform(u[:,3])
u[:,4] = labelencoder_X.fit_transform(u[:,4])
print(u)

labelencoder_y = LabelEncoder()

v= labelencoder_y.fit_transform(v)

print(v)
#print("splitting dataset into training set and test set")

x_train, x_test, y_train, y_test = train_test_split(u, v, random_state=0)
#print("creating decision tree classifier object")
clf = DecisionTreeClassifier(criterion="entropy")
#print("training decision tree classifier")

clf = clf.fit(x_train, y_train)
#print("predicting the response for test dataset")
y_pred = clf.predict(x_test)
#print("claculating model accuracy how often is the classifier correct")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print()
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, 
                feature_names=feature_cols,
                class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('weather.png')
Image(graph.create_png())