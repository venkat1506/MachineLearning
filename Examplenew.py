import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import graphviz


col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
data = pd.read_csv("E:\diabetes.csv", header=1, names=col_names)
print(data.head())
print("splitting data into features and target variables")
feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
x = data[feature_cols]
y = data.label
print("splitting dataset into training set and test set")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
print("creating decision tree classifier object")
clf = DecisionTreeClassifier()
print("training decision tree classifier")
clf = clf.fit(x_train, y_train)
print("predicting the response for test dataset")
y_pred = clf.predict(x_test)
print("claculating model accuracy how often is the classifier correct")
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print()
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols,
                class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
