import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np


infile = open("E:\WorkSpace\MatPlotLib\Project_Emp.pkl",'rb')
new_dict = pickle.load(infile)
infile.close()
print(new_dict)
#print(new_dict.info)

data=pd.DataFrame(list(new_dict))
data.columns = ['Deptname','gender','id','age','jdate']

print(data)
print(data.info())
print(data.describe())

gender = {'MALE': 1, 'FEMALE': 2}
data.gender = [gender[item] for item in data.gender]
print(data)


data['gender'].value_counts().sort_index().plot.bar()
plt.title("Gender Count bar graph")
plt.xlabel("male:1 Female:2")
plt.ylabel("count")
plt.show()


df = data[(data['age'] < 28)]
sns.boxplot('jdate', 'age', data=df)
plt.show()


sns.distplot(data['age'])
plt.show()

sns.heatmap(data.corr(), annot=True)
plt.show()

X = data[['gender', 'age']]
Y = data['id']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101)

lm = LinearRegression()
lm.fit(x_train, y_train)

print(lm.intercept_)
print("coefficent ", lm.coef_)
print(x_train.columns)

# prediction finding
prediction = lm.predict(x_test)
print("prediction", prediction)
print(y_test)

plt.scatter(y_test, prediction)
plt.show()

# creating didtplot for y-test data and Prediction
sns.distplot(y_test - prediction)
plt.show()

# finding MAE & MSE & MSLE
MAE = metrics.mean_absolute_error(y_test, prediction)
MSE = metrics.mean_squared_error(y_test, prediction)
MSLE = metrics.mean_squared_log_error(y_test, prediction)
print("MEAN ABSOULET ERROR", MAE, "\n"
                                  "mean squared error", MSE, "\n"
                                                             "mean squred log error", MSLE)
MSESQRT = np.sqrt(metrics.mean_squared_error(y_test, prediction))
print("sqrt of mean sqred error", MSESQRT)




expvariancescore = metrics.explained_variance_score(y_test, prediction)
print("variance score ", expvariancescore)





from sklearn.cluster import KMeans

kmean=KMeans(n_clusters=2)
a=data.drop('jdate',axis=1)
fit=kmean.fit(a.drop('Deptname',axis=1))
print(fit)

kmeancluster=kmean.cluster_centers_
print(kmeancluster)

# predicated label values
print(kmean.labels_)
print(kmean.inertia_)
print(kmean.n_iter_)

# Eveluation

def converter(cluster) :
    if cluster == 'FEMALE':
        return 1
    else:
        return 0


data['cluster']=data['gender'].apply(converter)

print(data['cluster'])

# confusion matrix

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(data['cluster'],kmean.labels_))
print(classification_report(data['cluster'],kmean.labels_))



import fpdf

Emp_project_Output=classification_report(data['cluster'],kmean.labels_)

pdf = fpdf.FPDF(format='letter')
pdf.add_page()
pdf.set_font("Arial", size=10)

str= ""

for i in Emp_project_Output:
    str=str+i
    str=str+ " "
    pdf.ln()

pdf.write(2,str)
pdf.output("Employee_Kmean_out.pdf")


