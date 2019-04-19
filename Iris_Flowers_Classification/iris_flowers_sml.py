"""
This project uses Multiclass Classification to identify four different
species of Iris Flowers using four known properties.  It uses the Iris
Flower dataset and Supervised Machine Learning.
"""

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

#Load Dataset and set column names
NAMES = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
DATASET = pandas.read_csv('iris.csv', names=NAMES)

#Shape of the data.  There should be 150 instances and 5 attributes. (150,5)
print(DATASET.shape)

#Head.  This will show the first 20 rows of data.
print(DATASET.head(20))

#Description.  This shows a description of the attributes as well ass
#the count, mean, min, and ax values and some percentiles.
print(DATASET.describe())

#Class Descriptions.  Number of rows that belong to each class.
print(DATASET.groupby('class').size())

#Box and whisker plots
DATASET.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#Histograms
DATASET.hist()
plt.show()

#Scatter Plot Matrix
scatter_matrix(DATASET)
plt.show()
