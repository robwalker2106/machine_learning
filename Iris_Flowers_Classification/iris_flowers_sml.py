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
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

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
DATASET.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

#Histograms
DATASET.hist()
plt.show()

#Scatter Plot Matrix
scatter_matrix(DATASET)
plt.show()

#Split-out validation dataset
ARRAY = DATASET.values
X = ARRAY[:, 0:4]
Y = ARRAY[:, 4]
VALIDATION_SIZE = 0.20
SEED = 7
X_TRAIN, X_VALIDATION, Y_TRAIN, Y_VALIDATION = model_selection.train_test_split(X, Y, test_size=VALIDATION_SIZE, random_state=SEED)

#Test options and evaulation metric
SCORING = 'accuracy'

# Spot Check Algorithms
MODELS = []
MODELS.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
MODELS.append(('LDA', LinearDiscriminantAnalysis()))
MODELS.append(('KNN', KNeighborsClassifier()))
MODELS.append(('CART', DecisionTreeClassifier()))
MODELS.append(('NB', GaussianNB()))
MODELS.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
RESULTS = []
NAMES = []
for name, model in MODELS:
	kfold = model_selection.KFold(n_splits=10, random_state=SEED)
	cv_results = model_selection.cross_val_score(model, X_TRAIN, Y_TRAIN, cv=kfold, scoring=SCORING)
	RESULTS.append(cv_results)
	NAMES.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
FIG = plt.figure()
FIG.suptitle('Algorithm Comparison')
AX = FIG.add_subplot(111)
plt.boxplot(RESULTS)
AX.set_xticklabels(NAMES)
plt.show()

#Make predictions on validation dataset
KNN = KNeighborsClassifier()
KNN.fit(X_TRAIN, Y_TRAIN)
PREDICTIONS = KNN.predict(X_VALIDATION)
print(accuracy_score(Y_VALIDATION, PREDICTIONS))
print(confusion_matrix(Y_VALIDATION, PREDICTIONS))
print(classification_report(Y_VALIDATION, PREDICTIONS))
