import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
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
import numpy as np

# Load dataset
dataset = pandas.read_csv("churn-bigml-802.csv")
array = dataset.values
X = array[:,1:18]
Y = array[:,19]
dataset2 = pandas.read_csv("churn-bigml-201.csv")
array = dataset2.values
X2 = array[:,1:18]
Y2 = array[:,19]

dataset.plot(kind='box', subplots=True, layout=(5,4), sharex=False, sharey=False)
plt.show()
# histograms
dataset.hist()
plt.show()

#Creating training data and testing data
X_train=X
Y_train=Y
X_validation=X2
Y_validation=Y2
X_train=X.astype(float)
Y_train=Y.astype(int)
X_validation=X2.astype(float)
Y_validation=Y2.astype(int)

seed=7
scoring = 'accuracy'
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

#Confusion Matrix
print(confusion_matrix(Y_validation, predictions))

print(classification_report(Y_validation, predictions))
