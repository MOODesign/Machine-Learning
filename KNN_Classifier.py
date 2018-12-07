import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing


#importing the dataset.
import csv
#reading the dataset.
df = pd.read_csv(r'D:\My Code\Machine Learning\teleCust1000t.csv')
df.head()

#to see how many of each class is in our dataset.
df['custcat'].value_counts()

#visualizing the data.
df.hist(column='income', bins=50)

#defining feature set, X.
df.columns
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]

#defining label set, Y.
y = df['custcat'].values
y[0:5]

#normalizing the data.
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#splitting teh data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

#importing KNN library.
from sklearn.neighbors import KNeighborsClassifier

#training the algorithm with k=4.
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

#testing on the test set.
yhat = neigh.predict(X_test)
yhat[0:5]

#evaluating the accuracy.
from sklearn import metrics
print("Train set Accuracy:", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy:", metrics.accuracy_score(y_test, yhat))

#calculating the accuracy of KNN for different K's.
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

#plotting the accuracy for different number of neighbors.
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc, mean_acc +1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '=/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors(K) ')
plt.tight_layout()
plt.show()

#choosing the highest accuracy.
print("The best accuracy was with", mean_acc.max(), "with k=" ,mean_acc.argmax()+1)

