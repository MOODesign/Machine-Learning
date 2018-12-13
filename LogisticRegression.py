import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

import csv
churn_df = pd.read_csv(r'D:\My Code\ChurnData.csv')
churn_df.head()

churn_df = pd.read_csv('ChurnData.csv')
churn_df.head()

#selecting features.
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]

#changing the target data type to integer , as it's required by sckitlearn algorithm.
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

#defining x and y.
x = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
x[0:5]

y = np.asarray(churn_df['churn'])
y[0:5]

#normalizing the dataset.
from sklearn import preprocessing
x = preprocessing.StandardScaler().fit(x).transform(x)
x[0:5]

#train/test split.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
print('Train set:', x_train.shape, y_train.shape)
print('Test set:', x_test.shape, y_test.shape)

#Modeling.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train, y_train) #c parameter indicates inverse of regularization strength.
LR

#predicting.
yhat = LR.predict(x_test)
yhat

#predicting the probability, using yhat_prob.
#first column is the probability of y=1 and the second column is the probability of y=0.
yhat_prob = LR.predict_proba(x_test)
yhat_prob

#Evaluating using jaccard index.
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

#Evaluating using a confusion matrix.
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matric, without normalization")

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print(confusion_matrix(y_test, yhat, labels=[1,0]))

#computing confusion matrix

cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

#plotting non-normalized confusion matrix.
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'], normalize= False, title='Confusion matrix')
plt.show()
print(classification_report(y_test, yhat))

#Evaluating using log loss.
from sklearn.metrics import log_loss
print("LogLoss1: : %.2f" % log_loss(y_test, yhat_prob))

#Modeling again using different solver and regularization values.
LR2 = LogisticRegression(C=0.01, solver='sag').fit(x_train, y_train)
yhat_prob2 = LR2.predict_proba(x_test)
print("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))
