import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import csv
cell_df = pd.read_csv(r'D:\My Code\cell_samples.csv')
cell_df.head()

ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

#checking columns data types.
cell_df.dtypes

#some rows in BareNuc contains non-numerical values , let's drop those rows.
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

#selecting features.
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]

#Changing the measurment of the Class column so it can predict one of 2 values.
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
y[0:5]

#splitting teh data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('Train set:' , X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

#Modeling.(using Radial Basis Function(RBF))
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

#predicting new values.
yhat = clf.predict(X_test)
yhat[0:5]

#Evaluation. (using confusion matrix)
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("confusion matrix, without normalization")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i,j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#compute confusion matrix.
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat))

#plot non-normalized confusion matrix.
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize = False, title='Confusion matrix')

#Evaluation. (using F1 score).
from sklearn.metrics import f1_score
print("F1_score: %.2f" % f1_score(y_test, yhat, average='weighted'))

#Evaluation, (using jaccard index).
from sklearn.metrics import jaccard_similarity_score
print("Jaccard Index: %.2f" % jaccard_similarity_score(y_test, yhat))


#Modeling.(using linear kernel)
from sklearn import svm
clf2 = svm.SVC(kernel='linear')
clf2.fit(X_train, y_train)

yhat2 = clf2.predict(X_test)
print("F1_score(2): %.2f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard Index(2): %.2f" % jaccard_similarity_score(y_test, yhat2))