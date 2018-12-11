import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import csv
df = pd.read_csv(r'D:\My Code\Drug200.csv')
df.head()

#reading the data.
my_data = pd.read_csv("Drug200.csv", delimiter=",")
my_data[0:5]

#Choosing columns
X = my_data[['Age', 'Sex', 'BP' , 'Cholesterol', 'Na_to_K']].values
X[0:5]

#converting categorical features to numerical values.
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

X[0:5]

#creating the target variable
y = my_data["Drug"]
y[0:5]

#splitting the data
from sklearn.model_selection import train_test_split

#train_test_split will return 4 different parameters, we will name them:
# X_trainset, X_testset, y_trainset, y_testset
#the train_test_split will need the parameters:
# X, y, test_size=0.3, and random_state=3
#the X and y are the arrays required before the split,
# the test_size represents the ratio of the testing dataset.
#the random_state ensures that we obtain the same splits.

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

#Modeling
#We will first create an instance of the DecisionTreeClassifier called drugTree.
#Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree

#Next, we will fit the data with
#the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset, y_trainset)

#Prediction.
#Let's make some predictions on the testing dataset and store it into a variable called predTree.
predTree = drugTree.predict(X_testset)

print(predTree[0:5])
print(y_testset[0:5])

#Evaluation
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#Visualization
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
