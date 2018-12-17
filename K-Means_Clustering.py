import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#creating a dataset using a random seed.
np.random.seed(0)

#making random clusters of points by using make_blobs class.
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2,1], [2, -3], [1,1]], cluster_std=0.9)

#display the scatter plt of the randomly generated data.
plt.scatter(X[:, 0], X[:, 1], marker='.')

#setting up k-Means.
#initializing k-means.
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

#fitting k-means model with the feature matrix X.
k_means.fit(X)

#grab the labels for each point in the model using k-means.labels_
k_means_labels = k_means.labels_
k_means_labels

#get the coordinates of the cluter centers using kmeans.cluster_centers_
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

#plotting.
#Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))
#colors uses a color map , which will produce an array of colors based on
#the number of labels there are.
#We use set(k_means_labels) to get the unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
#create a plot.
ax = fig.add_subplot(1, 1 , 1)
#create a for loop that plot the data points and centiroids.
#k will range from 0-3, which will match the possible clusters that each data point is in.
for k, col in zip(range(len([[4,4],[-2,-1], [2,-3], [1,1]])), colors):
    #create a list of all data points where teh data points that are in the cluster are labeled as true , else false.
    my_members = (k_means_labels == k)
    #define the centiroid.
    cluster_center = k_means_cluster_centers[k]
    #plot the datapoints with color.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    #plot teh centiroids with the specified color, but with a darker outline.
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

#title of the plot.
ax.set_title('KMeans')
#remove x-axis ticks.
ax.set_xticks(())
#remove y-axis ticks.
ax.set_yticks(())
#show the plot.
plt.show()

#-----------------Customer segmentation with K-Means----------------------

import pandas as pd
import csv
cust_df = pd.read_csv(r"D:\My Code\Cust_Segmentation.csv")
cust_df.head()

#Dropping the Address column beacause it has a categorical values
df = cust_df.drop('Address', axis=1)
df.head()

#normalizing the dataset using StandardScaler().
#Normalization is a statistical method that helps mathematical-based algorithms 
#to interpret features with different magnitudes and distributions equally.
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataset = StandardScaler().fit_transform(X)
Clus_dataset

#Modeling using K-means algorithm.
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

#Assigning the labels to each row in the dataframe.
df["Clus_km"] = labels
df.head(5)

#cheking the centiroids values by averaging the features in each cluster.
df.groupby('Clus_km').mean()

#plotting the distribution of customers based on their age and income.
area = np.pi * ( X[:,1])**2
plt.scatter(X[:,0],X[:,3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

#plotting the 3 clusters in 3d.
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
plt.show()