#Most of the traditional clustering techniques, such as k-means, 
#hierarchical and fuzzy clustering, can be used to group data without supervision.
#However, when applied to tasks with arbitrary shape clusters, or clusters within cluster, 
#the traditional techniques might be unable to achieve good results. 
#That is, elements in the same cluster might not share enough similarity or the performance may be poor. 
#Additionally, Density-based Clustering locates regions of high density that are separated from one another
#by regions of low density. Density, in this context, 
#is defined as the number of points within a specified radius.
#In this section, the main focus will be manipulating the data and properties of DBSCAN 
#and observing the resulting clustering.

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Generating the data using these inputs:
#centroidLocation: Coordinates of the centroids that will generate the random data.
#numSamples: The number of data points we want generated, 
#split over the number of centroids (# of centroids defined in centroidLocation)
#clusterDeviation: The standard deviation between the clusters. 
#The larger the number, the further the spacing.

def createDataPoints(centeroidLocation, numSamples, clusterDeviation):
    X, y = make_blobs(n_samples=numSamples, centers=centeroidLocation, cluster_std=clusterDeviation)

    X = StandardScaler().fit_transform(X)
    return X, y

#use createDataPoints with the 3 imputs and store the output into variables X and y.
X, y = createDataPoints([[4,3], [2,-1], [-1,4]] , 1500, 0.5)


#Modeling:
#DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.
#This technique is one of the most common clustering algorithms which works based on density of object.
#The whole idea is that if a particular point belongs to a cluster, 
#it should be near to lots of other points in that cluster.
#It works based on two parameters: Epsilon and Minimum Points:
#Epsilon >> determine a specified radius that if includes enough number of points within,
#we call it dense area
#minimumSamples >> determine the minimum number of data points we want in a neighborhood to define a cluster.

epsilon = 0.3
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_
labels

#Distinguishing Outliers.
#Lets Replace all elements with 'True' in core_samples_mask that are in the cluster, 
#'False' if the points are outliers.
#First, create an array of booleans using the labels from db.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
core_samples_mask
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_
# Remove repetition in labels by turning it into a set.
unique_labels = set(labels)
unique_labels

#Visualizing the data.
# Create colors for the clusters.
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors
# Plot the points with colors
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    # Plot the datapoints that are clustered
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=col, marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1],s=50, c=col, marker=u'o', alpha=0.5)

plt.show()

#Clustering the same dataset into 3 clusters using k-means.
from sklearn.cluster import KMeans
k_means = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_labels
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers

fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
ax = fig.add_subplot(1, 1 , 1)

for k, col in zip(range(len([[4,4],[-2,-1], [2,-3], [1,1]])), colors):
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.show()

#OR this way.
from sklearn.cluster import KMeans 
k = 3
k_means3 = KMeans(init = "k-means++", n_clusters = k, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(k), colors):
    my_members = (k_means3.labels_ == k)
    plt.scatter(X[my_members, 0], X[my_members, 1],  c=col, marker=u'o', alpha=0.5)
plt.show()

