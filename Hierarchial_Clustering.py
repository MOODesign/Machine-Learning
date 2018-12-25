import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

#generating random data using make_blobs.
#Input these parameters into make_blobs:
#n_samples: The total number of points equally divided among clusters.
#Choose a number from 10-1500

#centers: The number of centers to generate, or the fixed center locations.
#Choose arrays of x,y coordinates for generating the centers. Have 1-10 centers (ex. centers=[[1,1], [2,5]])

#cluster_std: The standard deviation of the clusters. The larger the number, the further apart the clusters
#Choose a number between 0.5-1.5

#save the result to X1 and y1.
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2,-1], [1,1],[10,4]], cluster_std=0.9)

#plot the scatter plot of the randomly generated data.
plt.scatter(X1[:, 0], X1[:,1], marker='o')
plt.show()

#------------------Agglomerative Clustering----------------------

#the agglomerative clustering class will require two imputs:
#n_clusters: The number of clusters to form as well as the number of centroids to generate.
#Value will be: 4

#linkage: Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.
#Value will be: 'complete'
#Note: It is recommended you try everything with 'average' as well

#save the rsult to a variable called agglom.
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')

#fitting the model with X1 and y1 from the generated data above.
agglom.fit(X1,y1)

#Now let's plot the Clustering.

#create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6,4))

#Scaling the data points down so they wont be scattered very far apart.
#creating a min and max range of X1.
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)
#getting the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min)

#this loop displays all of the datapoints.
for i in range(X1.shape[0]):
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]), color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

#removing the ticks and axis.
plt.xticks([])
plt.yticks([])
plt.axis('off')

#display the plot of the original data before clustering.
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
plt.show()

#---------Dendrogram Associated for the Agglomerative Hierarchical Clustering---------
#Remember that a distance matrix contains the distance from each point to every other point of a dataset. 
#Use the function distance_matrix, which requires two inputs. 
#Use the Feature Matrix, X2 as both inputs and save the distance matrix to a variable called dist_matrix 
#Remember that the distance values are symmetric, with a diagonal of 0's. 
#This is one way of making sure your matrix is correct. 
#(print out dist_matrix to make sure it's correct)

dist_matrix = distance_matrix(X1,X1)
print(dist_matrix)

#using the linkage class from heirarchy, pass in the parameters.
#save the result to a variable called Z.

Z = hierarchy.linkage(dist_matrix, 'complete')

#save the dendogram to a variable called dendro.
dendro = hierarchy.dendrogram(Z)


#----------Clutering a vehicle dataset-------------
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs
import csv
pdf = pd.read_csv(r'D:\My Code\cars_clus.csv')
print("Shape of dataset: ", pdf.shape)
pdf.head(5)

#Cleaning the data by dropping the rows that have NULL values.
print("Shape of dataset before cleaning: ", pdf.size)
pdf[['sales', 'resale', 'type', 'price', 'engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')

pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)

#selecting the feature set.
featureset = pdf[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

#normalizing the data.
#Now we can normalize the feature set. 
#MinMaxScaler transforms features by scaling each feature to a given range. 
#It is by default (0, 1). That is, 
#this estimator scales and translates each feature individually such that it is between zero and one.
from sklearn.preprocessing import MinMaxScaler
x = featureset.values
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx [0:5]

#-------------------clustering using scipy.-------------------------------------------------

#calculating teh distance between clusters using euclidean method.
import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

#clustering.
import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D, 'complete')

#Essentially, Hierarchical clustering does not require a pre-specified number of clusters. 
#However, in some applications we want a partition of disjoint clusters just as in flat clustering. 
#So you can use a cutting line:
from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
clusters

#you can also choose the number of clusters directly.
from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
clusters

#plotting the dendogram.
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )

dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')


#------------------------Clustering using scikit-learn------------------------------
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs
import csv
pdf = pd.read_csv(r'D:\My Code\cars_clus.csv')
print("Shape of dataset: ", pdf.shape)
pdf.head(5)

dist_matrix = distance_matrix(feature_mtx, feature_mtx)
print(dist_matrix)

#Ward minimizes the sum of squared differences within all clusters,
#It is a variance-minimizing approach and in this sense is similar to the k-means,
#objective function but tackled with an agglomerative hierarchical approach.

#Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
#Average linkage minimizes the average of the distances between all observations of pairs of clusters.


agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
agglom.labels_

pdf['cluster_'] = agglom.labels_
pdf.head()

#plotting.
import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
        plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25)
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)

plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')

#counting the number of cases in each group.
pdf.groupby(['cluster_','type'])['cluster_'].count()
#lookign at the characteristics of each group.
agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
agg_cars

#plotting.
plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')