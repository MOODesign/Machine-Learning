#importing needed packages.
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#download the data.
!wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

df = pd.read_csv("FuelConsumption.csv")
#reading the data.
df.head()

#summarizing the data.
df.describe()

#selecting some features to explore more.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

#plotting each of these features.
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()

#plotting each of these features vs the emission to see how are they linearly related.
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


#creating train and test datasets.
#splitting the data set into 80% training set and 20% testing set.
#we create a mask to select random rows using np.random.rand() function.
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cfd[msk]

#distributing the training data.
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.xlabel("Engine size")
plt.ylabel("Emissions")
plt.show()

#Modeling the data using sklearn package.
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
#the coefficients.
print ('Coefficients: ', regr.coef_)
print ('Intercept: ', regr.intercept_)

#plotting the outputs.
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine seize")
plt.ylabel("Emissions")

#evaluating the accuracy of the model using MSE(mean squared error).
from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(npabsolute(test_y_hat - test_y)))
print("Residual sum of squares(MSE)): %.2f" % np.mean((test_y_hat - test_y) ** 2)
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))
