import zipfile
with zipfile.ZipFile("D:\My Code\moviedataset.zip","r") as zip_ref:
    zip_ref.extractall("D:\My Code")
print('unzipping ...')

import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

#Reading the Data.
#Storing the movie information into a pandas dataframe.
movies_df = pd.read_csv('movies.csv')
#Storing the user information into a pandas dataframe.
ratings_df = pd.read_csv('ratings.csv')
#Head is a function that gets the first N rows of a dataframe, N's default is 5.
movies_df.head()

#let's remove the year from teh title column by using pandas replace function,
#and store the year in a new column.
#we specify the parantheses so we dont conflict with movies that have years in their title.
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
#removing the parantheses.
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
movies_df.head()

#let's split the values in the Geners column into a list of Geners to simplfy future us,
#this can be achieved by applying Python's split string function on the correct column.
movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()

#we will use the One Hot Encoding technique to convert the list of genres to a vector where,
#each column corresponds to one possible value of the feature. 
#This encoding is needed for feeding categorical data. 
#In this case, we store every different genre in columns that contain either 1 or 0. 
#1 shows that a movie has that genre and 0 shows that it doesn't. 
#Let's also store this dataframe in another variable since genres won't be important,
#for our first recommendation system.

##Copying the movie dataframe into a new one since we won't need to use the genre information in our first case.
moviesWithGenres_df = movies_df.copy()

#For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column.
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1

#Filling in the NaN values with 0 to show that a movie doesn't have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()

#let's look at the rating dataframe.
ratings_df.head()

#Every row in the ratings dataframe has a user id associated with at least one movie, 
#a rating and a timestamp showing when they reviewed it. 
#We won't be needing the timestamp column, so let's drop it to save on memory.
#Drop removes a specified row or column from a dataframe.
ratings_df = ratings_df.drop('timestamp', 1)
ratings_df.head()

#Content Based Recommendation System.
#Let's begin by creating an input user to recommend movies to:
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]
inputMovies = pd.DataFrame(userInput)
inputMovies

#add movie ID to input user.
#We can achieve this by first filtering out the rows that contain the input movie's, 
#title and then merging this subset with the input dataframe. 
#We also drop unnecessary columns for the input to save memory space.

#filtering out the movies by title.
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
#merging it so we can get the moviesID.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe.
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
inputMovies

#To learn the inputs preferences , we get the subset of movies that the input has watched from the dataframe.
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies

#Let's clean this up by resseting the index and dropping the moviesId, genres and year columns.
#resetting the index to avoid future issues.
userMovies = userMovies.reset_index(drop=True)
#dropping unnecessary columns to save memory.
userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
userGenreTable

#Now, we are ready to learn the input's preferences.
#To do this , we are going to turn each genre into weights,
#by using the inpyt's revies and multiplying then into the input's genre table and then summing up the table.
#This operation is called a dot product between a matrix and a vector,
#so we can simply called Panda's "dot" function.
inputMovies['rating']
userProfile = userGenreTable.transpose().dot(inputMovies['rating'])
userProfile

#Now we can recommend movies.
#let's extract the genre table from the original dataframe.
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
#dropping the unnecessary information.
genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)
genreTable.head()
genreTable.shape

#With the input's profile and the complete list of movies and their genres in hand, 
#we're going to take the weighted average of every movie based on the input profile and,
#recommend the top twenty movies that most satisfy it.
#Multiply the genres by the weights and then take the weighted average
recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()

#sorting our recommendations in descending order.
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
recommendationTable_df.head()

#The final recommendation table........
print(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())])
